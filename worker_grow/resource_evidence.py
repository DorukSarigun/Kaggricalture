"""Conservative evidence for a sale reducing already-open feeding demand.

Not an inventory reservation or a sales mask. Only executed wheat sales are
eligible. Newly opened jobs cannot inherit blame for an earlier sale. Wheat
that exists during the unit phase retires withdrawal evidence FIFO; satisfied
or cancelled demands retire it too. Market purchases arrive after units, so
they restore blame only at the next crew phase. Same-turn SELL cannot explain
a missed earlier unit-phase FEED.
"""
from lifecycle_tracking import LifecycleEventType as E
from kaggle_environments.envs.kaggriculture import kaggriculture as game


def wheat_stock(observation):
    private = observation['private']
    return int(private['shed'].get('WHEAT', 0)) + sum(int(b.get('WHEAT', 0))
                                                    for b in private['inventories'])


class FeedWithdrawalEvidence:
    def __init__(self):
        self.withdrawals = []
        self.history = []
        self.last_seen_stock = None
        self._deferred_restore = 0

    def _restore(self, quantity, step, event_id):
        remaining = quantity
        for entry in self.withdrawals:
            restored = min(remaining, entry['quantity'])
            if restored:
                entry['quantity'] -= restored
                remaining -= restored
                self.history.append({'kind': 'resource_restored', 'step': step,
                                     'event_id': event_id, 'sale_event_id': entry['event_id'],
                                     'quantity': restored})

    def observe(self, before, after, book, events):
        step = int(before['step'])
        stock = wheat_stock(before)
        if self.last_seen_stock is not None and stock != self.last_seen_stock:
            self.withdrawals.clear()
            self._deferred_restore = 0
            self.history.append({'step': step, 'kind': 'unexplained_stock_change'})
        elif self._deferred_restore:
            restored = min(self._deferred_restore, stock)
            if restored:
                self._restore(restored, step, None)
            self._deferred_restore = 0
        owners = {r.producer_id: key for key, r in book.task_book.records.items() if r.producer_id}
        lost_now = {e.animal_id for e in events if e.event_type == E.ANIMAL_LOST}
        hungry = set()
        for identity, key in owners.items():
            record = book.task_book.records[key]
            if record.task.item not in game.ANIMALS or record.tile_xy is None:
                continue
            if not record.needs_production and identity not in lost_now:
                continue
            x, y = record.tile_xy
            tile = before['farms'][int(before['player'])]['tiles'][y][x]
            if isinstance(tile, dict) and tile.get('animal') == record.task.item and not tile.get('fed_today'):
                hungry.add(key)
        # No implicit future feeding demand: only presently hungry producers.
        for entry in self.withdrawals:
            entry['eligible_issues'] = [k for k in entry['eligible_issues'] if k in hungry]
        self.withdrawals = [w for w in self.withdrawals if w['eligible_issues']]
        fed = set()
        for event in events:
            kind = event.event_type
            incoming = 0
            if kind == E.ANIMAL_PLACED and event.animal_id in owners:
                hungry.add(owners[event.animal_id])
            elif kind == E.ANIMAL_FED:
                key = owners.get(event.animal_id)
                if key is not None:
                    hungry.discard(key); fed.add(key)
                stock -= sum(int(a['quantity']) for a in event.data.get('consumed_allocations', ()))
            elif kind == E.HARVESTED and event.crop == 'WHEAT':
                incoming = int(event.quantity)
            elif kind == E.PRODUCT_PURCHASED and event.data.get('item') == 'WHEAT':
                bought = int(event.quantity)
                stock += bought
                self._deferred_restore += bought
                self.history.append({'kind': 'market_wheat_after_units', 'step': step,
                                     'event_id': event.event_id, 'quantity': bought})
            elif kind == E.PRODUCT_SOLD and event.crop == 'WHEAT':
                quantity = int(event.quantity)
                next_stock = stock - quantity
                removed_need = max(0, len(hungry) - next_stock) - max(0, len(hungry) - stock)
                if (removed_need > 0 and event.decision_id is not None
                        and event.decision_id % 2 == 0
                        and event.data.get('revenue_is_estimate') is False):
                    entry = {'event_id': event.event_id, 'decision_id': event.decision_id // 2,
                             'step': step, 'quantity': removed_need,
                             'eligible_issues': sorted(hungry),
                             'sold_quantity': quantity, 'stock_before': stock, 'stock_after': next_stock,
                             'source_lot_ids': [a['product_lot_id'] for a in event.data['allocations']]}
                    self.withdrawals.append(entry)
                    self.history.append({'kind': 'demand_resource_sold', **entry})
                stock = next_stock
            elif kind == E.INVENTORY_RECONCILED and event.crop == 'WHEAT':
                if event.data.get('source') == 'external_or_unobserved':
                    incoming = int(event.quantity)
                elif event.data.get('destination') == 'lost_or_consumed':
                    stock -= int(event.quantity)
            if incoming:
                stock += incoming
                self._restore(incoming, step, event.event_id)
        for entry in self.withdrawals:
            entry['eligible_issues'] = [k for k in entry['eligible_issues'] if k not in fed]
        self.withdrawals = [w for w in self.withdrawals if w['quantity'] > 0 and w['eligible_issues']]
        actual = wheat_stock(after)
        if stock != actual:
            # Attribution must not be manufactured from an incomplete trace.
            self.withdrawals.clear()
            self._deferred_restore = 0
            self.history.append({'step': step, 'kind': 'unexplained_stock_change',
                                 'projected_stock': stock, 'actual_stock': actual})
        self.last_seen_stock = actual

    def allocate(self, cost_events, step):
        """One withdrawn unit cannot explain two unmet feeds in the same turn."""
        remaining = {i: entry['quantity'] for i, entry in enumerate(self.withdrawals)}
        for cost in cost_events:
            if cost['reason'] != 'missed_feed':
                continue
            for i, entry in enumerate(self.withdrawals):
                if (remaining[i] > 0 and entry['step'] < step
                        and cost['issue_id'] in entry['eligible_issues']):
                    cost['withdrawal_evidence'] = [{**entry, 'quantity': 1}]
                    remaining[i] -= 1
                    break
