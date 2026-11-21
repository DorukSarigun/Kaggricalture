from copy import deepcopy
import math
import unittest

from kaggle_environments import make
from cash_accounting import CashEntry, CashLedger
from lifecycle_tracking import FarmLifecycleTracker, LifecycleEvent, LifecycleEventType as E
from production_graph_credit import ActionDAG, GraphCreditConfig, assign_graph_credit
from market_execution import trace_market_executions


def play_traced(env, tracker, script, configuration, ledger):
    refs, decision = {}, 0
    for step, (unit, market) in enumerate(script):
        observations = [deepcopy(s.observation) for s in env.state]
        for obs in observations:
            obs['step'] = step
        action = {'farmer': unit, 'hands': [], 'market': market}
        actions = [action, {'farmer': ['PASS'], 'hands': [], 'market': []}]
        executions = trace_market_executions(observations, actions, configuration)
        env.step(actions)
        after = deepcopy(env.state[0].observation)
        after['step'] = step+1
        local = {('unit', 0): decision}
        decision += 1
        for index in range(len(market)):
            local[('market', index)] = decision
            decision += 1
        refs.update({(step, *key): value for key, value in local.items()})
        tracker.observe_transition(observations[0], action, after,
                                   market_executions=executions[0], decision_refs=local)
        ledger.observe(step, executions, [f['money'] for f in observations[0]['farms']],
                       [f['money'] for f in after['farms']])
    return refs


class Journal:
    def __init__(self):
        self.events, self.cash, self.refs = [], [], {}

    def add(self, kind, *, quantity=None, data=None, **kwargs):
        i = len(self.events)
        e = LifecycleEvent(event_id=f"e{i}", event_type=kind, episode=1, step=i,
                           day=0, hour=i, player=0, decision_id=i, quantity=quantity,
                           data=data or {}, **kwargs)
        self.events.append(e)
        return e

    def buy(self, item, quantity, cost, kind="seed", ids=None):
        i = len(self.events)
        ids = ids or [f"lot{i}_{j}" for j in range(quantity)]
        if kind == "seed":
            e = self.add(E.SEED_PURCHASED, quantity=quantity, crop=item, seed_lot_id=ids[0],
                         data={"total_cost": cost})
        elif kind == "animal":
            e = self.add(E.ANIMAL_PURCHASED, quantity=quantity,
                         data={"total_cost": cost, "animal": item, "animal_lot_ids": ids})
        else:
            e = self.add(E.PRODUCT_PURCHASED, quantity=quantity,
                         data={"total_cost": cost, "item": item, "product_lot_ids": ids,
                               "unit_prices": [cost/quantity]*quantity})
        op = {"seed": "BUY_SEED", "animal": "BUY_ANIMAL", "product": "BUY_PRODUCT"}[kind]
        self.cash.append(CashEntry(e.step, 0, 0, op, item, quantity, -cost))
        self.refs[(e.step, "market", 0)] = e.decision_id
        return ids

    def plant(self, seed, identity="p"):
        return self.add(E.PLANTED, plant_id=identity, seed_lot_id=seed, crop="WHEAT", quantity=1,
                        tile_xy=(1, 1), data={"unit_index": 0})

    def water(self, identity="p"):
        return self.add(E.WATERED, plant_id=identity, data={"unit_index": 0})

    def lose_plant(self, identity="p"):
        return self.add(E.PLANT_LOST, plant_id=identity, data={"unit_index": 0})

    def feed_animal(self, identity="a"):
        return self.add(E.ANIMAL_FED, animal_id=identity, data={"unit_index": 0, "consumed_allocations": []})

    def lose_animal(self, identity="a"):
        return self.add(E.ANIMAL_LOST, animal_id=identity, data={"unit_index": 0})

    def animal(self, stock, identity="a"):
        return self.add(E.ANIMAL_PLACED, animal_id=identity, animal_lot_id=stock, quantity=1,
                        tile_xy=(2, 2), data={"unit_index": 0})

    def harvest(self, identity, lot, quantity, *, animal=False, fertilizer=False):
        fields = {"animal_id" if animal else "plant_id": identity}
        return self.add(E.FERTILIZER_COLLECTED if fertilizer else E.HARVESTED,
                        product_lot_id=lot, quantity=quantity, crop=None if animal else "WHEAT",
                        data={"unit_index": 0, "product": "EGG"}, **fields)

    def feed(self, animal, lot):
        return self.add(E.ANIMAL_FED, animal_id=animal, data={"unit_index": 0,
                        "consumed_allocations": [{"product_lot_id": lot, "quantity": 1}]})

    def fertilize(self, plant, lot):
        return self.add(E.FERTILIZED, plant_id=plant, data={"unit_index": 0,
                        "consumed_allocations": [{"product_lot_id": lot, "quantity": 1}]})

    def move(self, source, target, quantity, *, automatic=False):
        return self.add(E.INVENTORY_RECONCILED if automatic else E.PRODUCT_MOVED,
                        data={"source": "unit:0", "destination": "shed", "allocations": [
                            {"product_lot_id": source, "destination_product_lot_id": target,
                             "quantity": quantity}]})

    def sale(self, lot, quantity, revenue):
        e = self.add(E.PRODUCT_SOLD, quantity=quantity, attributed_value=revenue,
                     data={"revenue_is_estimate": False, "allocations": [
                         {"product_lot_id": lot, "quantity": quantity, "realized_revenue": revenue}]})
        self.cash.append(CashEntry(e.step, 0, 0, "SELL", "product", quantity, revenue))
        self.refs[(e.step, "market", 0)] = e.decision_id
        return e

    def finish(self, **config):
        return assign_graph_credit(self.events, self.cash, self.refs, 0, GraphCreditConfig(**config))


def sales(report):
    return [a for a in report['allocations'] if a['kind']=='sale']


class ActionGraphTests(unittest.TestCase):
    def test_linear_is_geometric_and_diamond_merges_without_double_profit(self):
        g = ActionDAG()
        a = g.node(0)
        b = g.node(10, [(a, 1)])
        c = g.node(999, [(a, 1)])
        root = g.node(1000, [(b, 1), (c, 1)])
        s = g.shares(root, 244, .8)
        self.assertAlmostEqual(s[1000], 100)
        self.assertAlmostEqual(s[10], 40)
        self.assertAlmostEqual(s[999], 40)
        self.assertAlmostEqual(s[0], 64)
        self.assertAlmostEqual(sum(s.values()), 244)

    def test_quantity_branches_and_bookkeeping_are_fragment_invariant(self):
        g = ActionDAG()
        a, b = g.node(0), g.node(1)
        x = g.node(None, [(a, 1), (b, 3)])
        fragmented = g.node(None, [(a, 1), (b, 1), (b, 2)])
        self.assertEqual(x, fragmented)
        self.assertEqual(g.node(None, [(x, 1)]), x)
        root = g.node(2, [(x, 1)])
        shares = g.shares(root, 180, .8)
        self.assertAlmostEqual(shares[2], 100)
        self.assertAlmostEqual(shares[0], 20)
        self.assertAlmostEqual(shares[1], 60)

    def test_oldest_loss_is_finite_on_long_history(self):
        g = ActionDAG()
        node = None
        for i in range(4000):
            node = g.node(i, [(node, 1)] if node is not None else [])
        s = g.shares(node, -300, .8, oldest=True)
        self.assertTrue(all(math.isfinite(v) for v in s.values()))
        self.assertAlmostEqual(math.fsum(s.values()), -300)
        self.assertLess(s[0], s[3999])


class MaterialGraphTests(unittest.TestCase):
    def test_crop_split_between_sale_and_animal_no_duplicate_seed_cost(self):
        j = Journal()
        seed = j.buy('WHEAT', 1, 10)[0]
        j.plant(seed)
        j.harvest('p', 'wheat', 2)
        animal = j.buy('GOOSE', 1, 300, 'animal')[0]
        j.animal(animal)
        j.feed('a', 'wheat')
        j.harvest('a', 'egg', 4, animal=True)
        j.sale('wheat', 1, 26)
        j.sale('egg', 4, 200)
        r = j.finish()
        self.assertEqual([a['cost'] for a in sales(r)], [5, 305])
        self.assertAlmostEqual(r['allocated_gold'], -84)
        self.assertEqual(len(r['graph']['purchase_cost_audit']), 2)

    def test_shared_wheat_seed_cost_splits_between_two_animals(self):
        j = Journal()
        seed = j.buy('WHEAT', 1, 10)[0]
        j.plant(seed)
        j.harvest('p', 'w', 2)
        animals = j.buy('GOOSE', 2, 600, 'animal')
        for i, animal in enumerate(animals):
            identity = f'a{i}'
            j.animal(animal, identity)
            j.feed(identity, 'w')
            j.harvest(identity, f'egg{i}', 1, animal=True)
            j.sale(f'egg{i}', 1, 50)
        r = j.finish()
        self.assertEqual([a['cost'] for a in sales(r)], [305, 305])
        self.assertAlmostEqual(r['allocated_gold'], -510)

    def test_animal_fertilizer_to_crop_transfers_real_cost_not_free_input(self):
        j = Journal()
        a = j.buy('GOOSE', 1, 300, 'animal')[0]
        j.animal(a)
        j.harvest('a', 'fert', 1, animal=True, fertilizer=True)
        seed = j.buy('WHEAT', 1, 10)[0]
        j.plant(seed)
        j.fertilize('p', 'fert')
        j.harvest('p', 'w', 2)
        j.sale('w', 2, 50)
        r = j.finish()
        self.assertEqual(sales(r)[0]['cost'], 310)
        self.assertAlmostEqual(r['allocated_gold'], -260)
        self.assertFalse(any(a['net_profit'] < 0 for a in r['allocations'] if a['kind']!='sale'))

    def test_feedback_to_same_animal_is_acyclic_by_event_time(self):
        j = Journal()
        a = j.buy('GOOSE', 1, 300, 'animal')[0]
        j.animal(a)
        j.harvest('a', 'fert', 1, animal=True, fertilizer=True)
        seed = j.buy('WHEAT', 1, 10)[0]
        j.plant(seed)
        j.fertilize('p', 'fert')
        j.harvest('p', 'w', 2)
        j.feed('a', 'w')
        j.harvest('a', 'egg', 1, animal=True)
        j.sale('egg', 1, 100)
        r = j.finish()
        self.assertAlmostEqual(sales(r)[0]['cost'], 230)
        self.assertAlmostEqual(r['allocated_gold'], -210)
        for source, edges in r['graph']['material_edges'].items():
            self.assertTrue(all(source < e['target'] for e in edges))

    def test_later_feed_cannot_change_earlier_harvest_ancestry_or_feed_basis(self):
        j = Journal()
        a = j.buy('GOOSE', 1, 300, 'animal')[0]
        j.animal(a)
        j.harvest('a', 'first', 1, animal=True)
        early = j.sale('first', 1, 50)
        wheat = j.buy('WHEAT', 1, 25, 'product')[0]
        feed = j.feed('a', wheat)
        j.harvest('a', 'second', 1, animal=True)
        j.sale('second', 1, 50)
        r = j.finish()
        first, second = sales(r)
        self.assertEqual(first['cost'], 150)
        self.assertEqual(second['cost'], 175)
        self.assertNotIn(feed.decision_id, {a['decision_id'] for a in first['shares']})
        self.assertTrue(all(a['decision_id'] <= early.decision_id for a in first['shares']))

    def test_unsold_final_descendant_includes_drop_and_pickup(self):
        j = Journal()
        seed = j.buy('WHEAT', 1, 10)[0]
        j.plant(seed)
        j.harvest('p', 'h', 2)
        drop = j.move('h', 'shed', 2)
        pickup = j.move('shed', 'hand', 1)
        r = j.finish(unsold_weighting='oldest')
        self.assertLess(r['decision_credit'][drop.decision_id], 0)
        self.assertLess(r['decision_credit'][pickup.decision_id], 0)
        self.assertAlmostEqual(r['allocated_gold'], -10)
        tree = j.finish()
        self.assertLess(tree['decision_credit'][drop.decision_id], 0)
        self.assertLess(tree['decision_credit'][pickup.decision_id], 0)
        purchase_only = j.finish(unsold_weighting='purchase_only')
        self.assertEqual(purchase_only['decision_credit'], {0: -10/3000})

    def test_tree_keeps_unused_seed_on_buy_but_charges_care_path_when_plant_dies(self):
        unused = Journal()
        unused.buy('WHEAT', 1, 10)
        leftover = unused.finish()
        self.assertEqual(leftover['decision_credit'], {0: -10/3000})

        j = Journal()
        seed = j.buy('WHEAT', 1, 10)[0]
        planted = j.plant(seed)
        watered = j.water()
        lost = j.lose_plant()
        r = j.finish()
        self.assertAlmostEqual(r['allocated_gold'], -10)
        self.assertLess(r['decision_credit'][planted.decision_id], 0)
        self.assertLess(r['decision_credit'][watered.decision_id], 0)
        self.assertLess(r['decision_credit'][lost.decision_id], 0)
        self.assertLess(r['decision_credit'][0], r['decision_credit'][watered.decision_id])
        only_buy = j.finish(unsold_weighting='purchase_only')
        self.assertEqual(only_buy['decision_credit'], {0: -10/3000})

    def test_tree_charges_feed_path_when_placed_animal_escapes(self):
        j = Journal()
        stock = j.buy('GOOSE', 1, 300, 'animal')[0]
        placed = j.animal(stock)
        fed = j.feed_animal()
        lost = j.lose_animal()
        r = j.finish()
        self.assertAlmostEqual(r['allocated_gold'], -300)
        self.assertLess(r['decision_credit'][placed.decision_id], 0)
        self.assertLess(r['decision_credit'][fed.decision_id], 0)
        self.assertLess(r['decision_credit'][lost.decision_id], 0)
        only_buy = j.finish(unsold_weighting='purchase_only')
        self.assertEqual(only_buy['decision_credit'], {0: -300/3000})

    def test_partial_sale_and_leftover_cost_are_quantity_conserved(self):
        j = Journal()
        lots = j.buy('WHEAT', 2, 50, 'product')
        j.sale(lots[0], 1, 20)
        r = j.finish()
        self.assertEqual(sales(r)[0]['cost'], 25)
        self.assertEqual(sales(r)[0]['net_profit'], -5)
        self.assertAlmostEqual(r['allocated_gold'], -30)

    def test_unknown_source_is_explicit_and_missing_known_lot_is_error(self):
        j = Journal()
        j.add(E.INVENTORY_RECONCILED, product_lot_id='unknown', quantity=1, crop='WHEAT',
              data={'source':'external_or_unobserved','destination':'shed'})
        j.sale('unknown', 1, 25)
        self.assertEqual(j.finish()['unknown_source_sale_revenue'], 25)
        j.events.pop(0)
        with self.assertRaisesRegex(ValueError, 'Missing tracked product'):
            j.finish()

    def test_cannot_consume_same_material_twice(self):
        j = Journal()
        lot = j.buy('WHEAT', 1, 25, 'product')[0]
        j.sale(lot, 1, 30)
        j.sale(lot, 1, 30)
        with self.assertRaisesRegex(ValueError, 'over-consumption'):
            j.finish()

    def test_one_drop_does_not_mix_unrelated_product_histories(self):
        j = Journal()
        wheat = j.buy('WHEAT', 1, 25, 'product')[0]
        fertilizer = j.buy('FERTILIZER', 1, 100, 'product')[0]
        j.add(E.PRODUCT_MOVED, data={'source':'unit:0','destination':'shed','allocations':[
            {'product_lot_id':wheat,'destination_product_lot_id':'w','quantity':1},
            {'product_lot_id':fertilizer,'destination_product_lot_id':'f','quantity':1}]})
        j.sale('w', 1, 50)
        j.sale('f', 1, 120)
        r = j.finish()
        wheat_shares, fertilizer_shares = [dict((s['decision_id'], s['gold']) for s in a['shares'])
                                         for a in sales(r)]
        self.assertNotIn(1, wheat_shares)
        self.assertNotIn(0, fertilizer_shares)
        self.assertIn(2, wheat_shares)
        self.assertIn(2, fertilizer_shares)

    def test_lost_animal_stock_consumes_exact_portion_and_keeps_path(self):
        j = Journal()
        lot = j.buy('COW', 1, 400, 'animal')[0]
        moved = j.add(E.ANIMAL_STOCK_MOVED, quantity=1, data={'source':'shed',
            'destination':'unit:0','source_lot_id':lot,'destination_lot_id':'held','animal':'COW'})
        j.add(E.INVENTORY_RECONCILED, quantity=1, data={'source':'unit:0',
            'destination':'lost_or_consumed','animal_allocations':[{'animal_lot_id':'held','quantity':1}]})
        r = j.finish(unsold_weighting='oldest')
        self.assertEqual([a['kind'] for a in r['allocations']], ['lost_animal_stock'])
        self.assertEqual(r['allocated_gold'], -400)
        self.assertIn(moved.decision_id, r['decision_credit'])

    def test_wrong_cash_or_duplicate_journal_fails_closed(self):
        j = Journal()
        j.buy('WHEAT', 1, 10)
        j.cash[0] = CashEntry(0,0,0,'BUY_SEED','WHEAT',1,-20)
        with self.assertRaisesRegex(ValueError, 'actual cash'):
            j.finish()
        j.events.append(j.events[0])
        with self.assertRaisesRegex(ValueError, 'Journal'):
            j.finish()

    def test_bulk_sell_preserves_each_lots_profit_sign(self):
        j = Journal()
        cheap = j.buy('WHEAT', 1, 10, 'product')[0]
        expensive = j.buy('WHEAT', 1, 100, 'product')[0]
        e = j.add(E.PRODUCT_SOLD, quantity=2, attributed_value=60, data={
            'revenue_is_estimate':False, 'allocations':[
                {'product_lot_id':cheap,'quantity':1,'realized_revenue':30},
                {'product_lot_id':expensive,'quantity':1,'realized_revenue':30}]})
        j.cash.append(CashEntry(e.step,0,0,'SELL','WHEAT',2,60))
        j.refs[(e.step,'market',0)] = e.decision_id
        r = j.finish()
        self.assertEqual([a['net_profit'] for a in sales(r)], [20,-70])
        self.assertGreater(r['decision_credit'][0], 0)
        self.assertLess(r['decision_credit'][1], 0)
        self.assertAlmostEqual(r['allocated_gold'], -50)


class EngineGraphTests(unittest.TestCase):
    def test_real_engine_previously_crashing_wheat_sale_and_feed(self):
        config = {'episodeSteps':80,'turnsPerDay':4,'weedSpawnChance':0.,'seed':77}
        env = make('kaggriculture', configuration=config)
        tracker = FarmLifecycleTracker(env.state[0].observation, episode_number=1, player_index=0,
                                       turns_per_day=4, graph_tracking=True)
        ledger = CashLedger([3000,3000])
        script = [(['PASS'],[['BUY_SEED','WHEAT',1],['BUY_ANIMAL','GOOSE',1]]),
            (['PLANT','WHEAT'],[]),(['WATER'],[]),(['PASS'],[]),(['WATER'],[]),(['PASS'],[]),
            (['PASS'],[]),(['PASS'],[]),(['WATER'],[]),(['HARVEST'],[]),(['DROP'],[]),
            (['BUILD_COOP'],[]),(['PICKUP','GOOSE'],[]),(['PLACE','GOOSE'],[]),
            (['PICKUP','WHEAT',1],[]),(['FEED'],[]),
            (['PASS'],[['SELL','WHEAT',1],['BUY_PRODUCT','WHEAT',10]])]
        script += [(['PICKUP','WHEAT',1],[]),(['FEED'],[]),(['CARE'],[]),(['PASS'],[])]*4
        script += [(['HARVEST'],[]),(['DROP'],[]),(['PASS'],[['SELL','EGG',4]])]
        refs = play_traced(env, tracker, script, config, ledger)
        r = assign_graph_credit(tracker.events, ledger.entries, refs, 0)
        self.assertEqual(ledger.max_reconciliation_error, 0)
        self.assertAlmostEqual(r['allocated_gold'], ledger.final_money[0]-3000)
        self.assertEqual(r['unknown_source_sale_revenue'], 0)
        self.assertEqual(sales(r)[0]['cost'], 5)

    def test_all_animal_species_with_bought_feed_and_nightly_storage(self):
        for species, product, structure in [('GOOSE','EGG','BUILD_COOP'),
                                             ('COW','MILK','BUILD_PASTURE'),
                                             ('SHEEP','WOOL','BUILD_PASTURE')]:
            with self.subTest(species=species):
                config = {'episodeSteps':140,'turnsPerDay':8,'weedSpawnChance':0.,'seed':77}
                env = make('kaggriculture', configuration=config)
                tracker = FarmLifecycleTracker(env.state[0].observation, episode_number=1,
                    player_index=0, turns_per_day=8, graph_tracking=True)
                ledger = CashLedger([3000,3000])
                # Carry the animal across midnight; its lot identity changes on
                # automatic storage, then PICKUP/PLACE must retain its purchase.
                script = [([structure],[['BUY_ANIMAL',species,1],['BUY_PRODUCT','WHEAT',12]])]
                script += [(['PASS'],[])]*6 + [(['PICKUP',species],[])]
                script += [(['PICKUP',species],[]),(['PLACE',species],[]),
                           (['PICKUP','WHEAT',1],[]),(['FEED'],[]),(['CARE'],[])]
                script += [(['PASS'],[])]*3
                for _ in range(10):
                    script += [(['PICKUP','WHEAT',1],[]),(['FEED'],[]),(['CARE'],[]),
                               (['HARVEST'],[]),(['COLLECT_FERTILIZER'],[]),
                               (['PASS'],[]),(['PASS'],[]),(['PASS'],[])]
                script += [(['PASS'],[['SELL',product,100],['SELL','FERTILIZER',100]])]
                refs = play_traced(env, tracker, script, config, ledger)
                r = assign_graph_credit(tracker.events, ledger.entries, refs, 0)
                self.assertGreater(len(sales(r)), 0)
                self.assertTrue(any(e.event_type == E.HARVESTED and e.animal_id for e in tracker.events))
                self.assertTrue(any(e.event_type == E.FERTILIZER_COLLECTED for e in tracker.events))
                self.assertEqual(len({a['event_id'] for a in sales(r)}), 2)
                self.assertTrue(any(e.event_type == E.ANIMAL_STOCK_MOVED and e.decision_id is None
                                    for e in tracker.events))
                self.assertEqual(r['unknown_source_sale_revenue'], 0)
                self.assertAlmostEqual(r['allocated_gold'], ledger.final_money[0]-3000)

    def test_every_crop_with_fertilizer_and_repeated_harvests(self):
        for crop in ('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON'):
            with self.subTest(crop=crop):
                config = {'episodeSteps':200,'turnsPerDay':8,'weedSpawnChance':0.,'seed':77}
                env = make('kaggriculture', configuration=config)
                tracker = FarmLifecycleTracker(env.state[0].observation, episode_number=1,
                    player_index=0, turns_per_day=8, graph_tracking=True)
                ledger = CashLedger([3000,3000])
                script = [(['PASS'],[['BUY_SEED',crop,1],['BUY_PRODUCT','FERTILIZER',1]]),
                    (['PLANT',crop],[]),(['PICKUP','FERTILIZER',1],[]),(['FERTILIZE'],[]),
                    (['WATER'],[]),(['PASS'],[]),(['PASS'],[]),(['PASS'],[])]
                for _ in range(20):
                    script += [(['WATER'],[]),(['HARVEST'],[]),(['DROP'],[]),
                               (['PASS'],[['SELL',crop,100]])] + [(['PASS'],[])]*4
                refs = play_traced(env, tracker, script, config, ledger)
                r = assign_graph_credit(tracker.events, ledger.entries, refs, 0)
                self.assertGreater(len(sales(r)), 0)
                if crop in ('TOMATO','STRAWBERRY'):
                    self.assertGreater(len(sales(r)), 1)
                self.assertTrue(any(e.event_type == E.FERTILIZED for e in tracker.events))
                self.assertEqual(r['unknown_source_sale_revenue'], 0)
                self.assertAlmostEqual(r['allocated_gold'], ledger.final_money[0]-3000)

    def test_bootstrap_inventory_is_explicit_unknown_not_missing(self):
        env = make('kaggriculture', configuration={'seed':77})
        obs = deepcopy(env.state[0].observation)
        obs['private']['shed']['WHEAT'] = 3
        obs['private']['inventories'][0]['GOOSE'] = 1
        tracker = FarmLifecycleTracker(obs, episode_number=1, player_index=0, graph_tracking=True)
        r = assign_graph_credit(tracker.events, [], {}, 0)
        self.assertEqual(r['allocated_gold'], 0)
        self.assertEqual(sum(a.get('quantity', 0) for a in r['allocations']), 4)


if __name__ == '__main__':
    unittest.main()
