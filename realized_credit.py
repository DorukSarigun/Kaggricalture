"""Episode-end realized chain credit: sale profit newest-heavy, unsold seed oldest-heavy.

No potential and no GAE on these local shares. Terminal gold difference is trained
separately. A plant that sells distributes net profit over purchase, tile care,
harvest, carriage and sale. A plant that never sells distributes that one seed's
purchase price as a loss, heaviest on the earliest linked action.
"""
from collections import defaultdict
from dataclasses import dataclass
import math

from lifecycle_tracking import LifecycleEventType as E


@dataclass(frozen=True)
class ChainCreditConfig:
    decay: float = 0.8
    reward_scale: float = 3000.0

    def __post_init__(self):
        if not math.isfinite(self.decay) or not 0 < self.decay <= 1:
            raise ValueError("chain decay must be in (0, 1]")
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward scale must be positive and finite")


def diminishing_shares(profit, decision_ids, decay, heaviest="newest"):
    """Unique chronological operations, normalized geometric weights.

    heaviest='newest': sale path — last linked action keeps the largest share.
    heaviest='oldest': unsold seed — first linked action takes the largest loss.
    Decay is per linked operation rank, not per physical turn.
    """
    ids = sorted(set(decision_ids))
    if heaviest not in {"newest", "oldest"}:
        raise ValueError("heaviest must be 'newest' or 'oldest'")
    if not ids or not math.isfinite(profit) or not 0 < decay <= 1:
        raise ValueError("invalid profit, decisions or decay")
    n = len(ids)
    if heaviest == "newest":
        weights = [decay ** (n - 1 - i) for i in range(n)]
        residual_id = ids[-1]
    else:
        weights = [decay ** i for i in range(n)]
        residual_id = ids[0]
    total = math.fsum(weights)
    shares = {i: profit * w / total for i, w in zip(ids, weights)}
    shares[residual_id] += profit - math.fsum(shares.values())
    return shares


def assign_realized_credit(events, cash_entries, decision_refs, player, config=None):
    """Return an auditable report with raw-gold allocations indexed by decision.

    decision_refs maps (step, section, unit/order index) -> episode decision ID.
    Unknown-source sales are explicit SELL-only credits, never invented chains.
    HIRE/BUY_LAND stay on their purchase decision. Animal purchase cost follows
    the placed animal: sold products share newest-heavy profit, unsold/escaped
    animals share the unit price oldest-heavy. Unplaced animals remain on BUY.
    """
    config = config or ChainCreditConfig()
    purchases, seeds, plants = {}, {}, {}
    paths, inputs = defaultdict(set), defaultdict(set)
    costs = {}  # resource identity -> (purchase event ID, realized cost)
    plant_paths, plant_inputs = defaultdict(set), defaultdict(set)
    animal_paths, animal_inputs = defaultdict(set), defaultdict(set)
    pending_walk = defaultdict(list)
    operation_walk, cleared_tile_paths = {}, {}
    structure_paths = defaultdict(set)
    plant_lots, animal_product_lots = defaultdict(set), defaultdict(set)
    animals, animal_purchases = {}, {}
    chains = {}
    unknown_revenue = 0.0

    def register_purchase(e):
        if e.decision_id is None:
            raise ValueError(f"Missing purchase decision: {e.event_id}")
        cost = float(e.data["total_cost"])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Invalid executed purchase cost")
        purchases[e.event_id] = (e.decision_id, cost)

    def consume_walk(e, unit=None):
        unit = e.data.get("unit_index") if unit is None else unit
        if unit is None or e.decision_id is None:
            return set()
        if e.decision_id not in operation_walk:
            operation_walk[e.decision_id] = set(pending_walk.pop((e.day, unit), []))
        return operation_walk[e.decision_id]

    for e in events:
        ref = {e.decision_id} if e.decision_id is not None else set()
        if e.event_type == E.SEED_PURCHASED:
            register_purchase(e)
            seeds[e.seed_lot_id] = e
        elif e.event_type == E.ANIMAL_PURCHASED:
            register_purchase(e)
            for lot_id in e.data.get("animal_lot_ids", ()):
                animal_purchases[lot_id] = e
        elif e.event_type == E.PRODUCT_PURCHASED:
            register_purchase(e)
            lot_ids, prices = e.data["product_lot_ids"], e.data["unit_prices"]
            if len(lot_ids) != len(prices):
                raise ValueError("Product purchase unit-cost lineage mismatch")
            for lot, price in zip(lot_ids, prices):
                key = ("product", lot)
                paths[lot] |= ref
                inputs[lot].add(key)
                costs[key] = (e.event_id, float(price))
        elif e.event_type == E.UNIT_MOVED:
            if e.decision_id is None:
                raise ValueError("Movement without decision ID")
            pending_walk[(e.day, e.data["unit_index"])].append(e.decision_id)
            for lot in e.data.get("carried_product_lot_ids", ()):
                paths[lot] |= ref
        elif e.event_type == E.PLANTED:
            plants[e.plant_id] = e
            plant_paths[e.plant_id] |= ref | consume_walk(e)
            plant_paths[e.plant_id] |= cleared_tile_paths.pop(e.tile_xy, set())
            purchase = seeds.get(e.seed_lot_id)
            if purchase is not None:
                key = ("seed", e.plant_id)
                costs[key] = (purchase.event_id, float(e.data["seed_unit_cost"]))
                plant_inputs[e.plant_id].add(key)
                plant_paths[e.plant_id].add(purchase.decision_id)
        elif e.event_type in (E.WATERED, E.FERTILIZED):
            plant_paths[e.plant_id] |= ref | consume_walk(e)
            for allocation in e.data.get("consumed_allocations", ()):
                lot = allocation["product_lot_id"]
                plant_paths[e.plant_id] |= paths[lot]
                if allocation.get("source_animal_id"):
                    continue
                plant_inputs[e.plant_id] |= inputs[lot]
        elif e.event_type == E.STRUCTURE_BUILT:
            structure_paths[e.tile_xy] |= ref | consume_walk(e)
        elif e.event_type == E.ANIMAL_PLACED:
            animals[e.animal_id] = e
            animal_paths[e.animal_id] |= ref | consume_walk(e)
            animal_paths[e.animal_id] |= structure_paths.pop(e.tile_xy, set())
            animal_paths[e.animal_id] |= set(e.data.get("extra_decision_ids", ()))
            purchase = animal_purchases.get(e.animal_lot_id)
            unit_cost = float(e.data.get("animal_unit_cost", 0.0))
            purchase_event_id = e.data.get("purchase_event_id")
            purchase_decision = e.data.get("purchase_decision_id")
            if purchase is not None:
                purchase_event_id = purchase.event_id
                purchase_decision = purchase.decision_id
            if purchase_event_id and unit_cost > 0:
                key = ("animal", e.animal_id)
                costs[key] = (purchase_event_id, unit_cost)
                animal_inputs[e.animal_id].add(key)
            if purchase_decision is not None:
                animal_paths[e.animal_id].add(purchase_decision)
        elif e.event_type in (E.ANIMAL_FED, E.ANIMAL_CARED):
            animal_paths[e.animal_id] |= ref | consume_walk(e)
            for allocation in e.data.get("consumed_allocations", ()):
                lot = allocation["product_lot_id"]
                animal_paths[e.animal_id] |= paths[lot]
                animal_inputs[e.animal_id] |= inputs[lot]
        elif e.event_type == E.HARVESTED:
            # Snapshot: later care/harvest cannot earn credit from earlier fruit.
            walk = consume_walk(e)
            if e.animal_id:
                paths[e.product_lot_id] = animal_paths[e.animal_id] | ref | walk
                inputs[e.product_lot_id] = set(animal_inputs[e.animal_id])
                animal_product_lots[e.animal_id].add(e.product_lot_id)
            else:
                paths[e.product_lot_id] = plant_paths[e.plant_id] | ref | walk
                inputs[e.product_lot_id] = set(plant_inputs[e.plant_id])
                plant_lots[e.plant_id].add(e.product_lot_id)
        elif e.event_type == E.FERTILIZER_COLLECTED:
            walk = consume_walk(e)
            paths[e.product_lot_id] = animal_paths[e.animal_id] | ref | walk
            inputs[e.product_lot_id] = set(animal_inputs[e.animal_id])
            animal_product_lots[e.animal_id].add(e.product_lot_id)
        elif e.event_type in (E.PRODUCT_MOVED, E.INVENTORY_RECONCILED):
            walk = set()
            if e.event_type == E.PRODUCT_MOVED:
                location = next((e.data.get(k, '') for k in ('source', 'destination')
                                 if e.data.get(k, '').startswith('unit:')), None)
                if location:
                    walk = consume_walk(e, int(location.split(':')[1]))
            for allocation in e.data.get("allocations", ()):
                source = allocation["product_lot_id"]
                target = allocation.get("destination_product_lot_id")
                if target:
                    paths[target] = paths[source] | ref | walk
                    inputs[target] = set(inputs[source])
        elif e.event_type == E.TILE_CLEARED:
            cleared_tile_paths[e.tile_xy] = ref | consume_walk(e)
        elif e.event_type == E.PLANT_LOST:
            walk = consume_walk(e)
            if e.plant_id:
                # Tile-ending DIG/decay stays on the unsold-loss chain only;
                # sold fruit already snapshotted at harvest.
                plant_paths[e.plant_id] |= ref | walk
        elif e.event_type == E.ANIMAL_LOST:
            walk = consume_walk(e)
            if e.animal_id:
                animal_paths[e.animal_id] |= ref | walk
        elif e.event_type == E.PRODUCT_SOLD:
            if e.decision_id is None or e.data.get("revenue_is_estimate", True):
                raise ValueError("Realized credit requires traced sale and decision ID")
            revenue_sum = 0.0
            for a in e.data["allocations"]:
                revenue = float(a["realized_revenue"])
                revenue_sum += revenue
                if not math.isfinite(revenue) or revenue < 0:
                    raise ValueError("Invalid realized revenue")
                plant, animal, purchase = (
                    a.get("source_plant_id"), a.get("source_animal_id"), a.get("source_purchase_id")
                )
                if plant in plants:
                    origin = ("plant", plant)
                elif animal in animals:
                    origin = ("animal", animal)
                elif purchase:
                    origin = ("purchase", purchase)
                else:
                    origin = ("unlinked_sale", e.event_id)
                if origin[0] == "unlinked_sale":
                    unknown_revenue += revenue
                chain = chains.setdefault(origin, {"kind": origin[0], "source_id": origin[1],
                    "revenue": 0.0, "sold_units": 0, "decisions": set(), "cost_refs": set(),
                    "sale_event_ids": set()})
                lot = a["product_lot_id"]
                chain["revenue"] += revenue
                chain["sold_units"] += int(a["quantity"])
                chain["decisions"] |= paths[lot] | ref
                chain["cost_refs"] |= inputs[lot]
                chain["sale_event_ids"].add(e.event_id)
            if not math.isclose(revenue_sum, e.attributed_value, abs_tol=1e-7):
                raise ValueError("Sale allocations do not conserve executed revenue")

    credits = defaultdict(float)
    allocations, claimed, used_cost = [], set(), defaultdict(float)
    for chain in chains.values():
        overlap = claimed & chain["cost_refs"]
        if overlap:
            raise ValueError(f"Input cost claimed by unrelated chains: {overlap}")
        claimed |= chain["cost_refs"]
        cost = 0.0
        for key in chain["cost_refs"]:
            purchase, amount = costs[key]
            used_cost[purchase] += amount
            cost += amount
        profit = chain["revenue"] - cost
        shares = diminishing_shares(profit, chain["decisions"], config.decay, heaviest="newest")
        for decision, amount in shares.items():
            credits[decision] += amount
        allocations.append({"kind": chain["kind"], "source_id": chain["source_id"],
            "revenue": chain["revenue"], "cost": cost, "net_profit": profit,
            "sold_units": chain["sold_units"], "sale_event_ids": sorted(chain["sale_event_ids"]),
            "cost_refs": [list(k) for k in sorted(chain["cost_refs"])],
            "heaviest": "newest",
            "shares": [{"decision_id": i, "gold": g} for i, g in shares.items()]})

    sold_plants = {chain["source_id"] for chain in chains.values() if chain["kind"] == "plant"}
    for plant_id, planted in plants.items():
        if plant_id in sold_plants:
            continue
        key = ("seed", plant_id)
        if key not in costs or key in claimed:
            continue
        purchase_id, seed_cost = costs[key]
        decisions = set(plant_paths[plant_id])
        for lot in plant_lots.get(plant_id, ()):
            decisions |= paths[lot]
        if planted.decision_id is not None:
            decisions.add(planted.decision_id)
        if not decisions:
            continue
        claimed.add(key)
        used_cost[purchase_id] += seed_cost
        profit = -float(seed_cost)
        shares = diminishing_shares(profit, decisions, config.decay, heaviest="oldest")
        for decision, amount in shares.items():
            credits[decision] += amount
        allocations.append({"kind": "unsold_plant", "source_id": plant_id,
            "revenue": 0.0, "cost": seed_cost, "net_profit": profit,
            "sold_units": 0, "sale_event_ids": [],
            "cost_refs": [list(key)], "heaviest": "oldest",
            "shares": [{"decision_id": i, "gold": g} for i, g in shares.items()]})

    sold_animals = {chain["source_id"] for chain in chains.values() if chain["kind"] == "animal"}
    for animal_id, placed in animals.items():
        if animal_id in sold_animals:
            continue
        key = ("animal", animal_id)
        if key not in costs or key in claimed:
            continue
        purchase_id, animal_cost = costs[key]
        decisions = set(animal_paths[animal_id])
        for lot in animal_product_lots.get(animal_id, ()):
            decisions |= paths[lot]
        if placed.decision_id is not None:
            decisions.add(placed.decision_id)
        if not decisions:
            continue
        claimed.add(key)
        used_cost[purchase_id] += animal_cost
        profit = -float(animal_cost)
        shares = diminishing_shares(profit, decisions, config.decay, heaviest="oldest")
        for decision, amount in shares.items():
            credits[decision] += amount
        allocations.append({"kind": "unsold_animal", "source_id": animal_id,
            "revenue": 0.0, "cost": animal_cost, "net_profit": profit,
            "sold_units": 0, "sale_event_ids": [],
            "cost_refs": [list(key)], "heaviest": "oldest",
            "shares": [{"decision_id": i, "gold": g} for i, g in shares.items()]})

    for purchase, (decision, cost) in purchases.items():
        remainder = cost - used_cost[purchase]
        if remainder < -1e-7:
            raise ValueError("Claimed input cost exceeds purchase cost")
        if remainder > 1e-7:
            credits[decision] -= remainder
            allocations.append({"kind": "unsold_or_unrealized_input", "source_id": purchase,
                "net_profit": -remainder, "heaviest": "oldest",
                "shares": [{"decision_id": decision, "gold": -remainder}]})

    own_entries = [e for e in cash_entries if e.player == player]
    for entry in own_entries:
        if entry.operation in {"HIRE", "BUY_LAND"} and entry.amount:
            decision = decision_refs[(entry.step, "market", entry.order_index)]
            credits[decision] += entry.amount
            allocations.append({"kind": "unallocated_capital_or_labor_cost", "operation": entry.operation,
                "net_profit": entry.amount, "shares": [{"decision_id": decision, "gold": entry.amount}]})
    net_cash = math.fsum(e.amount for e in own_entries)
    total = math.fsum(credits.values())
    error = total - net_cash
    if abs(error) > 1e-7:
        raise RuntimeError(f"Realized credits do not conserve own cash: {total} != {net_cash}")
    return {"version": "realized-chain-v1", "decay": config.decay,
        "decay_unit": "linked_operation_rank", "reward_scale": config.reward_scale,
        "net_cash": net_cash, "allocated_gold": total, "reconciliation_error": error,
        "unknown_source_sale_revenue": unknown_revenue, "allocations": allocations,
        "decision_credit": {i: gold / config.reward_scale for i, gold in credits.items()}}
