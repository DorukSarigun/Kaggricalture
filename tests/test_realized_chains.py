from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from kaggle_environments import make

from cash_accounting import CashEntry, CashLedger
from lifecycle_tracking import LifecycleEvent, LifecycleEventType as E, FarmLifecycleTracker
from market_execution import trace_market_executions
from realized_credit import ChainCreditConfig, assign_realized_credit, diminishing_shares
from realized_training import (RealizedChainActorCritic, propose_chain_turn,
                               finish_chain_episode, update_chain_policy, clipped_objective)
from train_realized_chains import collect_episode, parser
from test_structured import observation


def event(kind, decision, *, data=None, **kwargs):
    return LifecycleEvent(event_id=f"event-{decision}-{kind}", event_type=kind,
        episode=1, step=decision, day=0, hour=decision, player=0,
        decision_id=decision, data=data or {}, **kwargs)


def purchase(decision=0, quantity=3):
    return event(E.SEED_PURCHASED, decision, seed_lot_id="seeds", crop="WHEAT", quantity=quantity,
                 attributed_value=-10*quantity, data={"total_cost": 10*quantity})


def plant(decision=1, identity="p"):
    return event(E.PLANTED, decision, seed_lot_id="seeds", plant_id=identity, crop="WHEAT",
                 data={"seed_unit_cost": 10, "unit_index": 0})


def harvest(decision=3, lot="h", identity="p"):
    return event(E.HARVESTED, decision, plant_id=identity, product_lot_id=lot, quantity=2,
                 data={"unit_index": 0})


def sale(decision=5, lot="h", identity="p", revenue=50, **allocation):
    return event(E.PRODUCT_SOLD, decision, attributed_value=revenue,
        data={"revenue_is_estimate": False, "allocations": [dict(product_lot_id=lot,
              source_plant_id=identity, quantity=2, realized_revenue=revenue, **allocation)]})


def cash(operation, amount, step=0, item="WHEAT"):
    return CashEntry(step, 0, 0, operation, item, 1, amount)


class RealizedCreditTests(unittest.TestCase):
    def test_normalized_diminishing_unique_actions_and_negative_profit(self):
        shares = diminishing_shares(100, [4, 3, 2, 1, 0, 0], .8)
        self.assertAlmostEqual(sum(shares.values()), 100)
        for i in range(4):
            self.assertAlmostEqual(shares[i]/shares[i+1], .8)
        negative = diminishing_shares(-100, range(5), .8)
        self.assertTrue(all(v < 0 for v in negative.values()))
        self.assertAlmostEqual(sum(negative.values()), -100)
        long = diminishing_shares(123, range(10000), .8)
        self.assertAlmostEqual(math.fsum(long.values()), 123)
        oldest = diminishing_shares(-100, range(5), .8, heaviest="oldest")
        self.assertAlmostEqual(sum(oldest.values()), -100)
        for i in range(4):
            self.assertAlmostEqual(oldest[i] / oldest[i + 1], 1 / .8)
        self.assertLess(oldest[0], oldest[4])
        for value in [0, -1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                ChainCreditConfig(value)

    def test_partial_sales_repeated_harvests_single_seed_cost_and_unsold_purchase(self):
        events = [purchase(), plant(), event(E.WATERED, 2, plant_id="p", data={"unit_index": 0}),
                  harvest(), sale(), event(E.WATERED, 6, plant_id="p", data={"unit_index": 0}),
                  harvest(7, "h2"), sale(8, "h2", revenue=40), plant(9, "unused")]
        report = assign_realized_credit(events, [cash("BUY_SEED", -30), cash("SELL", 90)], {}, 0)
        chain = next(a for a in report['allocations'] if a['kind']=='plant')
        self.assertEqual(chain['cost'], 10)
        self.assertEqual(chain['net_profit'], 80)
        self.assertEqual(chain['sold_units'], 4)
        self.assertAlmostEqual(report['allocated_gold'], 60)
        unused = next(a for a in report['allocations'] if a['kind']=='unsold_plant')
        self.assertEqual(unused['source_id'], 'unused')
        self.assertAlmostEqual(unused['net_profit'], -10)
        unused_shares = {s['decision_id']: s['gold'] for s in unused['shares']}
        self.assertLess(unused_shares[0], unused_shares[9])
        leftover = next(a for a in report['allocations'] if a['kind']=='unsold_or_unrealized_input')
        self.assertEqual(leftover['shares'], [{'decision_id': 0, 'gold': -10}])

    def test_approach_and_carried_product_routes_follow_lot_through_split(self):
        move = event(E.UNIT_MOVED, 1, data={"unit_index": 0, "carried_product_lot_ids": []})
        carrying = event(E.UNIT_MOVED, 4, data={"unit_index": 0, "carried_product_lot_ids": ['h']})
        transfer = event(E.PRODUCT_MOVED, 5, data={"allocations": [
            {"product_lot_id": "h", "destination_product_lot_id": "shedlot"}]})
        events = [purchase(quantity=1), move, plant(2), harvest(), carrying, transfer,
                  sale(6, "shedlot"), event(E.WATERED, 7, plant_id="p", data={"unit_index": 0})]
        report = assign_realized_credit(events, [cash("BUY_SEED", -10), cash("SELL", 50)], {}, 0)
        self.assertEqual(set(report['decision_credit']), set(range(7)))
        self.assertAlmostEqual(sum(report['decision_credit'].values()), 40/3000)

    def test_unsold_care_chain_penalizes_purchase_most(self):
        events = [purchase(quantity=1), plant(),
                  event(E.WATERED, 2, plant_id="p", data={"unit_index": 0})]
        report = assign_realized_credit(events, [cash("BUY_SEED", -10)], {}, 0)
        unused = next(a for a in report['allocations'] if a['kind']=='unsold_plant')
        shares = {s['decision_id']: s['gold'] for s in unused['shares']}
        self.assertEqual(set(shares), {0, 1, 2})
        self.assertAlmostEqual(sum(shares.values()), -10)
        self.assertLess(shares[0], shares[1])
        self.assertLess(shares[1], shares[2])
        self.assertFalse(any(a['kind']=='unsold_or_unrealized_input' for a in report['allocations']))

    def test_no_sale_seed_loss_is_first_heavy_on_the_plant_chain(self):
        report = assign_realized_credit([purchase(), plant(), harvest()], [cash("BUY_SEED", -30)], {}, 0)
        self.assertAlmostEqual(sum(report['decision_credit'].values()), -30 / 3000)
        unused = next(a for a in report['allocations'] if a['kind']=='unsold_plant')
        shares = {s['decision_id']: s['gold'] for s in unused['shares']}
        self.assertEqual(set(shares), {0, 1, 3})
        self.assertLess(shares[0], shares[1])
        self.assertLess(shares[1], shares[3])
        leftover = next(a for a in report['allocations'] if a['kind']=='unsold_or_unrealized_input')
        self.assertEqual(leftover['shares'], [{'decision_id': 0, 'gold': -20}])

    def test_animal_sale_is_newest_heavy_and_unsold_is_first_heavy(self):
        buy = event(E.ANIMAL_PURCHASED, 0, animal_lot_id="a0",
                    data={"total_cost": 300, "animal_lot_ids": ["a0"], "animal": "GOOSE"})
        built = event(E.STRUCTURE_BUILT, 1, tile_xy=(4, 4), data={"unit_index": 0, "kind": "COOP"})
        placed = event(E.ANIMAL_PLACED, 2, animal_id="g", animal_lot_id="a0", tile_xy=(4, 4),
                       data={"animal": "GOOSE", "animal_unit_cost": 300, "unit_index": 0})
        fed = event(E.ANIMAL_FED, 3, animal_id="g", data={"unit_index": 0, "consumed_allocations": []})
        harvested = event(E.HARVESTED, 4, animal_id="g", product_lot_id="egg", quantity=1,
                          data={"unit_index": 0, "product": "EGG"})
        sold = event(E.PRODUCT_SOLD, 5, attributed_value=400,
                     data={"revenue_is_estimate": False, "allocations": [{
                         "product_lot_id": "egg", "source_animal_id": "g",
                         "quantity": 1, "realized_revenue": 400}]})
        report = assign_realized_credit(
            [buy, built, placed, fed, harvested, sold],
            [cash("BUY_ANIMAL", -300, item="GOOSE"), cash("SELL", 400, item="EGG")], {}, 0)
        chain = next(a for a in report["allocations"] if a["kind"] == "animal")
        self.assertEqual(chain["cost"], 300)
        self.assertAlmostEqual(chain["net_profit"], 100)
        shares = {s["decision_id"]: s["gold"] for s in chain["shares"]}
        self.assertEqual(set(shares), {0, 1, 2, 3, 4, 5})
        self.assertGreater(shares[5], shares[0])
        self.assertAlmostEqual(report["allocated_gold"], 100)

        unsold = assign_realized_credit(
            [buy, built, placed], [cash("BUY_ANIMAL", -300, item="GOOSE")], {}, 0)
        loss = next(a for a in unsold["allocations"] if a["kind"] == "unsold_animal")
        loss_shares = {s["decision_id"]: s["gold"] for s in loss["shares"]}
        self.assertEqual(set(loss_shares), {0, 1, 2})
        self.assertLess(loss_shares[0], loss_shares[2])
        self.assertAlmostEqual(unsold["allocated_gold"], -300)

    def test_unplaced_animal_stays_on_purchase(self):
        buy = event(E.ANIMAL_PURCHASED, 0, animal_lot_id="a0",
                    data={"total_cost": 300, "animal_lot_ids": ["a0"]})
        report = assign_realized_credit([buy], [cash("BUY_ANIMAL", -300, item="GOOSE")], {}, 0)
        leftover = next(a for a in report["allocations"] if a["kind"] == "unsold_or_unrealized_input")
        self.assertEqual(leftover["shares"], [{"decision_id": 0, "gold": -300}])

    def test_sibling_unplaced_animal_does_not_load_the_placed_chain(self):
        buy = event(E.ANIMAL_PURCHASED, 0, animal_lot_id="a0",
                    data={"total_cost": 600, "animal_lot_ids": ["a0", "a1"]})
        placed = event(E.ANIMAL_PLACED, 1, animal_id="g", animal_lot_id="a0",
                       data={"animal_unit_cost": 300, "unit_index": 0})
        harvested = event(E.HARVESTED, 2, animal_id="g", product_lot_id="egg", quantity=1)
        sold = event(E.PRODUCT_SOLD, 3, attributed_value=400,
                     data={"revenue_is_estimate": False, "allocations": [{
                         "product_lot_id": "egg", "source_animal_id": "g",
                         "quantity": 1, "realized_revenue": 400}]})
        report = assign_realized_credit(
            [buy, placed, harvested, sold],
            [cash("BUY_ANIMAL", -600, item="GOOSE"), cash("SELL", 400, item="EGG")], {}, 0)
        chain = next(a for a in report["allocations"] if a["kind"] == "animal")
        self.assertEqual(chain["cost"], 300)
        leftover = next(a for a in report["allocations"] if a["kind"] == "unsold_or_unrealized_input")
        self.assertEqual(leftover["shares"], [{"decision_id": 0, "gold": -300}])
        self.assertAlmostEqual(report["allocated_gold"], -200)

    def test_animal_feed_wheat_and_sold_fertilizer_share_one_chain(self):
        buy = event(E.ANIMAL_PURCHASED, 0, animal_lot_id="a0",
                    data={"total_cost": 300, "animal_lot_ids": ["a0"]})
        wheat = event(E.PRODUCT_PURCHASED, 1, purchase_id="wbuy", attributed_value=-25,
                      data={"total_cost": 25, "product_lot_ids": ["w1"], "unit_prices": [25]})
        placed = event(E.ANIMAL_PLACED, 2, animal_id="g", animal_lot_id="a0",
                       data={"animal_unit_cost": 300, "unit_index": 0})
        fed = event(E.ANIMAL_FED, 3, animal_id="g",
                    data={"unit_index": 0, "consumed_allocations": [{"product_lot_id": "w1"}]})
        harvested = event(E.HARVESTED, 4, animal_id="g", product_lot_id="egg", quantity=1)
        collected = event(E.FERTILIZER_COLLECTED, 5, animal_id="g", product_lot_id="fert")
        sold = event(E.PRODUCT_SOLD, 6, attributed_value=450,
                     data={"revenue_is_estimate": False, "allocations": [
                         {"product_lot_id": "egg", "source_animal_id": "g",
                          "quantity": 1, "realized_revenue": 400},
                         {"product_lot_id": "fert", "source_animal_id": "g",
                          "quantity": 1, "realized_revenue": 50}]})
        report = assign_realized_credit(
            [buy, wheat, placed, fed, harvested, collected, sold],
            [cash("BUY_ANIMAL", -300, item="GOOSE"), cash("BUY_PRODUCT", -25, item="WHEAT"),
             cash("SELL", 450, item="EGG")], {}, 0)
        chain = next(a for a in report["allocations"] if a["kind"] == "animal")
        self.assertEqual(chain["cost"], 325)
        self.assertAlmostEqual(chain["net_profit"], 125)
        self.assertEqual(chain["sold_units"], 2)
        self.assertAlmostEqual(report["allocated_gold"], 125)

    def test_animal_fertilizer_on_plant_does_not_double_claim_animal_cost(self):
        seed = purchase(quantity=1)
        planted = plant()
        buy = event(E.ANIMAL_PURCHASED, 2, animal_lot_id="a0",
                    data={"total_cost": 300, "animal_lot_ids": ["a0"]})
        placed = event(E.ANIMAL_PLACED, 3, animal_id="g", animal_lot_id="a0",
                       data={"animal_unit_cost": 300, "unit_index": 0})
        collected = event(E.FERTILIZER_COLLECTED, 4, animal_id="g", product_lot_id="fert")
        fertilized = event(E.FERTILIZED, 5, plant_id="p", data={"unit_index": 0,
            "consumed_allocations": [{"product_lot_id": "fert", "source_animal_id": "g"}]})
        harvested = harvest(6)
        sold = sale(7, revenue=50)
        report = assign_realized_credit(
            [seed, planted, buy, placed, collected, fertilized, harvested, sold],
            [cash("BUY_SEED", -10), cash("BUY_ANIMAL", -300, item="GOOSE"), cash("SELL", 50)], {}, 0)
        plant_chain = next(a for a in report["allocations"] if a["kind"] == "plant")
        self.assertEqual(plant_chain["cost"], 10)
        loss = next(a for a in report["allocations"] if a["kind"] == "unsold_animal")
        self.assertAlmostEqual(loss["net_profit"], -300)
        self.assertAlmostEqual(report["allocated_gold"], -260)

    def test_escaped_animal_loss_includes_lost_decision_oldest_heavy(self):
        buy = event(E.ANIMAL_PURCHASED, 0, animal_lot_id="a0",
                    data={"total_cost": 300, "animal_lot_ids": ["a0"]})
        placed = event(E.ANIMAL_PLACED, 1, animal_id="g", animal_lot_id="a0",
                       data={"animal_unit_cost": 300, "unit_index": 0})
        lost = event(E.ANIMAL_LOST, 2, animal_id="g", data={"reason": "ESCAPED", "unit_index": 0})
        report = assign_realized_credit(
            [buy, placed, lost], [cash("BUY_ANIMAL", -300, item="GOOSE")], {}, 0)
        loss = next(a for a in report["allocations"] if a["kind"] == "unsold_animal")
        shares = {s["decision_id"]: s["gold"] for s in loss["shares"]}
        self.assertEqual(set(shares), {0, 1, 2})
        self.assertLess(shares[0], shares[2])
        self.assertAlmostEqual(report["allocated_gold"], -300)

    def test_product_resale_partial_and_fertilizer_input_cost(self):
        raw = event(E.PRODUCT_PURCHASED, 1, purchase_id='fertbuy', attributed_value=-200,
            data={'total_cost': 200, 'product_lot_ids': ['f1', 'f2'], 'unit_prices': [100, 100]})
        fertilizer = event(E.FERTILIZED, 3, plant_id='p', data={'unit_index': 0,
                         'consumed_allocations': [{'product_lot_id': 'f1'}]})
        events = [purchase(quantity=1), raw, plant(2), fertilizer, harvest(4), sale(5, revenue=300),
                  sale(6, 'f2', identity=None, revenue=90, source_purchase_id='fertbuy')]
        report = assign_realized_credit(events, [cash('BUY_SEED', -10), cash('BUY_PRODUCT', -200),
                                                cash('SELL', 390)], {}, 0)
        chain = next(a for a in report['allocations'] if a['kind']=='plant')
        self.assertEqual(chain['cost'], 110)
        self.assertEqual(chain['net_profit'], 190)
        self.assertEqual(report['allocated_gold'], 180)
        self.assertFalse(any(a['kind']=='unsold_or_unrealized_input' for a in report['allocations']))
        report = assign_realized_credit([raw, sale(6, 'f1', identity=None, revenue=90,
                                        source_purchase_id='fertbuy')],
                                       [cash('BUY_PRODUCT', -200), cash('SELL', 90)], {}, 0)
        self.assertEqual(report['allocated_gold'], -110)
        self.assertTrue(any(a['net_profit']==-100 for a in report['allocations']))

    def test_unknown_sales_and_other_cash_are_explicit_not_fake_plant_chains(self):
        events = [sale(identity=None)]
        report = assign_realized_credit(events, [cash('SELL', 50), cash('HIRE', -8, step=6)],
                                        {(6,'market',0): 6}, 0)
        self.assertEqual(report['unknown_source_sale_revenue'], 50)
        self.assertEqual(report['allocated_gold'], 42)
        self.assertEqual(report['decision_credit'][6], -8/3000)
        with self.assertRaises(RuntimeError):
            assign_realized_credit(events, [cash('SELL', 51)], {}, 0)


class RealizedTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_no_potential_terminal_and_local_credit_stay_separate(self):
        model = RealizedChainActorCritic()
        with patch('reward.economic_potential_breakdown', side_effect=AssertionError('No potential')):
            _, rows = propose_chain_turn(model, observation())
        self.assertTrue(all(r.reward==0 for r in rows))
        report = {'reward_scale': 3000., 'decision_credit': {0: -.01}}
        finish_chain_episode(rows, -300, report)
        self.assertAlmostEqual(sum(r.reward for r in rows), -.1)
        self.assertEqual(rows[0].chain_credit, -.01)
        self.assertTrue(all(r.chain_credit==0 for r in rows[1:]))
        for row in rows:
            self.assertAlmostEqual(row.return_target, -.1)
            self.assertAlmostEqual(row.chain_advantage, row.chain_credit-row.chain_value)
        before = model.realized_value_head.weight.detach().clone()
        update_chain_policy(model, torch.optim.Adam(model.parameters(), lr=1e-4), rows,
                            epochs=1, minibatch_size=64)
        self.assertFalse(torch.equal(before, model.realized_value_head.weight))

    def test_local_credit_sign_reaches_actor_gradient(self):
        for advantage in [-.1, .1]:
            logp = torch.tensor(0., requires_grad=True)
            loss = clipped_objective(logp.exp(), advantage, .2)
            loss.backward()
            self.assertAlmostEqual(float(logp.grad), -advantage, places=6)

    def test_short_environment_both_sides_and_artifacts(self):
        args = parser().parse_args(['--run-dir', 'unused', '--episode-steps', '5', '--save-every', '1'])
        model = RealizedChainActorCritic()
        with tempfile.TemporaryDirectory() as directory:
            for episode in (1, 2):
                path = Path(directory)/str(episode)
                with patch('reward.economic_potential_breakdown', side_effect=AssertionError('No potential')):
                    rows, summary = collect_episode(model, args, episode, path)
                self.assertEqual(summary['cash_error'], 0)
                self.assertAlmostEqual(summary['chain_reconciliation_error'], 0)
                self.assertAlmostEqual(summary['chain_credit_sum'], (summary['money']-3000)/3000)
                self.assertAlmostEqual(summary['terminal_reward'], (summary['money']-summary['opponent_money'])/3000)
                for name in ['chain_credit.json', 'advantages.json', 'replay.json.gz', 'lifecycle_events.jsonl.gz']:
                    self.assertTrue((path/name).exists())

    def test_actual_engine_sale_chain_has_operation_ids_and_movement(self):
        configuration = {'episodeSteps': 30, 'turnsPerDay': 4, 'weedSpawnChance': 0., 'seed': 77}
        env = make('kaggriculture', configuration=configuration)
        tracker = FarmLifecycleTracker(env.state[0].observation, episode_number=1,
                                       player_index=0, turns_per_day=4)
        ledger = CashLedger([3000, 3000])
        # Units return to the shed at each day boundary; approach again.
        commands = [['PASS'], ['WEST'], ['PLANT','WHEAT'], ['WATER'], ['WEST'], ['WATER'],
                    ['PASS'], ['PASS'], ['WEST'], ['HARVEST'], ['EAST'], ['DROP']]
        refs = {}
        for step, command in enumerate(commands):
            market = [['BUY_SEED','WHEAT',1]] if step==0 else ([['SELL','WHEAT',2]] if step==11 else [])
            action = {'farmer': command, 'hands': [], 'market': market}
            observations = [deepcopy(s.observation) for s in env.state]
            for obs in observations:
                obs['step'] = step
            actions = [action, {'farmer':['PASS'], 'hands':[], 'market':[]}]
            executions = trace_market_executions(observations, actions, configuration)
            env.step(actions)
            after = deepcopy(env.state[0].observation)
            after['step'] = step+1
            local = {('unit',0): step*2}
            if market:
                local[('market',0)] = step*2+1
            refs.update({(step,*k):v for k,v in local.items()})
            tracker.observe_transition(observations[0], action, after,
                                       market_executions=executions[0], decision_refs=local)
            ledger.observe(step, executions, [f['money'] for f in observations[0]['farms']],
                           [f['money'] for f in after['farms']])
        report = assign_realized_credit(tracker.events, ledger.entries, refs, 0)
        self.assertEqual(report['unknown_source_sale_revenue'], 0)
        self.assertAlmostEqual(report['allocated_gold'], ledger.final_money[0]-3000)
        chain = next(a for a in report['allocations'] if a['kind']=='plant')
        self.assertEqual(chain['cost'], 10)
        # Requested two units, engine sold only the single harvested unit.
        self.assertEqual(chain['sold_units'], 1)
        ids = {s['decision_id'] for s in chain['shares']}
        self.assertEqual(ids, {1, 2, 4, 6, 8, 10, 16, 18, 20, 22, 23})


if __name__ == '__main__':
    unittest.main()
