from copy import deepcopy
import unittest

import numpy as np
import torch
from kaggle_environments import make

from action_encoding import ActionGrammarConfig, ActionHeadType, ActionToken, TOKENS_BY_HEAD
from cash_accounting import CashLedger
from compare_checkpoints import evaluation_cases, paired_comparison, validate_full_game_summary
from model import KaggricultureActorCritic
from policy import sample_action
from ppo import PPOConfig, actor_critic_ppo_loss, collate_ppo_batch, validate_terminal_actor_safety
from reward import terminal_gold_difference_reward
from span_credit import build_action_span_records
from market_execution import trace_market_executions
from tests.test_ppo import make_observation


PASS = {"farmer": ["PASS"], "hands": [], "market": []}


class CashAccountingTests(unittest.TestCase):
    def test_all_cash_operations_reconcile_both_players_with_real_engine(self):
        config = {"episodeSteps": 4, "seed": 10}
        env = make("kaggriculture", configuration=config)
        before = [deepcopy(s.observation) for s in env.state]
        action = {**PASS, "market": [["BUY_LAND"], ["HIRE"], ["BUY_ANIMAL", "GOOSE", 1],
                    ["BUY_SEED", "WHEAT", 2], ["BUY_PRODUCT", "FERTILIZER", 1],
                    ["SELL", "FERTILIZER", 1], ["BUY_PRODUCT", "WHEAT", 1]]}
        actions = [action, {**PASS, "market": [["BUY_LAND"], ["BUY_LAND"], ["HIRE"]]}]
        executions = trace_market_executions(before, actions, config)
        env.step(actions)
        ledger = CashLedger([3000, 3000])
        final_money = [f.money for f in env.state[0].observation.farms]
        ledger.observe(0, executions, [3000, 3000], final_money)
        summary = ledger.summary(0)
        self.assertEqual(set(summary["by_operation"]),
                         {"BUY_LAND", "HIRE", "BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT", "SELL"})
        self.assertAlmostEqual(sum(summary["by_operation"].values()), final_money[0] - 3000)
        self.assertEqual(ledger.summary(1)["by_operation"], {"BUY_LAND": -3000})
        self.assertEqual(summary["max_reconciliation_error"], 0)
        self.assertEqual(before[0]["farms"][0]["money"], 3000)  # tracer is read-only

    def test_missing_land_trace_fails_without_partial_commit(self):
        ledger = CashLedger([3000, 3000])
        with self.assertRaisesRegex(RuntimeError, "reconciliation"):
            ledger.observe(0, [(), ()], [3000, 3000], [2000, 3000])
        self.assertEqual(ledger.transitions, 0)
        self.assertEqual(ledger.entries, [])

    def test_wasted_investments_have_negative_terminal_outcome(self):
        for operation, cost in [("BUY_LAND", 1000), ("HIRE", 1)]:
            with self.subTest(operation=operation):
                env = make("kaggriculture", configuration={"episodeSteps": 3})
                env.step([{**PASS, "market": [[operation]]}, PASS])
                while not env.done:
                    env.step([PASS, PASS])
                self.assertAlmostEqual(terminal_gold_difference_reward(env.state[0].observation, True),
                                       -cost / 3000)


class TerminalPolicyTests(unittest.TestCase):
    def test_local_only_training_is_rejected_before_investment_actions(self):
        with self.assertRaisesRegex(ValueError, "global_actor_coef"):
            validate_terminal_actor_safety(PPOConfig(use_action_span_policy=True, global_actor_coef=0))
        validate_terminal_actor_safety(PPOConfig(use_action_span_policy=True))

    def test_every_stochastic_token_is_in_exactly_one_action_span(self):
        torch.manual_seed(4)
        model = KaggricultureActorCritic()
        for _ in range(4):
            sample = sample_action(model, make_observation())
            spans = build_action_span_records(sample, 0)
            coverage = np.zeros(len(sample.target_token_ids), dtype=int)
            for span in spans:
                coverage[span.token_start:span.token_end+1] += 1
            self.assertTrue(np.all(coverage[sample.decision_mask] == 1))
            self.assertEqual(spans[-1].operation, "STOP")

    def test_terminal_outcome_pushes_stop_hire_land_in_correct_direction(self):
        # No entropy, value gradient, or local lifecycle reward: isolate outcome credit.
        # Positive/negative advantage represents profitable/wasteful investment outcomes,
        # not an assertion that every investment itself is causally beneficial.
        for operation, token in [("STOP", ActionToken.STOP),
                                 ("HIRE", ActionToken.MARKET_HIRE),
                                 ("BUY_LAND", ActionToken.MARKET_BUY_LAND)]:
            for advantage in (-1.0, 1.0):
                with self.subTest(operation=operation, advantage=advantage):
                    torch.manual_seed(5)
                    model = KaggricultureActorCritic()
                    head = model.action_decoder.output_heads.heads["market_operation"]
                    token_index = TOKENS_BY_HEAD[ActionHeadType.MARKET_OPERATION].index(token)
                    with torch.no_grad():
                        head.weight.zero_()
                        head.bias.zero_()
                        head.bias[token_index] = 0.5
                    sample = sample_action(model, make_observation(), deterministic=True,
                        grammar_config=ActionGrammarConfig(max_market_orders=1))
                    spans = build_action_span_records(sample, 0)
                    self.assertTrue(any(s.operation == operation and s.has_stochastic_decision for s in spans))
                    batch = collate_ppo_batch([sample], [advantage], [sample.value], action_spans=[spans])
                    config = PPOConfig(use_action_span_policy=True, normalize_advantages=False,
                                       value_coef=0, entropy_coef=0, span_policy_coef=0)
                    loss = actor_critic_ppo_loss(model, batch, config)
                    loss.total.backward()
                    # Gradient descent raises selected probability for positive advantage.
                    self.assertLess(head.bias.grad[token_index].item() * advantage, 0)

    def test_optional_local_credit_cannot_change_default_actor_loss(self):
        from span_credit import SpanTrainingCredit, SpanCreditRole
        model = KaggricultureActorCritic()
        sample = sample_action(model, make_observation(), deterministic=True)
        spans = build_action_span_records(sample, 0)
        target = next(s for s in spans if s.has_stochastic_decision)
        plain = collate_ppo_batch([sample], [-1.0], [0.0], action_spans=[spans])
        credited = collate_ppo_batch([sample], [-1.0], [0.0], action_spans=[spans],
            span_credits=[(SpanTrainingCredit(target.action_id, SpanCreditRole.FARM, 1000),)])
        config = PPOConfig(use_action_span_policy=True, normalize_advantages=False)
        torch.testing.assert_close(actor_critic_ppo_loss(model, plain, config).total,
                                   actor_critic_ppo_loss(model, credited, config).total)


class PairedEvaluationTests(unittest.TestCase):
    def test_incomplete_masked_or_gifted_games_are_rejected(self):
        good = dict(steps=719, terminal_statuses=["DONE", "DONE"],
                    curriculum_stage="off", curriculum_starting_seeds={})
        validate_full_game_summary(good)
        for bad in (dict(steps=10), dict(terminal_statuses=["ERROR", "DONE"]),
                    dict(curriculum_stage="wheat_loop"), dict(curriculum_starting_seeds={"WHEAT": 8})):
            with self.assertRaises(RuntimeError):
                validate_full_game_summary({**good, **bad})

    def test_v4_optimizer_semantics_cannot_resume_as_v5(self):
        from train_scripted import build_parser, _validate_resume_configuration
        args = build_parser().parse_args(["train"])
        with self.assertRaisesRegex(ValueError, "v5"):
            _validate_resume_configuration(args, {"run_configuration": {
                "training_semantics_version": 4, "economic_span_scheme_version": 4,
                "lifecycle_credit": {"scheme_version": 4}}})

    def test_same_seeds_both_sides_and_matched_comparison(self):
        self.assertEqual(evaluation_cases([10, 20]), [(10, 0), (10, 1), (20, 0), (20, 1)])
        rows = [{"environment_seed": seed, "learner_player": side, "money_advantage": 10}
                for seed, side in evaluation_cases([10, 20])]
        candidates = [{**r, "money_advantage": 30} for r in reversed(rows)]
        self.assertEqual(paired_comparison(rows, candidates)["mean_advantage_delta"], 20)
        with self.assertRaises(ValueError):
            paired_comparison(rows, candidates[:-1])
        with self.assertRaises(ValueError):
            evaluation_cases([10, 10])


if __name__ == "__main__":
    unittest.main()
