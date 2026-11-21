from types import SimpleNamespace
import unittest

import torch

from ppo import PPOConfig, optimizer_step
from train_scripted import build_parser, _validate_resume_configuration


class KLControlTests(unittest.TestCase):
    def test_defaults_match(self):
        args = build_parser().parse_args(["train"])
        self.assertEqual(args.target_kl, 0.020)
        self.assertEqual(PPOConfig().target_kl, args.target_kl)
        self.assertEqual(args.target_kl * args.kl_skip_multiplier, 0.040)

    def test_resume_override_is_explicit_and_only_allows_kl(self):
        args = build_parser().parse_args([
            "train", "--lifecycle-credit-mode", "off", "--no-entropy-schedule"
        ])
        checkpoint = {"run_configuration": {"target_kl": 0.015, "learning_rate": args.learning_rate}}
        with self.assertRaisesRegex(ValueError, "target_kl"):
            _validate_resume_configuration(args, checkpoint)
        args.override_kl_on_resume = True
        _validate_resume_configuration(args, checkpoint)
        args.learning_rate *= 2
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            _validate_resume_configuration(args, checkpoint)

    def test_optimizer_clips_gradients_and_reports_preclip_norm(self):
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = SimpleNamespace(total=model(torch.tensor([[100., 100.]])).sum())
        before = model.weight.detach().clone()
        norm = optimizer_step(model, optimizer, loss, PPOConfig(max_grad_norm=1))
        self.assertGreater(norm, 100)
        self.assertLessEqual(float(model.weight.grad.norm()), 1.00001)
        self.assertLessEqual(float((model.weight.detach() - before).norm()), 0.100001)
