"""İşlem-MDP rollout ve PPO; eski token PPO'dan bağımsızdır.

İç geçişlerde gamma=lambda=1. Fiziksel tur sonundaki chance/settlement
geçişinde lambda uygulanır. Rakip ve gün-sonu etkileri bir aksiyon tokenına
gerçekleşmiş ödül diye etiketlenmez. İç ödüller tahmindir; toplamları gerçek
transition_reward ile her tur tam uzlaştırılır.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from action_encoding import ActionGrammarConfig
from observation_encoding import encode_observation
from reward import RewardConfig, transition_reward
from structured_actions import (
    OperationCursor, HeadInput, OperationSample, head_input,
    sample_operation, action_statistics,
)


@dataclass
class DecisionTransition:
    inputs: HeadInput
    sample: OperationSample | None
    value: float
    reward: float
    base_reward: float
    boundary: bool
    done: bool = False
    advantage: float = 0.0
    return_target: float = 0.0
    realized_target: float = 0.0
    environment_step: int = 0


@torch.no_grad()
def propose_turn(model, observation, reward_config, grammar_config=None, deterministic=False):
    if reward_config.mode != "cash_progress" or reward_config.gamma != 1:
        raise ValueError("İşlem-MDP cash_progress ve gamma=1 gerektirir.")
    cursor = OperationCursor(observation, grammar_config)
    original = encode_observation(observation)
    transitions = []
    while not cursor.finished:
        before = cursor.observation()
        sample = sample_operation(model, cursor, original, deterministic)
        cursor.apply(sample.action)
        provisional = transition_reward(before, cursor.observation(), False, reward_config)
        transitions.append(DecisionTransition(sample.inputs, sample, sample.value,
                                              provisional.total, 0.0, False))
    boundary_input = head_input(cursor, original, boundary=True)
    _, boundary_value, _ = model([boundary_input])
    transitions.append(DecisionTransition(boundary_input, None, float(boundary_value[0]),
                                          0.0, 0.0, True))
    for row in transitions:
        row.environment_step = int(observation.get("step", observation["day"]*24 + observation["hour"]))
    return deepcopy(cursor.actions), transitions


def settle_turn(transitions, before, after, done, reward_config):
    """Gerçek ortam sonucunu yalnız turn boundary'ye yazar; hayali cash yok."""
    if not transitions or not transitions[-1].boundary:
        raise ValueError("Eksik tur sonu geçişi.")
    actual = transition_reward(before, after, done, reward_config)
    provisional_sum = sum(t.reward for t in transitions[:-1])
    boundary = transitions[-1]
    boundary.reward = actual.total - provisional_sum
    boundary.base_reward = actual.base
    boundary.done = bool(done)
    error = sum(t.reward for t in transitions) - actual.total
    if abs(error) > 1e-9:
        raise RuntimeError("İşlem ödülleri gerçek tur ödülüne uzlaşmadı.")
    return actual, error


def assign_advantages(transitions, gae_lambda=0.95):
    if not 0 <= gae_lambda <= 1:
        raise ValueError("lambda 0..1 olmalı.")
    if not transitions or not transitions[-1].done:
        raise ValueError("Tamamlanmış episode gerekli.")
    advantage = 0.0
    next_value = 0.0
    realized = 0.0
    for row in reversed(transitions):
        if row.done:
            advantage, next_value, realized = 0.0, 0.0, 0.0
        delta = row.reward + next_value - row.value
        # Emir sayısı fiziksel zamanı hızlandırmaz / ek lambda cezası yaratmaz.
        trace = gae_lambda if row.boundary else 1.0
        advantage = delta + trace * advantage
        realized = row.base_reward + realized
        row.advantage = advantage
        row.return_target = advantage + row.value
        row.realized_target = realized
        next_value = row.value


def update_policy(model, optimizer, transitions, *, epochs=3, minibatch_size=64,
                  clip_ratio=0.2, target_kl=0.02, max_grad_norm=1.0,
                  entropy_coef=0.02, value_coef=0.5, realized_coef=0.5):
    if not transitions or epochs < 1 or minibatch_size < 1:
        raise ValueError("Geçersiz PPO batch ayarı.")
    device = next(model.parameters()).device
    active = [r.advantage for r in transitions if r.sample and r.sample.stochastic]
    center = float(np.mean(active)) if active else 0.0
    scale = max(float(np.std(active)), 1e-8) if active else 1.0
    metrics, skipped, stopped = [], 0, False
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(len(transitions)).tolist()
        attempted_kl = []
        for start in range(0, len(permutation), minibatch_size):
            batch = [transitions[i] for i in permutation[start:start + minibatch_size]]
            context, values, realized = model([r.inputs for r in batch])
            policy_terms, entropy_terms, kl_terms = [], [], []
            for i, row in enumerate(batch):
                if row.sample is None or not row.sample.stochastic:
                    continue
                logp, entropy = action_statistics(model, context[i], row.sample)
                logratio = logp - row.sample.log_prob
                ratio = logratio.clamp(-20, 20).exp()
                a = (row.advantage - center) / scale
                policy_terms.append(-torch.minimum(ratio * a, ratio.clamp(1-clip_ratio, 1+clip_ratio) * a))
                entropy_terms.append(entropy)
                kl_terms.append((ratio - 1) - logratio)
            zero = values.sum() * 0
            policy_loss = torch.stack(policy_terms).mean() if policy_terms else zero
            entropy = torch.stack(entropy_terms).mean() if entropy_terms else zero
            kl = float(torch.stack(kl_terms).mean().detach()) if kl_terms else 0.0
            attempted_kl.append(kl)
            if kl > 2 * target_kl:
                skipped += 1
                continue
            targets = torch.tensor([r.return_target for r in batch], dtype=torch.float32, device=device)
            actual_targets = torch.tensor([r.realized_target for r in batch], dtype=torch.float32, device=device)
            value_loss = 0.5 * (values - targets).square().mean()
            realized_loss = 0.5 * (realized - actual_targets).square().mean()
            total = policy_loss + value_coef * value_loss + realized_coef * realized_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            metrics.append({"loss": float(total.detach()), "policy_loss": float(policy_loss.detach()),
                            "value_loss": float(value_loss.detach()), "realized_value_loss": float(realized_loss.detach()),
                            "entropy": float(entropy.detach()), "kl": kl, "gradient_norm_preclip": float(norm)})
        if np.mean(attempted_kl) > target_kl:
            stopped = True
            break
    if not metrics:
        raise RuntimeError("Hiçbir PPO minibatch uygulanamadı.")
    return {**{key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]},
            "minibatches": len(metrics), "skipped_minibatches": skipped,
            "epochs_completed": epoch + 1, "kl_stopped": stopped,
            "last_epoch_attempted_kl": float(np.mean(attempted_kl))}
