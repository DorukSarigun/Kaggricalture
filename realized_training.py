"""Two explicit objectives: terminal game result and realized operation credit.

No calls to reward.transition_reward or any potential-value function. The
third output of this version's model is a LOCAL credit baseline, not a cash
return-to-go head. Legacy structured checkpoints are not resumable here.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from observation_encoding import encode_observation
from structured_actions import (StructuredActorCritic, OperationCursor, head_input,
                                sample_operation, action_statistics)
from structured_training import DecisionTransition, assign_advantages


class RealizedChainActorCritic(StructuredActorCritic):
    """value_head=terminal value; realized_value_head=local assigned-credit value.

    Architecture reused, checkpoint semantic version intentionally different.
    Neither baseline is conditioned on the current sampled action.
    """


@dataclass
class ChainTransition(DecisionTransition):
    chain_credit: float = 0.0
    chain_value: float = 0.0
    chain_advantage: float = 0.0


@torch.no_grad()
def propose_chain_turn(model, observation, grammar_config=None):
    cursor = OperationCursor(observation, grammar_config)
    original = encode_observation(observation)
    rows = []
    while not cursor.finished:
        sample = sample_operation(model, cursor, original)
        cursor.apply(sample.action)
        rows.append(ChainTransition(sample.inputs, sample, sample.value, 0.0, 0.0, False,
                                    chain_value=sample.realized_value))
    boundary = head_input(cursor, original, boundary=True)
    _, value, _ = model([boundary])
    rows.append(ChainTransition(boundary, None, float(value[0]), 0.0, 0.0, True))
    for row in rows:
        row.environment_step = int(observation.get("step", observation["day"]*24 + observation["hour"]))
    return deepcopy(cursor.actions), rows


def finish_chain_episode(rows, terminal_gold_difference, credit_report, gae_lambda=1.0):
    if not rows or not rows[-1].boundary:
        raise ValueError("Missing episode boundary")
    if any(r.reward != 0 or r.base_reward != 0 for r in rows):
        raise ValueError("Per-turn cash/potential reward must not enter chain training")
    scale = credit_report["reward_scale"]
    rows[-1].reward = rows[-1].base_reward = terminal_gold_difference / scale
    rows[-1].done = True
    assign_advantages(rows, gae_lambda)
    for decision in credit_report["decision_credit"]:
        if not 0 <= decision < len(rows) or rows[decision].sample is None:
            raise ValueError(f"Credit targets an invalid/settlement decision: {decision}")
    for i, row in enumerate(rows):
        row.chain_credit = credit_report["decision_credit"].get(i, 0.0)
        # No GAE, no cumulative sum, no re-discounting of assigned local credit.
        row.chain_advantage = row.chain_credit - row.chain_value


def clipped_objective(ratio, advantage, clip_ratio):
    return -torch.minimum(ratio * advantage, ratio.clamp(1-clip_ratio, 1+clip_ratio) * advantage)


def update_chain_policy(model, optimizer, transitions, *, epochs=3, minibatch_size=64,
                        clip_ratio=0.2, target_kl=0.02, max_grad_norm=1.0,
                        entropy_coef=0.02, value_coef=0.5, chain_value_coef=0.5,
                        terminal_policy_coef=1.0, chain_policy_coef=1.0):
    if not transitions or epochs < 1 or minibatch_size < 1:
        raise ValueError("Empty update or invalid batch/epoch count")
    device = next(model.parameters()).device
    metrics, skipped, stopped = [], 0, False
    model.train()
    for epoch in range(epochs):
        attempted_kl = []
        permutation = torch.randperm(len(transitions)).tolist()
        for start in range(0, len(permutation), minibatch_size):
            batch = [transitions[i] for i in permutation[start:start+minibatch_size]]
            context, values, local_values = model([r.inputs for r in batch])
            terminal_terms, chain_terms, entropy_terms, kl_terms, local_indices = [], [], [], [], []
            for i, row in enumerate(batch):
                if row.sample is None or not row.sample.stochastic:
                    continue
                logp, entropy = action_statistics(model, context[i], row.sample)
                logratio = logp - row.sample.log_prob
                ratio = logratio.clamp(-20, 20).exp()
                # Both objectives use gold/reward_scale units. Independently
                # standardizing one objective would silently change its weight.
                terminal_terms.append(clipped_objective(ratio, row.advantage, clip_ratio))
                # Do not center rare signed chain rewards into unrelated bonuses.
                chain_terms.append(clipped_objective(ratio, row.chain_advantage, clip_ratio))
                entropy_terms.append(entropy)
                kl_terms.append((ratio-1)-logratio)
                local_indices.append(i)
            zero = values.sum()*0
            mean = lambda terms: torch.stack(terms).mean() if terms else zero
            terminal_loss, chain_loss, entropy = mean(terminal_terms), mean(chain_terms), mean(entropy_terms)
            kl = float(mean(kl_terms).detach())
            attempted_kl.append(kl)
            if kl > 2*target_kl:
                skipped += 1
                continue
            targets = torch.tensor([r.return_target for r in batch], dtype=torch.float32, device=device)
            value_loss = 0.5*(values-targets).square().mean()
            if local_indices:
                local_targets = torch.tensor([batch[i].chain_credit for i in local_indices],
                                             dtype=torch.float32, device=device)
                local_loss = 0.5*(local_values[local_indices]-local_targets).square().mean()
            else:
                local_loss = zero
            total = (terminal_policy_coef*terminal_loss + chain_policy_coef*chain_loss
                     + value_coef*value_loss + chain_value_coef*local_loss - entropy_coef*entropy)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            metrics.append({"loss": float(total.detach()), "terminal_policy_loss": float(terminal_loss.detach()),
                "chain_policy_loss": float(chain_loss.detach()), "terminal_value_loss": float(value_loss.detach()),
                "chain_value_loss": float(local_loss.detach()), "entropy": float(entropy.detach()),
                "kl": kl, "gradient_norm_preclip": float(norm)})
        if np.mean(attempted_kl) > target_kl:
            stopped = True
            break
    if not metrics:
        raise RuntimeError("No PPO minibatches accepted")
    return {**{k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
        "minibatches": len(metrics), "skipped_minibatches": skipped, "epochs_completed": epoch+1,
        "kl_stopped": stopped, "last_epoch_attempted_kl": float(np.mean(attempted_kl))}
