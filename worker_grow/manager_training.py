"""Manager PPO: terminal gold gap + local cash/issue-close credits.

Every decision and every accepted minibatch is written to disk.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from structured_training import assign_advantages
from worker_grow.issues import input_list_price
from worker_grow.manager import ManagerDecision, manager_action_statistics

OPTIMIZER_VERSION = "gold-stream-rms-v1"
STREAM_UNIT_FLOOR = 1.0


def stream_unit(values):
    """RMS of one PPO stream. Credits stay gold; only the optimizer step is scaled."""
    if not values:
        return STREAM_UNIT_FLOOR
    return max(math.sqrt(sum(v * v for v in values) / len(values)), STREAM_UNIT_FLOOR)


def scaled_mse(pred, target):
    unit = target.detach().abs().mean().clamp(min=STREAM_UNIT_FLOOR)
    return 0.5 * ((pred - target) / unit).square().mean()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False, default=_json) + "\n")


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def apply_issue_rewards(rows, closed_issues):
    by_id = {row.decision_id: row for row in rows}
    for issue in closed_issues:
        if issue.decision_id not in by_id:
            raise ValueError(f"Issue close targets missing manager decision {issue.decision_id}")
        if issue.close_gold is None:
            raise ValueError("Closed issue is missing gold")
        by_id[issue.decision_id].local_credit += issue.close_gold


def apply_market_cash(rows, entries, player, step):
    market = [row for row in rows if row.kind == "market" and row.action and row.action[0] != "STOP"
              and row.step == step]
    cash = [entry for entry in entries if entry.player == player and entry.step == step]
    if len(cash) > len(market):
        raise ValueError("More cash executions than manager market decisions")
    for entry in cash:
        if entry.order_index >= len(market):
            raise ValueError("Cash order index is outside this turn's manager orders")
        market[entry.order_index].local_credit += entry.amount


PURCHASE_OPS = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND")


def _purchase_list_price(operation, item, prices):
    if operation in ("BUY_SEED", "BUY_ANIMAL") and item:
        try:
            return input_list_price(item)
        except KeyError:
            return 0.0
    if operation == "BUY_PRODUCT" and item:
        return float(prices.get(item, 0) or 0)
    return 0.0


def apply_unfilled_market(rows, executions, player, step, prices):
    """Charge catalog price for BUY/HIRE/LAND units the engine did not fill."""
    market = [row for row in rows if row.kind == "market" and row.action and row.action[0] != "STOP"
              and row.step == step]
    filled = {}
    for execution in executions:
        if execution.player != player:
            continue
        filled[execution.order_index] = execution
        if execution.order_index >= len(market):
            continue
        short = max(0, execution.requested_quantity - execution.executed_quantity)
        if not short:
            continue
        unit = _purchase_list_price(execution.operation, execution.item, prices)
        if unit <= 0:
            continue
        market[execution.order_index].local_credit -= short * unit
    for index, row in enumerate(market):
        if index in filled:
            continue
        operation = row.action[0]
        if operation not in PURCHASE_OPS:
            continue
        requested = 1 if operation in ("HIRE", "BUY_LAND") or len(row.action) < 3 else int(row.action[2])
        item = row.action[1] if len(row.action) > 1 else None
        unit = _purchase_list_price(operation, item, prices)
        if unit <= 0 or requested <= 0:
            continue
        row.local_credit -= requested * unit


def finish_manager_episode(rows, terminal_gold, closed_issues, gae_lambda=1.0, *, credit_report=None,
                           task_report=None, throughput_report=None):
    if not rows or not rows[-1].boundary:
        raise ValueError("Manager episode must end on a boundary")
    if rows[-1].done:
        raise ValueError("Manager episode was already finalized")
    if credit_report is None:
        apply_issue_rewards(rows, closed_issues)
    else:
        if any(r.local_credit != 0 or r.task_credit != 0 or r.throughput_credit != 0
               or r.reward != 0 or r.base_reward != 0 for r in rows):
            raise ValueError("Immediate cash/issue rewards cannot be mixed with realized DAG credit")
        byid={r.decision_id:r for r in rows}
        if len(byid)!=len(rows):
            raise ValueError("Duplicate manager decision ID")
        for decision, credit in credit_report['decision_credit'].items():
            if decision not in byid or byid[decision].boundary or not byid[decision].action:
                raise ValueError("Invalid manager economic credit target")
            byid[decision].local_credit=credit
        if not math.isclose(math.fsum(r.local_credit for r in rows),
                            credit_report['net_cash'], abs_tol=1e-8):
            raise ValueError("Manager credit does not reconcile to actual cash")
    if task_report is not None:
        byid = {r.decision_id: r for r in rows}
        if len(byid) != len(rows):
            raise ValueError("Duplicate manager decision ID")
        for decision, credit in task_report['decision_credit'].items():
            row = byid.get(decision)
            issue_target = row is not None and row.kind == 'contract' and row.action[:1] == ['ISSUE']
            sale_target = (row is not None and row.kind == 'market' and row.action[:2] == ['SELL', 'WHEAT']
                           and task_report.get('version') == 'issue-completion-dag-v1'
                           and task_report.get('decision_roles', {}).get(decision) == ['SELL', 'WHEAT'])
            if task_report.get('version') == 'issue-completion-dag-v1' and row is not None:
                if row.action[:2] != task_report.get('decision_roles', {}).get(decision):
                    raise ValueError('Task DAG action identity does not match rollout')
            if (row is None or row.boundary or not (issue_target or sale_target)
                    or not math.isfinite(credit) or credit > 0):
                raise ValueError("Invalid manager task cost target")
            row.task_credit = credit
        if not math.isclose(math.fsum(r.task_credit for r in rows),
                            task_report['task_credit'], abs_tol=1e-10):
            raise ValueError("Manager task cost does not reconcile")
    if throughput_report is not None:
        if throughput_report.get('version') != 'issue-throughput-v1':
            raise ValueError('Invalid throughput report')
        byid = {r.decision_id: r for r in rows}
        if len(byid) != len(rows):
            raise ValueError("Duplicate manager decision ID")
        for decision, credit in throughput_report['decision_credit'].items():
            row = byid.get(decision)
            if (row is None or row.boundary or row.kind != 'contract'
                    or row.action[:1] != ['ISSUE'] or not math.isfinite(credit)):
                raise ValueError("Invalid manager throughput target")
            row.throughput_credit = credit
        if not math.isclose(math.fsum(r.throughput_credit for r in rows),
                            throughput_report['throughput_credit'], abs_tol=1e-10):
            raise ValueError("Manager throughput credit does not reconcile")
    rows[-1].reward = rows[-1].base_reward = terminal_gold
    rows[-1].done = True
    assign_advantages(rows, gae_lambda=gae_lambda)
    for row in rows:
        row.chain_advantage = row.local_credit + row.task_credit - row.chain_value
        row.throughput_advantage = row.throughput_credit - row.throughput_value


def decision_record(row):
    return {
        "episode": row.episode,
        "decision_id": row.decision_id,
        "step": row.step,
        "kind": row.kind,
        "action": row.action,
        "boundary": row.boundary,
        "log_prob": row.log_prob,
        "value": row.value,
        "reward": row.reward,
        "local_credit": row.local_credit,
        "task_credit": row.task_credit,
        "throughput_credit": row.throughput_credit,
        "local_target": row.local_credit + row.task_credit,
        "advantage": row.advantage,
        "chain_value": row.chain_value,
        "chain_advantage": row.chain_advantage,
        "throughput_value": row.throughput_value,
        "throughput_advantage": row.throughput_advantage,
        "return_target": row.return_target,
        "stochastic": row.stochastic,
        "allowed_ops": list(row.allowed_ops),
        "allowed_items": list(row.allowed_items),
        "workload": list(getattr(row.inputs, "workload", ())),
    }


def update_manager(model, optimizer, transitions, *, epochs=3, minibatch_size=64,
                   clip_ratio=0.2, target_kl=0.02, max_grad_norm=1.0,
                   entropy_coef=0.02, value_coef=0.5, chain_value_coef=0.5,
                   terminal_policy_coef=1.0, chain_policy_coef=1.0,
                   throughput_policy_coef=1.0, throughput_value_coef=0.5,
                   batch_log=None, update_id=0):
    if not transitions or epochs < 1 or minibatch_size < 1:
        raise ValueError("Empty manager update")
    device = next(model.parameters()).device
    metrics, skipped, stopped = [], 0, False
    model.train()
    for epoch in range(epochs):
        attempted = []
        order = torch.randperm(len(transitions)).tolist()
        for start in range(0, len(order), minibatch_size):
            batch = [transitions[i] for i in order[start:start + minibatch_size]]
            contexts, values, local_values, throughput_values = model.body([row.inputs for row in batch])
            policy_rows = [row for row in batch
                           if row.kind != "boundary" and (row.sample is not None or row.kind == "contract")
                           and row.stochastic]
            terminal_unit = stream_unit([row.advantage for row in policy_rows])
            chain_unit = stream_unit([row.chain_advantage for row in policy_rows])
            throughput_unit = stream_unit([row.throughput_advantage for row in policy_rows])
            terminal, chain, throughput, entropy, kl_terms, local_indices = [], [], [], [], [], []
            for i, row in enumerate(batch):
                if row.kind == "boundary" or (row.sample is None and row.kind != "contract"):
                    continue
                if not row.stochastic:
                    continue
                logp, ent = manager_action_statistics(model, row, contexts[i])
                logratio = logp - row.log_prob
                ratio = logratio.clamp(-20, 20).exp()
                terminal.append(_clip(ratio, row.advantage / terminal_unit, clip_ratio))
                chain.append(_clip(ratio, row.chain_advantage / chain_unit, clip_ratio))
                throughput.append(_clip(ratio, row.throughput_advantage / throughput_unit, clip_ratio))
                entropy.append(ent)
                kl_terms.append((ratio - 1) - logratio)
                local_indices.append(i)
            zero = values.sum() * 0
            mean = lambda xs: torch.stack(xs).mean() if xs else zero
            terminal_loss, chain_loss, throughput_loss, ent = (
                mean(terminal), mean(chain), mean(throughput), mean(entropy))
            kl = float(mean(kl_terms).detach())
            attempted.append(kl)
            if kl > 2 * target_kl:
                skipped += 1
                if batch_log is not None:
                    append_jsonl(batch_log, _batch_payload(update_id, epoch, start, batch, kl, skipped=True))
                continue
            targets = torch.tensor([row.return_target for row in batch], dtype=torch.float32, device=device)
            value_loss = scaled_mse(values, targets)
            if local_indices:
                local_targets = torch.tensor([batch[i].local_credit + batch[i].task_credit for i in local_indices],
                                             dtype=torch.float32, device=device)
                local_loss = scaled_mse(local_values[local_indices], local_targets)
                throughput_targets = torch.tensor([batch[i].throughput_credit for i in local_indices],
                                                  dtype=torch.float32, device=device)
                throughput_value_loss = scaled_mse(throughput_values[local_indices], throughput_targets)
            else:
                local_loss = zero
                throughput_value_loss = zero
            total = (terminal_policy_coef * terminal_loss + chain_policy_coef * chain_loss
                     + throughput_policy_coef * throughput_loss
                     + value_coef * value_loss + chain_value_coef * local_loss
                     + throughput_value_coef * throughput_value_loss - entropy_coef * ent)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            row_metrics = {"loss": float(total.detach()), "terminal_policy_loss": float(terminal_loss.detach()),
                           "chain_policy_loss": float(chain_loss.detach()),
                           "throughput_policy_loss": float(throughput_loss.detach()),
                           "terminal_value_loss": float(value_loss.detach()),
                           "chain_value_loss": float(local_loss.detach()),
                           "throughput_value_loss": float(throughput_value_loss.detach()),
                           "entropy": float(ent.detach()),
                           "kl": kl, "gradient_norm_preclip": float(norm),
                           "terminal_adv_unit": terminal_unit, "chain_adv_unit": chain_unit,
                           "throughput_adv_unit": throughput_unit}
            metrics.append(row_metrics)
            if batch_log is not None:
                append_jsonl(batch_log, {**_batch_payload(update_id, epoch, start, batch, kl), **row_metrics})
        if np.mean(attempted) > target_kl:
            stopped = True
            break
    if not metrics:
        raise RuntimeError("No manager PPO minibatches accepted")
    return {**{key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]},
            "minibatches": len(metrics), "skipped_minibatches": skipped,
            "epochs_completed": epoch + 1, "kl_stopped": stopped}


def _clip(ratio, advantage, clip_ratio):
    advantage = torch.as_tensor(advantage, device=ratio.device, dtype=ratio.dtype)
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)


def _batch_payload(update_id, epoch, start, batch, kl, skipped=False):
    return {
        "event": "batch",
        "update": update_id,
        "epoch": epoch,
        "batch_start": start,
        "skipped": skipped,
        "kl": kl,
        "size": len(batch),
        "decisions": [decision_record(row) for row in batch],
    }
