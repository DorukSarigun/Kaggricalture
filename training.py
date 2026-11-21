"""Episode transition'larını PPO kayıtlarına ve update'lerine bağlar."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np
import torch

from lifecycle_credit import NO_TOKEN_CREDIT_ROLE, LifecycleTokenCreditRole
from model import KaggricultureActorCritic
from policy import SampledAction
from ppo import (
    PPOConfig,
    actor_critic_ppo_loss,
    collate_ppo_batch,
    compute_gae,
    optimizer_step,
    validate_terminal_actor_safety,
)
from reward import RewardConfig, TransitionReward, transition_reward
from span_credit import (
    ActionSpanRecord,
    RunningRoleRMS,
    SpanCreditRole,
    SpanTrainingCredit,
    build_action_span_records,
)


@dataclass(frozen=True, slots=True)
class RolloutTransition:
    sample: SampledAction
    reward: TransitionReward
    done: bool
    environment_step: int
    token_advantages: np.ndarray
    token_credit_roles: np.ndarray
    action_spans: tuple[ActionSpanRecord, ...]
    span_credits: tuple[SpanTrainingCredit, ...]


@dataclass(frozen=True, slots=True)
class PPORecord:
    sample: SampledAction
    advantage: float
    return_target: float
    token_advantages: np.ndarray
    token_credit_roles: np.ndarray
    action_spans: tuple[ActionSpanRecord, ...]
    span_credits: tuple[SpanTrainingCredit, ...]
    realized_return_target: float | None = None
    potential_offset: float = 0.0


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    minibatches: int
    skipped_minibatches: int
    epochs_completed: int
    stopped_for_kl: bool
    total_loss: float
    policy_loss: float
    lifecycle_policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    explained_variance: float
    gradient_norm: float
    unit_span_kl: float
    market_span_kl: float
    purchase_credit_kl: float
    farm_credit_kl: float
    trade_credit_kl: float
    hire_credit_kl: float
    unit_operation_kl: float
    market_operation_kl: float
    item_kl: float
    quantity_kl: float
    realized_value_loss: float = 0.0
    last_epoch_attempted_kl: float = 0.0
    target_kl: float = 0.0
    kl_skip_threshold: float = 0.0


class EpisodeRollout:
    """Tek oyuncu perspektifindeki sıralı episode/segment buffer'ı."""

    def __init__(
        self,
        reward_config: RewardConfig | None = None,
        ppo_config: PPOConfig | None = None,
    ) -> None:
        self.reward_config = reward_config or RewardConfig()
        self.ppo_config = ppo_config or PPOConfig()
        if self.reward_config.mode == "cash_progress" and self.ppo_config.gamma != 1.0:
            raise ValueError("cash_progress PPO gamma=1 gerektirir.")
        if self.reward_config.mode == "cash_progress" and self.ppo_config.span_policy_coef > 0:
            raise ValueError("cash_progress ile yerel span kredisi birlikte etkinleştirilemez.")
        if not np.isclose(self.reward_config.gamma, self.ppo_config.gamma):
            raise ValueError(
                "Potential shaping ve PPO aynı gamma değerini kullanmalı."
            )
        self.transitions: list[RolloutTransition] = []
        self._closed = False

    def add(
        self,
        sample: SampledAction,
        observation: Mapping[str, Any],
        next_observation: Mapping[str, Any],
        done: bool,
        lifecycle_credit: float = 0.0,
    ) -> TransitionReward:
        if self._closed:
            raise RuntimeError("Terminal transition'dan sonra episode'a veri eklenemez.")
        if not math.isfinite(lifecycle_credit):
            raise ValueError("lifecycle_credit sonlu olmalı.")
        if self.reward_config.mode == "cash_progress" and lifecycle_credit != 0:
            raise ValueError("cash_progress gerçekleşmiş nakde ayrıca lifecycle reward eklemez.")
        if self.ppo_config.realized_value_coef > 0 and lifecycle_credit != 0:
            raise ValueError("Gerçekleşmiş getiri critic'i ek lifecycle reward ile karıştırılamaz.")
        shaped = transition_reward(
            observation,
            next_observation,
            done,
            self.reward_config,
        )
        shaped = replace(
            shaped,
            total=shaped.total + float(lifecycle_credit),
            lifecycle_credit=float(lifecycle_credit),
        )
        environment_step = int(
            observation.get(
                "step",
                int(observation["day"])
                * self.reward_config.turns_per_day
                + int(observation["hour"]),
            )
        )
        self.transitions.append(
            RolloutTransition(
                sample=sample,
                reward=shaped,
                done=bool(done),
                environment_step=environment_step,
                token_advantages=np.zeros(
                    len(sample.target_token_ids),
                    dtype=np.float32,
                ),
                token_credit_roles=np.full(
                    len(sample.target_token_ids),
                    NO_TOKEN_CREDIT_ROLE,
                    dtype=np.int8,
                ),
                action_spans=build_action_span_records(sample, environment_step),
                span_credits=(),
            )
        )
        self._closed = bool(done)
        return shaped

    def apply_token_credit(
        self,
        *,
        environment_step: int,
        token_start: int,
        token_end: int,
        amount: float,
        role: LifecycleTokenCreditRole | None = None,
    ) -> int:
        """Geçmiş transition'daki ilgili stochastic tokenlara kredi ekler.

        Span inclusive'dir ve ``SampledAction.target_token_ids`` indekslerini
        kullanır. Zorunlu/yapısal tokenlar decision_mask ile otomatik elenir.
        Dönüş değeri gerçekten kredi alan token sayısıdır.
        """
        if not math.isfinite(amount):
            raise ValueError("Token credit amount sonlu olmalı.")
        if token_start < 0 or token_end < token_start:
            raise ValueError("Token credit span geçersiz.")

        matching_indices = [
            index
            for index, transition in enumerate(self.transitions)
            if transition.environment_step == int(environment_step)
        ]
        if len(matching_indices) != 1:
            raise ValueError(
                "Token credit environment step tam bir transition ile "
                f"eşleşmeli; step={environment_step}, matches={len(matching_indices)}"
            )

        transition_index = matching_indices[0]
        transition = self.transitions[transition_index]
        token_count = len(transition.sample.target_token_ids)
        if token_end >= token_count:
            raise ValueError(
                f"Token credit span dizi dışı: {token_start}..{token_end}, "
                f"token_count={token_count}"
            )
        selected = np.zeros(token_count, dtype=np.bool_)
        selected[token_start : token_end + 1] = True
        selected &= transition.sample.decision_mask
        credited_count = int(selected.sum())
        if credited_count == 0:
            return 0

        token_advantages = transition.token_advantages.copy()
        token_advantages[selected] += float(amount)
        token_credit_roles = transition.token_credit_roles.copy()
        if role is not None:
            token_credit_roles[selected] = np.int8(role.role_id)
        self.transitions[transition_index] = replace(
            transition,
            token_advantages=token_advantages,
            token_credit_roles=token_credit_roles,
        )
        return credited_count

    def apply_span_credit(
        self,
        *,
        action_id: str,
        amount: float,
        role: SpanCreditRole,
    ) -> bool:
        """Bir ekonomik krediyi tek semantik action span'ına ekler.

        Caller attribution assignment'larını action bazında önceden toplar.
        Bu metot aynı action için ikinci PPO terimini reddeder; böylece karma
        kaynaklı SELL tek ratio ile öğrenir. Span stochastic karar içermese de
        kredi kaydı saklanır; actor loss collate aşamasında onu eler.
        """

        if not math.isfinite(amount):
            raise ValueError("Span credit amount sonlu olmalı.")
        matches: list[tuple[int, ActionSpanRecord]] = []
        for transition_index, transition in enumerate(self.transitions):
            for span in transition.action_spans:
                if span.action_id == action_id:
                    matches.append((transition_index, span))
        if len(matches) != 1:
            raise ValueError(
                "Span credit action_id tam bir action span ile eşleşmeli; "
                f"action_id={action_id!r}, matches={len(matches)}"
            )
        transition_index, span = matches[0]
        transition = self.transitions[transition_index]
        credits = list(transition.span_credits)
        if any(credit.action_id == action_id for credit in credits):
            raise ValueError(
                f"Action span yalnızca bir training credit alabilir: {action_id}"
            )
        credits.append(
            SpanTrainingCredit(
                action_id=action_id,
                credit_role=role,
                amount=float(amount),
            )
        )
        self.transitions[transition_index] = replace(
            transition,
            span_credits=tuple(credits),
        )
        return span.has_stochastic_decision

    def records(self, bootstrap_value: float = 0.0) -> list[PPORecord]:
        """GAE hesaplar; truncated segment'te caller critic bootstrap verir."""
        if not self.transitions:
            return []
        if self.transitions[-1].done:
            bootstrap_value = 0.0
        if self.ppo_config.realized_value_coef > 0 and not self.transitions[-1].done:
            raise ValueError("Gerçekleşmiş getiri hedefi için tam episode gerekli; truncation kabul edilmez.")
        realized_returns: list[float | None] = [None] * len(self.transitions)
        if self.transitions[-1].done:
            future = 0.0
            for index in reversed(range(len(self.transitions))):
                future = self.transitions[index].reward.base + self.ppo_config.gamma * future
                realized_returns[index] = future

        rewards = torch.tensor(
            [[transition.reward.total] for transition in self.transitions],
            dtype=torch.float32,
        )
        values = torch.tensor(
            [[transition.sample.value] for transition in self.transitions],
            dtype=torch.float32,
        )
        dones = torch.tensor(
            [[transition.done] for transition in self.transitions],
            dtype=torch.bool,
        )
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            next_values=torch.tensor([bootstrap_value], dtype=torch.float32),
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
        )
        return [
            PPORecord(
                sample=transition.sample,
                advantage=float(advantages[index, 0]),
                return_target=float(returns[index, 0]),
                token_advantages=transition.token_advantages.copy(),
                token_credit_roles=transition.token_credit_roles.copy(),
                action_spans=transition.action_spans,
                span_credits=transition.span_credits,
                realized_return_target=realized_returns[index],
                potential_offset=self.reward_config.shaping_beta * transition.reward.current_potential,
            )
            for index, transition in enumerate(self.transitions)
        ]


def ppo_update(
    model: KaggricultureActorCritic,
    optimizer: torch.optim.Optimizer,
    records: Sequence[PPORecord],
    config: PPOConfig | None = None,
    epochs: int = 3,
    minibatch_size: int = 256,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    span_credit_scaler: RunningRoleRMS | None = None,
) -> PPOUpdateMetrics:
    """Donmuş rollout policy verisiyle bir PPO update'i yapar."""
    if not records:
        raise ValueError("PPO update için en az bir record gerekli.")
    if epochs <= 0 or minibatch_size <= 0:
        raise ValueError("epochs ve minibatch_size pozitif olmalı.")
    config = config or PPOConfig()
    validate_terminal_actor_safety(config)
    if device is None:
        device = next(model.parameters()).device

    metric_rows: list[tuple[float, ...]] = []
    stopped_for_kl = False
    skipped_minibatches = 0
    epochs_completed = 0
    skip_threshold = config.target_kl * config.kl_skip_multiplier

    if config.use_action_span_policy:
        if span_credit_scaler is None:
            raise ValueError(
                "Action-span PPO için kalıcı RunningRoleRMS gerekli."
            )
        span_credit_scaler.update(
            credit
            for record in records
            for credit in record.span_credits
        )

    for epoch in range(epochs):
        # torch.randperm CPU'da çalışırsa CUDA generator kabul etmez. Aynı
        # generator hem policy sampling hem PPO shuffle için kullanılabildiğinden,
        # permütasyonu generator'ın kendi cihazında üretiriz.
        permutation_device = generator.device if generator is not None else "cpu"
        permutation = torch.randperm(
            len(records),
            generator=generator,
            device=permutation_device,
        ).cpu().tolist()
        epoch_kls: list[float] = []
        for start in range(0, len(records), minibatch_size):
            indices = permutation[start : start + minibatch_size]
            selected = [records[index] for index in indices]
            batch = collate_ppo_batch(
                [record.sample for record in selected],
                [record.advantage for record in selected],
                [record.return_target for record in selected],
                token_advantages=[
                    record.token_advantages for record in selected
                ],
                token_credit_roles=[
                    record.token_credit_roles for record in selected
                ],
                action_spans=[record.action_spans for record in selected],
                span_credits=[record.span_credits for record in selected],
                span_credit_scaler=span_credit_scaler,
                device=device,
            )
            realized_returns = None
            potential_offsets = None
            if config.realized_value_coef > 0:
                if any(record.realized_return_target is None for record in selected):
                    raise ValueError("Eksik gerçekleşmiş getiri hedefi.")
                realized_returns = torch.tensor([r.realized_return_target for r in selected],
                                               dtype=torch.float32, device=device)
                potential_offsets = torch.tensor([r.potential_offset for r in selected],
                                                dtype=torch.float32, device=device)
            loss = actor_critic_ppo_loss(model, batch, config,
                                        realized_returns=realized_returns,
                                        potential_offsets=potential_offsets)
            minibatch_kl = float(loss.approximate_kl)
            epoch_kls.append(minibatch_kl)

            # Tek bir aykırı dizi minibatch KL'ini eşiğin üstüne itebilir.
            # Bu yalnızca o adımı atlamalı; geri kalan veriyi çöpe atmamalı.
            if minibatch_kl > skip_threshold:
                skipped_minibatches += 1
                continue

            gradient_norm = optimizer_step(model, optimizer, loss, config)
            metric_rows.append(
                (
                    float(loss.total.detach()),
                    float(loss.policy.detach()),
                    float(loss.lifecycle_policy.detach()),
                    float(loss.value.detach()),
                    float(loss.entropy.detach()),
                    minibatch_kl,
                    float(loss.clip_fraction),
                    float(loss.explained_variance),
                    gradient_norm,
                    float(loss.unit_span_kl),
                    float(loss.market_span_kl),
                    float(loss.purchase_credit_kl),
                    float(loss.farm_credit_kl),
                    float(loss.trade_credit_kl),
                    float(loss.hire_credit_kl),
                    float(loss.unit_operation_kl),
                    float(loss.market_operation_kl),
                    float(loss.item_kl),
                    float(loss.quantity_kl),
                    0.0 if loss.realized_value is None else float(loss.realized_value.detach()),
                )
            )
        epochs_completed = epoch + 1

        # Politikanın gerçekten uzaklaştığının ölçüsü tek bir minibatch değil,
        # epoch ortalamasıdır.
        if epoch_kls and float(np.mean(epoch_kls)) > config.target_kl:
            stopped_for_kl = True
            break

    if not metric_rows:
        raise RuntimeError(
            "PPO hiçbir minibatch uygulamadı; tüm minibatch'ler "
            f"{skip_threshold:.4f} KL eşiğini aştı."
        )
    means = np.asarray(metric_rows, dtype=np.float64).mean(axis=0)
    return PPOUpdateMetrics(
        minibatches=len(metric_rows),
        skipped_minibatches=skipped_minibatches,
        epochs_completed=epochs_completed,
        stopped_for_kl=stopped_for_kl,
        total_loss=float(means[0]),
        policy_loss=float(means[1]),
        lifecycle_policy_loss=float(means[2]),
        value_loss=float(means[3]),
        entropy=float(means[4]),
        approximate_kl=float(means[5]),
        clip_fraction=float(means[6]),
        explained_variance=float(means[7]),
        gradient_norm=float(means[8]),
        unit_span_kl=float(means[9]),
        market_span_kl=float(means[10]),
        purchase_credit_kl=float(means[11]),
        farm_credit_kl=float(means[12]),
        trade_credit_kl=float(means[13]),
        hire_credit_kl=float(means[14]),
        unit_operation_kl=float(means[15]),
        market_operation_kl=float(means[16]),
        item_kl=float(means[17]),
        quantity_kl=float(means[18]),
        realized_value_loss=float(means[19]),
        last_epoch_attempted_kl=float(np.mean(epoch_kls)),
        target_kl=config.target_kl,
        kl_skip_threshold=skip_threshold,
    )
