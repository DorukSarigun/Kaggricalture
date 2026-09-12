"""Masked otoregresif policy için PPO ve GAE temel bileşenleri."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch

from action_encoding import (
    ACTION_VOCAB_SIZE,
    HEAD_TYPE_ID_BY_TOKEN_ID,
    TOKEN_TO_ID,
    ActionHeadType,
    ActionToken,
)
from lifecycle_credit import NO_TOKEN_CREDIT_ROLE
from model import KaggricultureActorCritic
from policy import SampledAction
from span_credit import (
    ActionSection,
    ActionSpanRecord,
    RunningRoleRMS,
    SpanCreditRole,
    SpanTrainingCredit,
)


def validate_terminal_actor_safety(config: "PPOConfig") -> None:
    """Investment/full-action collection must not run on partial local credit alone."""
    if config.use_action_span_policy and (
        not math.isfinite(config.global_actor_coef) or config.global_actor_coef <= 0
    ):
        raise ValueError("Economic-span training requires global_actor_coef > 0; "
                         "HIRE/BUY_LAND cannot rely on partial lifecycle credit.")


@dataclass(frozen=True, slots=True)
class PPOConfig:
    gamma: float = 0.999
    # lambda=1.0 terminal-ağırlıklı hedefte bir episode'un tüm adımlarına
    # neredeyse aynı avantajı verir. 0.95 adım-başı ayrımı korur.
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_clip: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    # ActionHeadType sırasıyla: STRUCTURAL, UNIT_OPERATION, MARKET_OPERATION,
    # ITEM, QUANTITY. STRUCTURAL asla stochastic karar olmadığı için etkisizdir.
    entropy_head_weights: tuple[float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    max_grad_norm: float = 1.00
    target_kl: float = 0.015
    # Tek bir aykırı örnek minibatch KL'ini hedefin üstüne itebilir. O
    # minibatch atlanır; update ancak epoch ORTALAMASI hedefi aşarsa durur.
    kl_skip_multiplier: float = 2.0
    normalize_advantages: bool = True
    # Lifecycle kredileri ham para oranıdır. Normalize edilmezse ana
    # (birim varyansa çekilmiş) avantajlarla kıyaslanamaz bir ölçekte kalır
    # ve lifecycle_policy_coef'in etkisi eğitim boyunca sürüklenir.
    normalize_token_advantages: bool = True
    # purchase kredisi tüm siparişin netidir; ölçülen std'si üretim rollerinin
    # ~36 katı. Tek havuzda normalize edilince global std'yi purchase belirler
    # ve üretim kararları arasındaki fark ayırt edilemez hale gelir. Bu yüzden
    # baseline/ölçek rol bazında hesaplanır. Örneklemi bu eşiğin altında kalan
    # rol, kendi gürültülü std'si yerine global istatistiğe düşer.
    min_role_samples: int = 8
    lifecycle_policy_coef: float = 1.0
    # Yeni ekonomik v1 yolu. Legacy joint/token PPO varsayılan olarak kapalı
    # kalır; economic_span run'ı CLI üzerinden bunları açıkça etkinleştirir.
    use_action_span_policy: bool = False
    # Terminal gold hedefi her gerçek kararı (STOP dahil) eğitir.
    # Eksik kapsamdaki lifecycle surrogate yalnız açık bir ablation seçeneği.
    global_actor_coef: float = 1.0
    span_policy_coef: float = 0.0
    head_balanced_entropy: bool = False
    # PURCHASE, FARM, TRADE, HIRE role-balanced local loss ağırlıkları.
    span_credit_role_weights: tuple[float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma (0, 1] aralığında olmalı.")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda [0, 1] aralığında olmalı.")
        if self.clip_ratio <= 0.0 or self.value_clip <= 0.0:
            raise ValueError("PPO clip değerleri pozitif olmalı.")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm pozitif olmalı.")
        if self.kl_skip_multiplier < 1.0:
            raise ValueError("kl_skip_multiplier en az 1.0 olmalı.")
        if len(self.entropy_head_weights) != len(ActionHeadType):
            raise ValueError(
                "entropy_head_weights her ActionHeadType için bir değer içermeli."
            )
        if any(weight < 0.0 for weight in self.entropy_head_weights):
            raise ValueError("entropy_head_weights negatif olamaz.")
        if self.lifecycle_policy_coef < 0.0:
            raise ValueError("lifecycle_policy_coef negatif olamaz.")
        if self.global_actor_coef < 0.0 or self.span_policy_coef < 0.0:
            raise ValueError("Span PPO loss katsayıları negatif olamaz.")
        if len(self.span_credit_role_weights) != len(SpanCreditRole):
            raise ValueError(
                "span_credit_role_weights her SpanCreditRole için değer içermeli."
            )
        if any(weight < 0.0 for weight in self.span_credit_role_weights):
            raise ValueError("span_credit_role_weights negatif olamaz.")


@dataclass(frozen=True, slots=True)
class PPOTrainingBatch:
    own_categorical_ids: torch.Tensor
    own_numeric_features: torch.Tensor
    own_worker_positions: torch.Tensor
    opponent_categorical_ids: torch.Tensor
    opponent_numeric_features: torch.Tensor
    opponent_worker_positions: torch.Tensor
    global_features: torch.Tensor
    unit_positions_yx: torch.Tensor
    unit_features: torch.Tensor
    unit_mask: torch.Tensor
    input_token_ids: torch.Tensor
    target_token_ids: torch.Tensor
    context_unit_indices: torch.Tensor
    head_type_ids: torch.Tensor
    valid_token_masks: torch.Tensor
    sequence_mask: torch.Tensor
    decision_mask: torch.Tensor
    old_action_log_probs: torch.Tensor
    old_token_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    token_advantages: torch.Tensor
    token_credit_roles: torch.Tensor
    token_credit_mask: torch.Tensor
    returns: torch.Tensor
    span_starts: torch.Tensor
    span_ends: torch.Tensor
    span_sections: torch.Tensor
    span_old_log_probs: torch.Tensor
    span_mask: torch.Tensor
    credit_span_starts: torch.Tensor
    credit_span_ends: torch.Tensor
    credit_span_roles: torch.Tensor
    credit_span_values: torch.Tensor
    credit_span_old_log_probs: torch.Tensor
    credit_span_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class SequenceStatistics:
    action_log_probs: torch.Tensor
    token_log_probs: torch.Tensor
    mean_entropies: torch.Tensor
    weighted_mean_entropies: torch.Tensor
    decision_counts: torch.Tensor


@dataclass(frozen=True, slots=True)
class PPOLoss:
    total: torch.Tensor
    policy: torch.Tensor
    lifecycle_policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor
    explained_variance: torch.Tensor
    unit_span_kl: torch.Tensor
    market_span_kl: torch.Tensor
    purchase_credit_kl: torch.Tensor
    farm_credit_kl: torch.Tensor
    trade_credit_kl: torch.Tensor
    hire_credit_kl: torch.Tensor
    unit_operation_kl: torch.Tensor
    market_operation_kl: torch.Tensor
    item_kl: torch.Tensor
    quantity_kl: torch.Tensor


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float = 0.999,
    gae_lambda: float = 0.97,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zaman-major ``(T, B)`` rollout için GAE ve return hesaplar."""
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values ve dones aynı (T, B) shape'e sahip olmalı.")
    if rewards.ndim != 2:
        raise ValueError("GAE girdileri (T, B) olmalı.")
    if next_values.shape != rewards.shape[1:]:
        raise ValueError("next_values shape'i (B,) olmalı.")

    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros_like(next_values)
    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            following_value = next_values
        else:
            following_value = values[step + 1]
        nonterminal = 1.0 - dones[step].to(values.dtype)
        delta = rewards[step] + gamma * following_value * nonterminal - values[step]
        running_advantage = (
            delta
            + gamma * gae_lambda * nonterminal * running_advantage
        )
        advantages[step] = running_advantage
    return advantages, advantages + values


def _stack(samples: Sequence[SampledAction], field: str) -> np.ndarray:
    return np.stack(
        [getattr(sample.encoded_observation, field) for sample in samples]
    )


def collate_ppo_batch(
    samples: Sequence[SampledAction],
    advantages: Sequence[float] | np.ndarray | torch.Tensor,
    returns: Sequence[float] | np.ndarray | torch.Tensor,
    token_advantages: Sequence[np.ndarray] | None = None,
    token_credit_roles: Sequence[np.ndarray] | None = None,
    action_spans: Sequence[Sequence[ActionSpanRecord]] | None = None,
    span_credits: Sequence[Sequence[SpanTrainingCredit]] | None = None,
    span_credit_scaler: RunningRoleRMS | None = None,
    device: torch.device | str = "cpu",
) -> PPOTrainingBatch:
    """Değişken unit/token uzunluklarını pad ederek model batch'i kurar."""
    if not samples:
        raise ValueError("En az bir SampledAction gerekli.")
    batch_size = len(samples)
    if len(advantages) != batch_size or len(returns) != batch_size:
        raise ValueError("advantages ve returns sample sayısıyla eşleşmeli.")

    max_units = max(len(sample.encoded_observation.own_unit_positions_yx) for sample in samples)
    max_tokens = max(len(sample.input_token_ids) for sample in samples)
    unit_feature_dim = samples[0].encoded_observation.own_unit_features.shape[-1]

    unit_positions = np.full((batch_size, max_units, 2), -1, dtype=np.int64)
    unit_features = np.zeros(
        (batch_size, max_units, unit_feature_dim),
        dtype=np.float32,
    )
    unit_mask = np.zeros((batch_size, max_units), dtype=np.bool_)

    input_ids = np.full(
        (batch_size, max_tokens),
        TOKEN_TO_ID[ActionToken.BOS],
        dtype=np.int64,
    )
    target_ids = np.zeros((batch_size, max_tokens), dtype=np.int64)
    context_indices = np.full((batch_size, max_tokens), -1, dtype=np.int64)
    head_type_ids = np.zeros((batch_size, max_tokens), dtype=np.int64)
    valid_masks = np.zeros(
        (batch_size, max_tokens, ACTION_VOCAB_SIZE),
        dtype=np.bool_,
    )
    sequence_mask = np.zeros((batch_size, max_tokens), dtype=np.bool_)
    decision_mask = np.zeros((batch_size, max_tokens), dtype=np.bool_)
    old_token_log_probs = np.zeros(
        (batch_size, max_tokens),
        dtype=np.float32,
    )
    padded_token_advantages = np.zeros(
        (batch_size, max_tokens),
        dtype=np.float32,
    )
    padded_token_credit_roles = np.full(
        (batch_size, max_tokens),
        NO_TOKEN_CREDIT_ROLE,
        dtype=np.int64,
    )

    if action_spans is None:
        action_spans = [() for _ in samples]
    if span_credits is None:
        span_credits = [() for _ in samples]
    if len(action_spans) != batch_size or len(span_credits) != batch_size:
        raise ValueError("action_spans/span_credits sample sayısıyla eşleşmeli.")
    max_spans = max(1, max((len(row) for row in action_spans), default=0))
    max_credits = max(1, max((len(row) for row in span_credits), default=0))
    span_starts = np.zeros((batch_size, max_spans), dtype=np.int64)
    span_ends = np.zeros((batch_size, max_spans), dtype=np.int64)
    span_sections = np.zeros((batch_size, max_spans), dtype=np.int64)
    span_old_log_probs = np.zeros((batch_size, max_spans), dtype=np.float32)
    span_mask = np.zeros((batch_size, max_spans), dtype=np.bool_)
    credit_span_starts = np.zeros((batch_size, max_credits), dtype=np.int64)
    credit_span_ends = np.zeros((batch_size, max_credits), dtype=np.int64)
    credit_span_roles = np.zeros((batch_size, max_credits), dtype=np.int64)
    credit_span_values = np.zeros((batch_size, max_credits), dtype=np.float32)
    credit_span_old_log_probs = np.zeros(
        (batch_size, max_credits), dtype=np.float32
    )
    credit_span_mask = np.zeros((batch_size, max_credits), dtype=np.bool_)

    if token_advantages is not None and len(token_advantages) != batch_size:
        raise ValueError("token_advantages sample sayısıyla eşleşmeli.")
    if token_credit_roles is not None and len(token_credit_roles) != batch_size:
        raise ValueError("token_credit_roles sample sayısıyla eşleşmeli.")

    for row, sample in enumerate(samples):
        unit_count = len(sample.encoded_observation.own_unit_positions_yx)
        token_count = len(sample.input_token_ids)
        unit_positions[row, :unit_count] = sample.encoded_observation.own_unit_positions_yx
        unit_features[row, :unit_count] = sample.encoded_observation.own_unit_features
        unit_mask[row, :unit_count] = True

        input_ids[row, :token_count] = sample.input_token_ids
        target_ids[row, :token_count] = sample.target_token_ids
        context_indices[row, :token_count] = sample.context_unit_indices
        head_type_ids[row, :token_count] = sample.head_type_ids
        valid_masks[row, :token_count] = sample.valid_token_masks
        sequence_mask[row, :token_count] = True
        decision_mask[row, :token_count] = sample.decision_mask
        old_token_log_probs[row, :token_count] = sample.token_log_probs
        if token_advantages is not None:
            row_advantages = np.asarray(
                token_advantages[row],
                dtype=np.float32,
            )
            if row_advantages.shape != (token_count,):
                raise ValueError(
                    "Her token_advantages satırı kendi token sayısıyla "
                    f"eşleşmeli; row={row}, shape={row_advantages.shape}, "
                    f"expected={(token_count,)}"
                )
            padded_token_advantages[row, :token_count] = row_advantages
        if token_credit_roles is not None:
            row_roles = np.asarray(token_credit_roles[row], dtype=np.int64)
            if row_roles.shape != (token_count,):
                raise ValueError(
                    "Her token_credit_roles satırı kendi token sayısıyla "
                    f"eşleşmeli; row={row}, shape={row_roles.shape}, "
                    f"expected={(token_count,)}"
                )
            padded_token_credit_roles[row, :token_count] = row_roles

        span_lookup: dict[str, ActionSpanRecord] = {}
        for span_index, span in enumerate(action_spans[row]):
            if span.action_id in span_lookup:
                raise ValueError(f"Tekrarlanan action_id: {span.action_id}")
            span_lookup[span.action_id] = span
            if not (0 <= span.token_start <= span.token_end < token_count):
                raise ValueError(
                    f"Action span token aralığı geçersiz: {span.action_id} "
                    f"{span.token_start}..{span.token_end}, tokens={token_count}"
                )
            inferred_has_decision = bool(
                sample.decision_mask[span.token_start : span.token_end + 1].any()
            )
            if inferred_has_decision != span.has_stochastic_decision:
                raise ValueError(
                    f"Action span stochastic kararı sample ile uyuşmuyor: {span.action_id}"
                )
            span_starts[row, span_index] = span.token_start
            span_ends[row, span_index] = span.token_end
            span_sections[row, span_index] = int(span.section)
            span_old_log_probs[row, span_index] = span.old_log_probability
            span_mask[row, span_index] = inferred_has_decision

        for credit_index, credit in enumerate(span_credits[row]):
            span = span_lookup.get(credit.action_id)
            if span is None:
                raise ValueError(
                    f"Span credit bilinmeyen action_id kullanıyor: {credit.action_id}"
                )
            credit_span_starts[row, credit_index] = span.token_start
            credit_span_ends[row, credit_index] = span.token_end
            credit_span_roles[row, credit_index] = int(credit.credit_role)
            credit_span_values[row, credit_index] = (
                span_credit_scaler.transform(credit)
                if span_credit_scaler is not None
                else float(credit.amount)
            )
            credit_span_old_log_probs[row, credit_index] = span.old_log_probability
            credit_span_mask[row, credit_index] = span.has_stochastic_decision

    if not np.array_equal(
        decision_mask,
        (valid_masks.sum(axis=-1) > 1) & sequence_mask,
    ):
        raise ValueError("Kaydedilmiş decision_mask geçerlilik maskeleriyle uyuşmuyor.")
    matching_head = (
        HEAD_TYPE_ID_BY_TOKEN_ID.reshape(1, 1, -1)
        == head_type_ids[..., None]
    )
    if np.any(valid_masks & ~matching_head):
        raise ValueError("Token maskesi kaydedilmiş output head türüyle uyuşmuyor.")

    def tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=device)

    return PPOTrainingBatch(
        own_categorical_ids=tensor(_stack(samples, "own_categorical_ids"), torch.long),
        own_numeric_features=tensor(_stack(samples, "own_numeric_features"), torch.float32),
        own_worker_positions=tensor(_stack(samples, "own_worker_positions"), torch.float32),
        opponent_categorical_ids=tensor(
            _stack(samples, "opponent_categorical_ids"), torch.long
        ),
        opponent_numeric_features=tensor(
            _stack(samples, "opponent_numeric_features"), torch.float32
        ),
        opponent_worker_positions=tensor(
            _stack(samples, "opponent_worker_positions"), torch.float32
        ),
        global_features=tensor(_stack(samples, "global_features"), torch.float32),
        unit_positions_yx=tensor(unit_positions, torch.long),
        unit_features=tensor(unit_features, torch.float32),
        unit_mask=tensor(unit_mask, torch.bool),
        input_token_ids=tensor(input_ids, torch.long),
        target_token_ids=tensor(target_ids, torch.long),
        context_unit_indices=tensor(context_indices, torch.long),
        head_type_ids=tensor(head_type_ids, torch.long),
        valid_token_masks=tensor(valid_masks, torch.bool),
        sequence_mask=tensor(sequence_mask, torch.bool),
        decision_mask=tensor(decision_mask, torch.bool),
        old_action_log_probs=torch.as_tensor(
            [sample.action_log_prob for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        old_token_log_probs=tensor(old_token_log_probs, torch.float32),
        old_values=torch.as_tensor(
            [sample.value for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
        token_advantages=tensor(padded_token_advantages, torch.float32),
        token_credit_roles=tensor(padded_token_credit_roles, torch.int64),
        token_credit_mask=tensor(
            (padded_token_advantages != 0.0) & decision_mask,
            torch.bool,
        ),
        returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        span_starts=tensor(span_starts, torch.long),
        span_ends=tensor(span_ends, torch.long),
        span_sections=tensor(span_sections, torch.long),
        span_old_log_probs=tensor(span_old_log_probs, torch.float32),
        span_mask=tensor(span_mask, torch.bool),
        credit_span_starts=tensor(credit_span_starts, torch.long),
        credit_span_ends=tensor(credit_span_ends, torch.long),
        credit_span_roles=tensor(credit_span_roles, torch.long),
        credit_span_values=tensor(credit_span_values, torch.float32),
        credit_span_old_log_probs=tensor(
            credit_span_old_log_probs, torch.float32
        ),
        credit_span_mask=tensor(credit_span_mask, torch.bool),
    )


def _normalize_token_advantages(
    advantages: torch.Tensor,
    *,
    roles: torch.Tensor | None,
    min_role_samples: int,
) -> torch.Tensor:
    """Lifecycle kredilerini baseline'a çeker ve birim varyansa ölçekler.

    Rol bilgisi varsa istatistikler rol içinde hesaplanır. Aksi halde tüm
    aktif tokenlar tek havuz sayılır (eski davranış).
    """

    def standardize(values: torch.Tensor) -> torch.Tensor:
        return (values - values.mean()) / (values.std(unbiased=False) + 1e-8)

    if roles is None:
        return standardize(advantages)

    normalized = standardize(advantages)
    for role_id in torch.unique(roles):
        if int(role_id) == NO_TOKEN_CREDIT_ROLE:
            continue
        member = roles == role_id
        if int(member.sum()) < max(min_role_samples, 2):
            continue
        normalized = normalized.masked_scatter(
            member,
            standardize(advantages[member]),
        )
    return normalized


def span_log_probabilities(
    token_log_probs: torch.Tensor,
    decision_mask: torch.Tensor,
    span_starts: torch.Tensor,
    span_ends: torch.Tensor,
    span_mask: torch.Tensor,
) -> torch.Tensor:
    """Inclusive span aralıklarında stochastic token log-prob toplamı."""

    if token_log_probs.shape != decision_mask.shape:
        raise ValueError("token_log_probs ve decision_mask shape'leri eşleşmeli.")
    if not (
        span_starts.shape
        == span_ends.shape
        == span_mask.shape
    ):
        raise ValueError("Span index/mask tensor shape'leri eşleşmeli.")
    if token_log_probs.ndim != 2 or span_starts.ndim != 2:
        raise ValueError("Token ve span tensorları iki boyutlu olmalı.")
    if token_log_probs.shape[0] != span_starts.shape[0]:
        raise ValueError("Token ve span batch boyutları eşleşmeli.")

    token_count = token_log_probs.shape[1]
    active = span_mask.bool()
    if torch.any(active & (span_starts < 0)) or torch.any(
        active & (span_ends < span_starts)
    ) or torch.any(active & (span_ends >= token_count)):
        raise ValueError("Aktif span token aralığı geçersiz.")
    positions = torch.arange(
        token_count,
        device=token_log_probs.device,
    ).view(1, 1, token_count)
    membership = (
        (positions >= span_starts.unsqueeze(-1))
        & (positions <= span_ends.unsqueeze(-1))
        & active.unsqueeze(-1)
        & decision_mask.bool().unsqueeze(1)
    )
    return (
        token_log_probs.unsqueeze(1)
        * membership.to(token_log_probs.dtype)
    ).sum(dim=-1)


def _group_balanced_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    groups: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_ratio: float,
    group_weights: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    """Önce grup içinde, sonra aktif gruplar arasında ağırlıklı PPO ortalaması."""

    if not (
        new_log_probs.shape
        == old_log_probs.shape
        == advantages.shape
        == groups.shape
        == mask.shape
    ):
        raise ValueError("Grouped PPO tensor shape'leri eşleşmeli.")
    active = mask.bool()
    zero = new_log_probs.new_zeros(())
    if not torch.any(active):
        return zero, zero, zero, {}

    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    losses = -torch.minimum(unclipped, clipped)
    kl_values = (ratio - 1.0) - log_ratio
    clipped_flags = (torch.abs(ratio - 1.0) > clip_ratio).to(ratio.dtype)

    weighted_loss = zero
    weighted_kl = zero
    weighted_clip = zero
    weight_sum = zero
    group_kls: dict[int, torch.Tensor] = {}
    for group_id, raw_weight in enumerate(group_weights):
        member = active & (groups == group_id)
        if not torch.any(member) or raw_weight <= 0.0:
            continue
        weight = new_log_probs.new_tensor(float(raw_weight))
        group_loss = losses[member].mean()
        group_kl = kl_values[member].mean()
        group_clip = clipped_flags[member].mean()
        weighted_loss = weighted_loss + weight * group_loss
        weighted_kl = weighted_kl + weight * group_kl
        weighted_clip = weighted_clip + weight * group_clip
        weight_sum = weight_sum + weight
        group_kls[group_id] = group_kl.detach()
    if float(weight_sum.detach()) == 0.0:
        return zero, zero, zero, {}
    return (
        weighted_loss / weight_sum,
        weighted_kl / weight_sum,
        weighted_clip / weight_sum,
        group_kls,
    )


def masked_sequence_statistics(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    valid_token_masks: torch.Tensor,
    sequence_mask: torch.Tensor,
    decision_mask: torch.Tensor | None = None,
    head_type_ids: torch.Tensor | None = None,
    entropy_head_weights: Sequence[float] | None = None,
    head_balanced_entropy: bool = False,
) -> SequenceStatistics:
    """Joint action log-probability ve token-başına entropy hesaplar."""
    if logits.shape != valid_token_masks.shape:
        raise ValueError("logits ve valid_token_masks aynı (B, T, V) shape'te olmalı.")
    if logits.shape[:2] != target_token_ids.shape:
        raise ValueError("target_token_ids (B, T) olmalı.")
    if sequence_mask.shape != target_token_ids.shape:
        raise ValueError("sequence_mask (B, T) olmalı.")

    valid_masks = valid_token_masks.bool().clone()
    # Padding satırında softmax'in tamamı -inf olmasın; bu satırlar loss'ta
    # sequence_mask ile zaten atılır.
    valid_masks[..., 0] |= ~sequence_mask.bool()
    valid_counts = valid_masks.sum(dim=-1)
    if torch.any(valid_counts[sequence_mask.bool()] == 0):
        raise ValueError("Gerçek sequence adımında en az bir geçerli token olmalı.")

    masked_logits = logits.masked_fill(~valid_masks, -torch.inf)
    log_probabilities = torch.log_softmax(masked_logits, dim=-1)
    chosen_valid = torch.gather(
        valid_masks,
        dim=-1,
        index=target_token_ids.unsqueeze(-1),
    ).squeeze(-1)
    if torch.any(sequence_mask.bool() & ~chosen_valid):
        raise ValueError("Target tokenlardan en az biri kendi maskesinde geçersiz.")

    chosen_log_probs = torch.gather(
        log_probabilities,
        dim=-1,
        index=target_token_ids.unsqueeze(-1),
    ).squeeze(-1)
    inferred_decisions = (valid_counts > 1) & sequence_mask.bool()
    if decision_mask is not None:
        if not torch.equal(decision_mask.bool(), inferred_decisions):
            raise ValueError("decision_mask, valid token sayılarıyla uyuşmuyor.")
    decision_mask = inferred_decisions

    action_log_probs = (
        chosen_log_probs * decision_mask.to(chosen_log_probs.dtype)
    ).sum(dim=-1)

    probabilities = torch.softmax(masked_logits, dim=-1)
    safe_log_probabilities = torch.where(
        valid_masks,
        log_probabilities,
        torch.zeros_like(log_probabilities),
    )
    token_entropies = -(
        probabilities * safe_log_probabilities
    ).sum(dim=-1)
    decision_counts = decision_mask.sum(dim=-1)
    decision_weights = decision_mask.to(token_entropies.dtype)
    normalizer = decision_counts.clamp(min=1).to(token_entropies.dtype)
    mean_entropies = (token_entropies * decision_weights).sum(dim=-1) / normalizer

    # Keşif çöküşü kafalar arasında eşit dağılmaz: hareket kafası doğal olarak
    # yüksek entropili kalırken market kafası deterministikleşebilir. Tek bir
    # global katsayı bu ikisini ayıramadığı için ağırlık desteklenir.
    if entropy_head_weights is None:
        weighted_mean_entropies = mean_entropies
    else:
        if head_type_ids is None:
            raise ValueError(
                "entropy_head_weights kullanmak için head_type_ids gerekli."
            )
        if head_type_ids.shape != target_token_ids.shape:
            raise ValueError("head_type_ids (B, T) olmalı.")
        weights = torch.as_tensor(
            tuple(entropy_head_weights),
            dtype=token_entropies.dtype,
            device=token_entropies.device,
        )
        if torch.any(weights < 0.0):
            raise ValueError("entropy_head_weights negatif olamaz.")
        if torch.any(head_type_ids >= weights.numel()) or torch.any(head_type_ids < 0):
            raise ValueError("head_type_ids entropy_head_weights aralığı dışında.")
        if not head_balanced_entropy:
            weighted_mean_entropies = (
                token_entropies * weights[head_type_ids] * decision_weights
            ).sum(dim=-1) / normalizer
        else:
            weighted_sums = torch.zeros_like(mean_entropies)
            active_weight_sums = torch.zeros_like(mean_entropies)
            for head_id in range(len(ActionHeadType)):
                head_mask = decision_mask & (head_type_ids == head_id)
                head_counts = head_mask.sum(dim=-1)
                active = head_counts > 0
                head_means = (
                    (token_entropies * head_mask.to(token_entropies.dtype)).sum(dim=-1)
                    / head_counts.clamp(min=1).to(token_entropies.dtype)
                )
                weight = weights[head_id]
                weighted_sums = weighted_sums + torch.where(
                    active,
                    head_means * weight,
                    torch.zeros_like(head_means),
                )
                active_weight_sums = active_weight_sums + active.to(
                    token_entropies.dtype
                ) * weight
            weighted_mean_entropies = torch.where(
                active_weight_sums > 0.0,
                weighted_sums / active_weight_sums.clamp(min=1e-8),
                torch.zeros_like(weighted_sums),
            )

    return SequenceStatistics(
        action_log_probs=action_log_probs,
        token_log_probs=chosen_log_probs,
        mean_entropies=mean_entropies,
        weighted_mean_entropies=weighted_mean_entropies,
        decision_counts=decision_counts,
    )


def ppo_loss(
    new_action_log_probs: torch.Tensor,
    old_action_log_probs: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    mean_entropies: torch.Tensor,
    config: PPOConfig | None = None,
    new_token_log_probs: torch.Tensor | None = None,
    old_token_log_probs: torch.Tensor | None = None,
    token_advantages: torch.Tensor | None = None,
    token_credit_roles: torch.Tensor | None = None,
    token_credit_mask: torch.Tensor | None = None,
    new_span_log_probs: torch.Tensor | None = None,
    old_span_log_probs: torch.Tensor | None = None,
    span_sections: torch.Tensor | None = None,
    span_mask: torch.Tensor | None = None,
    new_credit_span_log_probs: torch.Tensor | None = None,
    old_credit_span_log_probs: torch.Tensor | None = None,
    credit_span_values: torch.Tensor | None = None,
    credit_span_roles: torch.Tensor | None = None,
    credit_span_mask: torch.Tensor | None = None,
    head_type_ids: torch.Tensor | None = None,
    decision_mask: torch.Tensor | None = None,
) -> PPOLoss:
    config = config or PPOConfig()
    if config.normalize_advantages and advantages.numel() > 1:
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

    log_ratio = new_action_log_probs - old_action_log_probs
    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
    unclipped_objective = ratio * advantages
    clipped_objective = ratio.clamp(
        1.0 - config.clip_ratio,
        1.0 + config.clip_ratio,
    ) * advantages
    policy_loss = -torch.minimum(
        unclipped_objective,
        clipped_objective,
    ).mean()
    approximate_kl = ((ratio - 1.0) - log_ratio).mean().detach()
    clip_fraction = (
        (torch.abs(ratio - 1.0) > config.clip_ratio)
        .to(torch.float32)
        .mean()
        .detach()
    )
    unit_span_kl = policy_loss.new_zeros(())
    market_span_kl = policy_loss.new_zeros(())
    purchase_credit_kl = policy_loss.new_zeros(())
    farm_credit_kl = policy_loss.new_zeros(())
    trade_credit_kl = policy_loss.new_zeros(())
    hire_credit_kl = policy_loss.new_zeros(())
    unit_operation_kl = policy_loss.new_zeros(())
    market_operation_kl = policy_loss.new_zeros(())
    item_kl = policy_loss.new_zeros(())
    quantity_kl = policy_loss.new_zeros(())
    global_group_kls: dict[int, torch.Tensor] = {}

    if config.use_action_span_policy:
        global_span_inputs = (
            new_span_log_probs,
            old_span_log_probs,
            span_sections,
            span_mask,
        )
        if any(value is None for value in global_span_inputs):
            raise ValueError("Action-span global PPO tensorları birlikte verilmelidir.")
        assert new_span_log_probs is not None
        assert old_span_log_probs is not None
        assert span_sections is not None
        assert span_mask is not None
        span_advantages = advantages.unsqueeze(-1).expand_as(new_span_log_probs)
        policy_loss, approximate_kl, clip_fraction, global_group_kls = (
            _group_balanced_ppo_loss(
                new_span_log_probs,
                old_span_log_probs,
                span_advantages,
                span_sections,
                span_mask,
                clip_ratio=config.clip_ratio,
                group_weights=(1.0, 1.0),
            )
        )
        unit_span_kl = global_group_kls.get(
            int(ActionSection.UNIT), unit_span_kl
        )
        market_span_kl = global_group_kls.get(
            int(ActionSection.MARKET), market_span_kl
        )

    lifecycle_policy_loss = policy_loss.new_zeros(())
    token_inputs = (
        new_token_log_probs,
        old_token_log_probs,
        token_advantages,
        token_credit_mask,
    )
    if any(value is not None for value in token_inputs):
        if any(value is None for value in token_inputs):
            raise ValueError(
                "Token-level PPO için log-prob, advantage ve mask birlikte verilmelidir."
            )
        assert new_token_log_probs is not None
        assert old_token_log_probs is not None
        assert token_advantages is not None
        assert token_credit_mask is not None
        if not (
            new_token_log_probs.shape
            == old_token_log_probs.shape
            == token_advantages.shape
            == token_credit_mask.shape
        ):
            raise ValueError("Token-level PPO tensor shape'leri aynı olmalı.")
        active = token_credit_mask.bool()
        if torch.any(active):
            token_log_ratio = (
                new_token_log_probs[active] - old_token_log_probs[active]
            )
            token_ratio = torch.exp(token_log_ratio.clamp(-20.0, 20.0))
            active_advantages = token_advantages[active]
            # Ortalamayı çıkarmak eksik olan baseline'ı verir: kredi artık
            # "kâr pozitif miydi" değil, "bu batch'teki diğer zincirlerden iyi
            # miydi" sorusunu yanıtlar. Bölme ise ana avantajlarla aynı ölçeğe
            # taşır, böylece lifecycle_policy_coef anlamlı bir ağırlık olur.
            if (
                config.normalize_token_advantages
                and active_advantages.numel() > 1
            ):
                active_advantages = _normalize_token_advantages(
                    active_advantages,
                    roles=(
                        token_credit_roles[active]
                        if token_credit_roles is not None
                        else None
                    ),
                    min_role_samples=config.min_role_samples,
                )
            token_unclipped = token_ratio * active_advantages
            token_clipped = token_ratio.clamp(
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            ) * active_advantages
            lifecycle_policy_loss = -torch.minimum(
                token_unclipped,
                token_clipped,
            ).mean()

    # Sampled conditional KL'i output head bazında raporla. Bu joint action
    # token sayısıyla şişmez ve economic-span early stop'un ana gözlemidir.
    head_inputs = (
        new_token_log_probs,
        old_token_log_probs,
        head_type_ids,
        decision_mask,
    )
    head_kls: dict[int, torch.Tensor] = {}
    if any(value is not None for value in head_inputs):
        if any(value is None for value in head_inputs):
            raise ValueError("Head KL tensorları birlikte verilmelidir.")
        assert new_token_log_probs is not None
        assert old_token_log_probs is not None
        assert head_type_ids is not None
        assert decision_mask is not None
        if not (
            new_token_log_probs.shape
            == old_token_log_probs.shape
            == head_type_ids.shape
            == decision_mask.shape
        ):
            raise ValueError("Head KL tensor shape'leri aynı olmalı.")
        token_log_ratio = new_token_log_probs - old_token_log_probs
        token_ratio = torch.exp(token_log_ratio.clamp(-20.0, 20.0))
        token_kl = (token_ratio - 1.0) - token_log_ratio
        for head in (
            ActionHeadType.UNIT_OPERATION,
            ActionHeadType.MARKET_OPERATION,
            ActionHeadType.ITEM,
            ActionHeadType.QUANTITY,
        ):
            member = decision_mask.bool() & (head_type_ids == int(head))
            if torch.any(member):
                head_kls[int(head)] = token_kl[member].mean().detach()
        unit_operation_kl = head_kls.get(
            int(ActionHeadType.UNIT_OPERATION), unit_operation_kl
        )
        market_operation_kl = head_kls.get(
            int(ActionHeadType.MARKET_OPERATION), market_operation_kl
        )
        item_kl = head_kls.get(int(ActionHeadType.ITEM), item_kl)
        quantity_kl = head_kls.get(int(ActionHeadType.QUANTITY), quantity_kl)

    if config.use_action_span_policy:
        local_span_inputs = (
            new_credit_span_log_probs,
            old_credit_span_log_probs,
            credit_span_values,
            credit_span_roles,
            credit_span_mask,
        )
        if any(value is None for value in local_span_inputs):
            raise ValueError("Economic span PPO tensorları birlikte verilmelidir.")
        assert new_credit_span_log_probs is not None
        assert old_credit_span_log_probs is not None
        assert credit_span_values is not None
        assert credit_span_roles is not None
        assert credit_span_mask is not None
        (
            lifecycle_policy_loss,
            local_span_kl,
            local_clip_fraction,
            local_group_kls,
        ) = _group_balanced_ppo_loss(
            new_credit_span_log_probs,
            old_credit_span_log_probs,
            credit_span_values,
            credit_span_roles,
            credit_span_mask,
            clip_ratio=config.clip_ratio,
            group_weights=config.span_credit_role_weights,
        )
        purchase_credit_kl = local_group_kls.get(
            int(SpanCreditRole.PURCHASE), purchase_credit_kl
        )
        farm_credit_kl = local_group_kls.get(
            int(SpanCreditRole.FARM), farm_credit_kl
        )
        trade_credit_kl = local_group_kls.get(
            int(SpanCreditRole.TRADE), trade_credit_kl
        )
        hire_credit_kl = local_group_kls.get(
            int(SpanCreditRole.HIRE), hire_credit_kl
        )
        monitored_kls: list[torch.Tensor] = []
        if config.span_policy_coef > 0.0:
            monitored_kls.extend(local_group_kls.values())
            monitored_kls.extend(head_kls.values())
            clip_fraction = local_clip_fraction
        if config.global_actor_coef > 0.0:
            monitored_kls.extend(global_group_kls.values())
        approximate_kl = (
            torch.stack(monitored_kls).max()
            if monitored_kls
            else local_span_kl
        )

    clipped_values = old_values + (new_values - old_values).clamp(
        -config.value_clip,
        config.value_clip,
    )
    value_error = (new_values - returns).square()
    clipped_value_error = (clipped_values - returns).square()
    value_loss = 0.5 * torch.maximum(
        value_error,
        clipped_value_error,
    ).mean()
    entropy = mean_entropies.mean()
    if config.use_action_span_policy:
        total = (
            config.global_actor_coef * policy_loss
            + config.span_policy_coef * lifecycle_policy_loss
            + config.value_coef * value_loss
            - config.entropy_coef * entropy
        )
    else:
        total = (
            policy_loss
            + config.lifecycle_policy_coef * lifecycle_policy_loss
            + config.value_coef * value_loss
            - config.entropy_coef * entropy
        )

    with torch.no_grad():
        return_variance = torch.var(returns, unbiased=False)
        explained_variance = torch.where(
            return_variance > 1e-8,
            1.0 - torch.var(returns - new_values, unbiased=False) / return_variance,
            torch.zeros_like(return_variance),
        )

    return PPOLoss(
        total=total,
        policy=policy_loss,
        lifecycle_policy=lifecycle_policy_loss,
        value=value_loss,
        entropy=entropy,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        explained_variance=explained_variance,
        unit_span_kl=unit_span_kl,
        market_span_kl=market_span_kl,
        purchase_credit_kl=purchase_credit_kl,
        farm_credit_kl=farm_credit_kl,
        trade_credit_kl=trade_credit_kl,
        hire_credit_kl=hire_credit_kl,
        unit_operation_kl=unit_operation_kl,
        market_operation_kl=market_operation_kl,
        item_kl=item_kl,
        quantity_kl=quantity_kl,
    )


def actor_critic_ppo_loss(
    model: KaggricultureActorCritic,
    batch: PPOTrainingBatch,
    config: PPOConfig | None = None,
) -> PPOLoss:
    config = config or PPOConfig()
    output = model(
        batch.own_categorical_ids,
        batch.own_numeric_features,
        batch.own_worker_positions,
        batch.opponent_categorical_ids,
        batch.opponent_numeric_features,
        batch.opponent_worker_positions,
        batch.global_features,
        batch.unit_positions_yx,
        batch.unit_features,
        batch.unit_mask,
        batch.input_token_ids,
        batch.context_unit_indices,
        batch.head_type_ids,
    )
    statistics = masked_sequence_statistics(
        output.token_logits,
        batch.target_token_ids,
        batch.valid_token_masks,
        batch.sequence_mask,
        batch.decision_mask,
        head_type_ids=batch.head_type_ids,
        entropy_head_weights=config.entropy_head_weights,
        head_balanced_entropy=config.head_balanced_entropy,
    )
    new_span_log_probs = span_log_probabilities(
        statistics.token_log_probs,
        batch.decision_mask,
        batch.span_starts,
        batch.span_ends,
        batch.span_mask,
    )
    new_credit_span_log_probs = span_log_probabilities(
        statistics.token_log_probs,
        batch.decision_mask,
        batch.credit_span_starts,
        batch.credit_span_ends,
        batch.credit_span_mask,
    )
    return ppo_loss(
        new_action_log_probs=statistics.action_log_probs,
        old_action_log_probs=batch.old_action_log_probs,
        new_values=output.value,
        old_values=batch.old_values,
        advantages=batch.advantages,
        returns=batch.returns,
        mean_entropies=statistics.weighted_mean_entropies,
        config=config,
        new_token_log_probs=statistics.token_log_probs,
        old_token_log_probs=batch.old_token_log_probs,
        token_advantages=batch.token_advantages,
        token_credit_roles=batch.token_credit_roles,
        token_credit_mask=batch.token_credit_mask,
        new_span_log_probs=new_span_log_probs,
        old_span_log_probs=batch.span_old_log_probs,
        span_sections=batch.span_sections,
        span_mask=batch.span_mask,
        new_credit_span_log_probs=new_credit_span_log_probs,
        old_credit_span_log_probs=batch.credit_span_old_log_probs,
        credit_span_values=batch.credit_span_values,
        credit_span_roles=batch.credit_span_roles,
        credit_span_mask=batch.credit_span_mask,
        head_type_ids=batch.head_type_ids,
        decision_mask=batch.decision_mask,
    )


def optimizer_step(
    model: KaggricultureActorCritic,
    optimizer: torch.optim.Optimizer,
    loss: PPOLoss,
    config: PPOConfig | None = None,
) -> float:
    """Tek PPO minibatch gradient adımı; gradient norm'unu döndürür."""
    config = config or PPOConfig()
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.max_grad_norm,
    )
    optimizer.step()
    return float(gradient_norm)
