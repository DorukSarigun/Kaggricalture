"""Actor-critic modelinden geçerli otoregresif Kaggriculture aksiyonu üretme."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from action_encoding import (
    ACTION_VOCAB_SIZE,
    TOKEN_TO_ID,
    ActionGrammar,
    ActionGrammarConfig,
    ActionToken,
)
from model import KaggricultureActorCritic
from observation_encoding import EncodedObservation, encode_observation


@dataclass(frozen=True, slots=True)
class SampledAction:
    action: dict[str, list[Any]]
    encoded_observation: EncodedObservation
    token_ids: np.ndarray
    input_token_ids: np.ndarray
    target_token_ids: np.ndarray
    valid_token_masks: np.ndarray
    context_unit_indices: np.ndarray
    head_type_ids: np.ndarray
    decision_mask: np.ndarray
    token_log_probs: np.ndarray
    token_entropies: np.ndarray
    action_log_prob: float
    mean_entropy: float
    value: float


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _batch_tensor(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device).unsqueeze(0)


@torch.no_grad()
def sample_action(
    model: KaggricultureActorCritic,
    observation: Mapping[str, Any],
    grammar_config: ActionGrammarConfig | None = None,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
    max_tokens: int = 512,
) -> SampledAction:
    """Bir observation için EOS'a kadar masked token üretir."""
    if max_tokens < 2:
        raise ValueError("max_tokens en az 2 olmalı.")

    device = _model_device(model)
    encoded = encode_observation(observation)
    grammar = ActionGrammar(observation, grammar_config)

    was_training = model.training
    model.eval()
    try:
        state, unit_contexts, value = model.encode_state_and_units(
            _batch_tensor(encoded.own_categorical_ids, device, torch.long),
            _batch_tensor(encoded.own_numeric_features, device, torch.float32),
            _batch_tensor(encoded.own_worker_positions, device, torch.float32),
            _batch_tensor(encoded.opponent_categorical_ids, device, torch.long),
            _batch_tensor(encoded.opponent_numeric_features, device, torch.float32),
            _batch_tensor(encoded.opponent_worker_positions, device, torch.float32),
            _batch_tensor(encoded.global_features, device, torch.float32),
            _batch_tensor(encoded.own_unit_positions_yx, device, torch.long),
            _batch_tensor(encoded.own_unit_features, device, torch.float32),
            torch.ones(
                (1, len(encoded.own_unit_positions_yx)),
                dtype=torch.bool,
                device=device,
            ),
        )

        # BOS gramer tarafından zorunludur; policy'nin ilk tahmini FARMER'dır.
        grammar.push(ActionToken.BOS)
        token_ids = [TOKEN_TO_ID[ActionToken.BOS]]
        input_ids: list[int] = []
        target_ids: list[int] = []
        masks: list[np.ndarray] = []
        contexts: list[int] = []
        head_types: list[int] = []
        decisions: list[bool] = []
        token_log_probs: list[float] = []
        token_entropies: list[float] = []
        hidden: torch.Tensor | None = None

        while not grammar.is_finished:
            if len(token_ids) >= max_tokens:
                raise RuntimeError(
                    f"Action token dizisi {max_tokens} tokenı aştı; "
                    "gramer tamamlanamadı."
                )

            input_id = token_ids[-1]
            valid_mask_np = grammar.valid_token_mask()
            context_index = grammar.context_unit_index()
            head_type = grammar.next_head_type()

            logits, hidden = model.action_decoder(
                state.state_vector,
                torch.tensor([[input_id]], dtype=torch.long, device=device),
                torch.tensor(
                    [[context_index]],
                    dtype=torch.long,
                    device=device,
                ),
                torch.tensor(
                    [[int(head_type)]],
                    dtype=torch.long,
                    device=device,
                ),
                unit_contexts,
                hidden,
            )
            logits = logits[0, 0]
            valid_mask = torch.as_tensor(
                valid_mask_np,
                dtype=torch.bool,
                device=device,
            )
            if not torch.any(valid_mask):
                raise RuntimeError("Gramer hiçbir geçerli token üretmedi.")
            masked_logits = logits.masked_fill(~valid_mask, -torch.inf)
            log_probabilities = torch.log_softmax(masked_logits, dim=-1)
            probabilities = torch.softmax(masked_logits, dim=-1)
            is_decision = int(valid_mask.sum().item()) > 1

            if not is_decision:
                next_id = int(torch.nonzero(valid_mask, as_tuple=False)[0, 0])
                chosen_log_prob = 0.0
                entropy = 0.0
            elif deterministic:
                next_id = int(torch.argmax(masked_logits).item())
                chosen_log_prob = float(log_probabilities[next_id].item())
                entropy = float(
                    -(probabilities[valid_mask] * log_probabilities[valid_mask]).sum().item()
                )
            else:
                next_id = int(
                    torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=generator,
                    ).item()
                )
                chosen_log_prob = float(log_probabilities[next_id].item())
                entropy = float(
                    -(probabilities[valid_mask] * log_probabilities[valid_mask]).sum().item()
                )

            input_ids.append(input_id)
            target_ids.append(next_id)
            masks.append(valid_mask_np)
            contexts.append(context_index)
            head_types.append(int(head_type))
            decisions.append(is_decision)
            token_log_probs.append(chosen_log_prob)
            token_entropies.append(entropy)
            token_ids.append(next_id)
            grammar.push(next_id)

        decision_count = sum(decisions)
        mean_entropy = (
            sum(entropy for entropy, keep in zip(token_entropies, decisions) if keep)
            / decision_count
            if decision_count
            else 0.0
        )
        return SampledAction(
            action=grammar.to_action(),
            encoded_observation=encoded,
            token_ids=np.asarray(token_ids, dtype=np.int64),
            input_token_ids=np.asarray(input_ids, dtype=np.int64),
            target_token_ids=np.asarray(target_ids, dtype=np.int64),
            valid_token_masks=np.stack(masks).astype(np.bool_),
            context_unit_indices=np.asarray(contexts, dtype=np.int64),
            head_type_ids=np.asarray(head_types, dtype=np.int64),
            decision_mask=np.asarray(decisions, dtype=np.bool_),
            token_log_probs=np.asarray(token_log_probs, dtype=np.float32),
            token_entropies=np.asarray(token_entropies, dtype=np.float32),
            action_log_prob=float(sum(token_log_probs)),
            mean_entropy=float(mean_entropy),
            value=float(value.item()),
        )
    finally:
        model.train(was_training)
