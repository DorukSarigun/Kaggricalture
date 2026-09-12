"""Aşamalı aksiyon maskesi ile oyun mekaniği müfredatı.

Yasal gramer her zaman geçerlidir. Müfredat onun üstüne, o aşamada gereksiz
yasal aksiyonları kapatır. PASS ve STOP kapatılamaz.

Probe episode'larda maske kalkar (tohum hediyesi aynı kalır) ki kapatılmış
head'ler tamamen soğumasın.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from action_encoding import ActionGrammarConfig


_MOVE_AND_PASS = frozenset({"PASS", "NORTH", "SOUTH", "EAST", "WEST"})
_WHEAT_FIELD_OPS = _MOVE_AND_PASS | frozenset(
    {"PLANT", "WATER", "HARVEST", "DROP"}
)
_WHEAT_MARKET_OPS = frozenset({"BUY_SEED", "SELL"})
_WHEAT_ITEMS = frozenset({"WHEAT"})


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    name: str
    allowed_unit_ops: frozenset[str] | None
    allowed_market_ops: frozenset[str] | None
    allowed_items: frozenset[str] | None
    starting_seeds: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    enabled: bool = False
    probe_probability: float = 0.1
    episodes_per_stage: int = 100
    starting_wheat_seeds: int = 8

    def __post_init__(self) -> None:
        if not 0.0 <= self.probe_probability <= 1.0:
            raise ValueError("probe_probability [0, 1] aralığında olmalı.")
        if self.episodes_per_stage <= 0:
            raise ValueError("episodes_per_stage pozitif olmalı.")
        if self.starting_wheat_seeds < 0:
            raise ValueError("starting_wheat_seeds negatif olamaz.")


@dataclass(frozen=True, slots=True)
class EpisodeCurriculum:
    stage_name: str
    stage_index: int
    is_probe: bool
    grammar_config: ActionGrammarConfig
    starting_seeds: dict[str, int]


WHEAT_CURRICULUM_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        name="wheat_loop",
        allowed_unit_ops=_WHEAT_FIELD_OPS,
        allowed_market_ops=_WHEAT_MARKET_OPS,
        allowed_items=_WHEAT_ITEMS,
    ),
    CurriculumStage(
        name="wheat_hire",
        allowed_unit_ops=_WHEAT_FIELD_OPS,
        allowed_market_ops=_WHEAT_MARKET_OPS | frozenset({"HIRE"}),
        allowed_items=_WHEAT_ITEMS,
    ),
    CurriculumStage(
        name="wheat_land",
        allowed_unit_ops=_WHEAT_FIELD_OPS,
        allowed_market_ops=_WHEAT_MARKET_OPS | frozenset({"HIRE", "BUY_LAND"}),
        allowed_items=_WHEAT_ITEMS,
    ),
    CurriculumStage(
        name="wheat_carrot",
        allowed_unit_ops=_WHEAT_FIELD_OPS,
        allowed_market_ops=_WHEAT_MARKET_OPS | frozenset({"HIRE", "BUY_LAND"}),
        allowed_items=frozenset({"WHEAT", "CARROT"}),
    ),
    CurriculumStage(
        name="full",
        allowed_unit_ops=None,
        allowed_market_ops=None,
        allowed_items=None,
    ),
)


UNRESTRICTED_CURRICULUM = EpisodeCurriculum(
    stage_name="off",
    stage_index=-1,
    is_probe=False,
    grammar_config=ActionGrammarConfig(),
    starting_seeds={},
)


def _stage_starting_seeds(
    stage: CurriculumStage,
    wheat_count: int,
) -> dict[str, int]:
    if stage.allowed_unit_ops is None and stage.allowed_market_ops is None:
        return {}
    seeds = dict(stage.starting_seeds)
    if wheat_count > 0 and (
        stage.allowed_items is None or "WHEAT" in stage.allowed_items
    ):
        seeds["WHEAT"] = wheat_count
    return seeds


def grammar_config_for_stage(stage: CurriculumStage) -> ActionGrammarConfig:
    return ActionGrammarConfig(
        allowed_unit_ops=stage.allowed_unit_ops,
        allowed_market_ops=stage.allowed_market_ops,
        allowed_items=stage.allowed_items,
    )


def select_stage(
    config: CurriculumConfig,
    episode_number: int,
) -> CurriculumStage:
    if episode_number <= 0:
        raise ValueError("episode_number pozitif olmalı.")
    if not config.enabled:
        return WHEAT_CURRICULUM_STAGES[-1]
    last_index = len(WHEAT_CURRICULUM_STAGES) - 1
    index = min((episode_number - 1) // config.episodes_per_stage, last_index)
    return WHEAT_CURRICULUM_STAGES[index]


def resolve_episode_curriculum(
    config: CurriculumConfig,
    episode_number: int,
    episode_seed: int,
) -> EpisodeCurriculum:
    if not config.enabled:
        return UNRESTRICTED_CURRICULUM

    stage = select_stage(config, episode_number)
    stage_index = WHEAT_CURRICULUM_STAGES.index(stage)
    rng = np.random.RandomState(int(episode_seed) + 7919)
    is_probe = bool(rng.random() < config.probe_probability)
    starting_seeds = _stage_starting_seeds(stage, config.starting_wheat_seeds)
    if is_probe:
        return EpisodeCurriculum(
            stage_name=stage.name,
            stage_index=stage_index,
            is_probe=True,
            grammar_config=ActionGrammarConfig(),
            starting_seeds=starting_seeds,
        )
    return EpisodeCurriculum(
        stage_name=stage.name,
        stage_index=stage_index,
        is_probe=False,
        grammar_config=grammar_config_for_stage(stage),
        starting_seeds=starting_seeds,
    )


def inject_starting_seeds(
    observation: Mapping[str, Any],
    seeds: Mapping[str, int],
) -> None:
    """Learner private seed stoğuna (in-place) tohum ekler."""
    if not seeds:
        return
    private = observation["private"]
    seed_store = private["seeds"]
    for crop, quantity in seeds.items():
        if crop not in game.CROPS:
            raise ValueError(f"Bilinmeyen tohum: {crop}")
        if int(quantity) < 0:
            raise ValueError("Tohum miktarı negatif olamaz.")
        seed_store[crop] = int(seed_store.get(crop, 0)) + int(quantity)
