"""Tek tarafın eğitildiği, gecikmeli ve donmuş self-play rakibi.

Learner her PPO update'inde normal şekilde güncellenir. Rakip ise learner'ın
en az ``lag_updates`` update eski bir snapshot'ını kullanır; rakibin
parametreleri optimizer'a girmez ve gradient toplamaz.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from action_encoding import ActionGrammarConfig
from model import KaggricultureActorCritic
from policy import sample_action


SELF_PLAY_OPPONENT_NAME = "selfplay"
SELF_PLAY_STATE_VERSION = 1
SELF_PLAY_SNAPSHOT_VERSION = 1


@dataclass(frozen=True, slots=True)
class LaggedSelfPlayConfig:
    """Rakip snapshot'ının learner'dan ne kadar geride kalacağı."""

    lag_updates: int = 100
    snapshot_interval: int = 10
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.lag_updates <= 0:
            raise ValueError("self-play lag_updates pozitif olmalı.")
        if self.snapshot_interval <= 0:
            raise ValueError("self-play snapshot_interval pozitif olmalı.")


@dataclass(frozen=True, slots=True)
class SelfPlayRefresh:
    learner_update: int
    opponent_update: int
    actual_lag_updates: int
    snapshot_saved: bool
    opponent_changed: bool


class LaggedSelfPlayOpponent:
    """Learner'ın eski checkpoint'inden aksiyon üreten salt-okunur rakip."""

    def __init__(
        self,
        learner: KaggricultureActorCritic,
        *,
        run_directory: Path,
        device: torch.device,
        config: LaggedSelfPlayConfig,
        seed: int,
        learner_update: int = 0,
        resume_state: Mapping[str, Any] | None = None,
    ) -> None:
        if learner_update < 0:
            raise ValueError("learner_update negatif olamaz.")

        self.config = config
        self.device = device
        self.snapshot_directory = run_directory / "selfplay_snapshots"
        self.snapshot_directory.mkdir(parents=True, exist_ok=True)
        self.model = KaggricultureActorCritic().to(device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.generator = torch.Generator(device=device.type)
        self.generator.manual_seed(int(seed))
        self.learner_update = int(learner_update)
        self.opponent_update = 0

        if resume_state is None:
            if learner_update != 0:
                raise ValueError(
                    "Sıfırdan self-play rakibi yalnızca learner update=0 iken "
                    "oluşturulabilir; resume_state eksik."
                )
            initial_path = self._snapshot_path(0)
            if initial_path.exists():
                raise FileExistsError(
                    f"Yeni self-play koşusunda snapshot zaten var: {initial_path}"
                )
            self._save_snapshot(learner, update=0)
            self._load_snapshot(0)
        else:
            self.load_state_dict(resume_state, learner_update=learner_update)

    def _snapshot_path(self, update: int) -> Path:
        return self.snapshot_directory / f"update_{update:06d}.pt"

    def _save_snapshot(
        self,
        learner: KaggricultureActorCritic,
        *,
        update: int,
    ) -> Path:
        path = self._snapshot_path(update)
        temporary = path.with_name(f".{path.name}.tmp")
        state = {
            name: value.detach().cpu().clone()
            for name, value in learner.state_dict().items()
        }
        torch.save(
            {
                "snapshot_version": SELF_PLAY_SNAPSHOT_VERSION,
                "learner_update": int(update),
                "model_state_dict": state,
            },
            temporary,
        )
        os.replace(temporary, path)
        return path

    def _load_snapshot(self, update: int) -> None:
        path = self._snapshot_path(update)
        if not path.exists():
            raise FileNotFoundError(f"Self-play snapshot bulunamadı: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        version = int(payload.get("snapshot_version", -1))
        if version != SELF_PLAY_SNAPSHOT_VERSION:
            raise ValueError(
                "Desteklenmeyen self-play snapshot sürümü: "
                f"{version}; beklenen={SELF_PLAY_SNAPSHOT_VERSION}."
            )
        saved_update = int(payload.get("learner_update", -1))
        if saved_update != update:
            raise ValueError(
                f"Self-play snapshot update uyuşmazlığı: dosya={update}, "
                f"payload={saved_update}."
            )
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self.opponent_update = update

    def _target_opponent_update(self, learner_update: int) -> int:
        target = max(0, learner_update - self.config.lag_updates)
        return (target // self.config.snapshot_interval) * self.config.snapshot_interval

    def after_learner_update(
        self,
        learner: KaggricultureActorCritic,
        learner_update: int,
    ) -> SelfPlayRefresh:
        """Yeni learner snapshot'ını saklar ve gecikmeli rakibi ilerletir."""
        if learner_update <= self.learner_update:
            raise ValueError(
                "Self-play learner update sıkı artmalı: "
                f"önceki={self.learner_update}, gelen={learner_update}."
            )
        self.learner_update = int(learner_update)
        snapshot_saved = learner_update % self.config.snapshot_interval == 0
        if snapshot_saved:
            self._save_snapshot(learner, update=learner_update)

        target_update = self._target_opponent_update(learner_update)
        opponent_changed = target_update != self.opponent_update
        if opponent_changed:
            self._load_snapshot(target_update)

        return SelfPlayRefresh(
            learner_update=learner_update,
            opponent_update=self.opponent_update,
            actual_lag_updates=learner_update - self.opponent_update,
            snapshot_saved=snapshot_saved,
            opponent_changed=opponent_changed,
        )

    def act(
        self,
        observation: Mapping[str, Any],
        *,
        grammar_config: ActionGrammarConfig,
    ) -> dict[str, list[Any]]:
        """Yalnızca donmuş modelden aksiyon üretir; gradient oluşmaz."""
        return sample_action(
            self.model,
            observation,
            grammar_config=grammar_config,
            deterministic=self.config.deterministic,
            generator=self.generator,
        ).action

    def summary(self) -> dict[str, Any]:
        return {
            "self_play_learner_update": self.learner_update,
            "self_play_opponent_update": self.opponent_update,
            "self_play_actual_lag_updates": (
                self.learner_update - self.opponent_update
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": SELF_PLAY_STATE_VERSION,
            "config": asdict(self.config),
            "learner_update": self.learner_update,
            "opponent_update": self.opponent_update,
            "generator_state": self.generator.get_state().cpu(),
            "generator_device": str(self.generator.device),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        learner_update: int,
    ) -> None:
        version = int(state.get("version", -1))
        if version != SELF_PLAY_STATE_VERSION:
            raise ValueError(
                "Desteklenmeyen self-play state sürümü: "
                f"{version}; beklenen={SELF_PLAY_STATE_VERSION}."
            )
        saved_config = dict(state.get("config", {}))
        if saved_config != asdict(self.config):
            raise ValueError(
                "Self-play resume ayarları uyuşmuyor: "
                f"checkpoint={saved_config}, CLI={asdict(self.config)}."
            )
        saved_learner_update = int(state.get("learner_update", -1))
        if saved_learner_update != learner_update:
            raise ValueError(
                "Self-play learner update checkpoint ile uyuşmuyor: "
                f"ana={learner_update}, self_play={saved_learner_update}."
            )
        opponent_update = int(state.get("opponent_update", -1))
        expected_update = self._target_opponent_update(learner_update)
        if opponent_update != expected_update:
            raise ValueError(
                "Self-play rakip update'i gecikme kuralıyla uyuşmuyor: "
                f"checkpoint={opponent_update}, beklenen={expected_update}."
            )
        self.learner_update = learner_update
        self._load_snapshot(opponent_update)
        self.generator.set_state(state["generator_state"].cpu())
