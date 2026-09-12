"""Müfredat aşamasına duyarlı PPO entropy katsayısı.

Dar aksiyon maskelerinde politika erken deterministik hale gelmesin diye ilk
aşamalarda daha güçlü keşif kullanılır. Tam aksiyon uzayı açıldıktan sonra
katsayı yavaşça azaltılır; alt sınır sıfır olmadığı için keşif tamamen
kapanmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


ENTROPY_SCHEDULE_VERSION = 1


@dataclass(frozen=True, slots=True)
class EntropyScheduleConfig:
    enabled: bool = True
    wheat_loop: float = 0.030
    wheat_hire: float = 0.025
    wheat_land: float = 0.020
    wheat_carrot: float = 0.015
    full_start: float = 0.012
    full_end: float = 0.005
    full_decay_episodes: int = 3000
    version: int = ENTROPY_SCHEDULE_VERSION

    def __post_init__(self) -> None:
        coefficients = (
            self.wheat_loop,
            self.wheat_hire,
            self.wheat_land,
            self.wheat_carrot,
            self.full_start,
            self.full_end,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in coefficients):
            raise ValueError("Entropy katsayıları sonlu ve negatif olmayan sayılar olmalı.")
        if self.full_decay_episodes <= 0:
            raise ValueError("full_decay_episodes pozitif olmalı.")
        if self.version != ENTROPY_SCHEDULE_VERSION:
            raise ValueError(
                f"Desteklenmeyen entropy schedule version={self.version}; "
                f"beklenen={ENTROPY_SCHEDULE_VERSION}."
            )

    def coefficient(
        self,
        stage_name: str,
        episode_in_stage: int,
        *,
        fixed_coefficient: float,
    ) -> float:
        """Bir episode için uygulanacak entropy bonus katsayısını döndürür."""
        if episode_in_stage <= 0:
            raise ValueError("episode_in_stage pozitif olmalı.")
        if not math.isfinite(fixed_coefficient) or fixed_coefficient < 0.0:
            raise ValueError("fixed_coefficient sonlu ve negatif olmayan olmalı.")
        if not self.enabled:
            return float(fixed_coefficient)

        stage_coefficients = {
            "wheat_loop": self.wheat_loop,
            "wheat_hire": self.wheat_hire,
            "wheat_land": self.wheat_land,
            "wheat_carrot": self.wheat_carrot,
        }
        if stage_name in stage_coefficients:
            return float(stage_coefficients[stage_name])
        if stage_name not in ("full", "off"):
            raise ValueError(f"Bilinmeyen curriculum stage: {stage_name!r}")

        if self.full_decay_episodes == 1:
            progress = 1.0
        else:
            progress = min(
                max((episode_in_stage - 1) / (self.full_decay_episodes - 1), 0.0),
                1.0,
            )
        return float(self.full_start + progress * (self.full_end - self.full_start))


def episode_in_stage(
    episode_number: int,
    stage_index: int,
    episodes_per_stage: int,
) -> int:
    """Müfredat aşaması içindeki 1-tabanlı episode numarası."""
    if episode_number <= 0 or episodes_per_stage <= 0:
        raise ValueError("Episode numaraları pozitif olmalı.")
    if stage_index < 0:
        return episode_number
    stage_start = stage_index * episodes_per_stage + 1
    return max(episode_number - stage_start + 1, 1)
