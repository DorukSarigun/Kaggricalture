"""Kaggriculture global reward ve ekonomik potential shaping hesabı.

Global hedef episode sonundaki rakibe karşı gold farkıdır. Bu tek başına 719
adımda tek bir skaler demektir; o yüzden üzerine potential-based shaping
eklenir:

    F(s, s') = beta * (gamma * Phi(s') - Phi(s))

Terminal state'te Phi=0 kullanılır, dolayısıyla shaping bir episode boyunca
teleskoplanarak sıfıra yakınsar: optimal politikayı değiştirmez, yalnızca
krediyi adımlara dağıtır. Ekim, sulama ve ihmal gibi kararlar böylece
gerçekleştikleri adımda sinyal alır.

Gerçekleşmiş üretim kârının aksiyonlara dağıtımı bu dosyada değil
``lifecycle_credit.py`` içindedir.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as game


OBJECT_TYPES_PATH = Path(__file__).parent / "data" / "object_types.json"
with OBJECT_TYPES_PATH.open(encoding="utf-8") as _file:
    OBJECT_TYPE_ITEMS = json.load(_file)["items"]


@dataclass(frozen=True, slots=True)
class RewardConfig:
    gamma: float = 0.999
    # Bir buğday ekimi Phi'yi ~0.03 oynatır; beta=0.25 bunu adım avantajının
    # standart sapmasıyla aynı mertebeye getirir. 0.0 shaping'i kapatır.
    shaping_beta: float = 0.25
    reward_scale: float = 3000.0
    starting_money: float = 3000.0
    season_days: int = 30
    turns_per_day: int = 24

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma (0, 1] aralığında olmalı.")
        if self.shaping_beta < 0.0:
            raise ValueError("shaping_beta negatif olamaz.")
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale pozitif olmalı.")
        if self.season_days <= 0 or self.turns_per_day <= 0:
            raise ValueError("season_days ve turns_per_day pozitif olmalı.")


@dataclass(frozen=True, slots=True)
class PotentialBreakdown:
    cash: float
    liquid_products: float
    seeds: float
    unplaced_animals: float
    growing_crops: float
    placed_animals: float

    @property
    def total(self) -> float:
        return (
            self.cash
            + self.liquid_products
            + self.seeds
            + self.unplaced_animals
            + self.growing_crops
            + self.placed_animals
        )


@dataclass(frozen=True, slots=True)
class TransitionReward:
    total: float
    base: float
    shaping: float
    current_potential: float
    next_potential: float
    current_breakdown: PotentialBreakdown
    next_breakdown: PotentialBreakdown | None
    lifecycle_credit: float = 0.0


def _player_farm(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    return observation["farms"][int(observation["player"])]


def _owned_item_counts(observation: Mapping[str, Any]) -> dict[str, int]:
    """Shed ve tüm unit inventory'lerindeki item toplamları."""
    counts = {
        item: int(observation["private"]["shed"].get(item, 0))
        for item in (*game.PRODUCTS, *game.ANIMALS)
    }
    for inventory in observation["private"]["inventories"]:
        for item, quantity in inventory.items():
            if item in counts:
                counts[item] += int(quantity)
    return counts


def liquidation_revenue(
    item: str,
    quantity: int,
    market: Mapping[str, Any],
) -> float:
    """Bir ürünü motorun SELL eğrisiyle birer birer satmanın geliri.

    Rakibin gelecekteki emirleri bilinmediği için bu, mevcut market state'i
    üzerinden kendi tasfiye tahminimizdir. Fiyat $1 ise motor market inventory'sini
    artırmaz; hesap aynı kuralı izler.
    """
    if item not in game.PRODUCTS:
        raise ValueError(f"{item!r} markette satılabilir bir product değil.")
    if quantity < 0:
        raise ValueError("quantity negatif olamaz.")

    market_inventory = int(market["inventory"][item])
    params = market.get("params")
    revenue = 0.0
    for _ in range(int(quantity)):
        price = game.market_price(item, market_inventory, params)
        revenue += price
        if price > 1:
            market_inventory += 1
    return revenue


def _current_marginal_price(item: str, market: Mapping[str, Any]) -> float:
    return float(
        game.market_price(
            item,
            int(market["inventory"][item]),
            market.get("params"),
        )
    )


def _day_discount(days_until_value: int, config: RewardConfig) -> float:
    turns = max(int(days_until_value), 0) * config.turns_per_day
    return config.gamma**turns


def _last_actionable_day(config: RewardConfig) -> int:
    return config.season_days - 1


def _seed_option_value(
    crop: str,
    quantity: int,
    current_day: int,
    config: RewardConfig,
) -> float:
    """Ekilebilecek tohumun muhafazakâr replacement value'su.

    İlk hasada sezon bitmeden ulaşılamıyorsa tohumun terminal para hedefi
    açısından ekonomik değeri sıfırdır.
    """
    first_yield = int(game.CROPS[crop]["first_yield_day"])
    if current_day + first_yield > _last_actionable_day(config):
        return 0.0
    return float(quantity * game.CROPS[crop]["seed"])


def _unplaced_animal_value(
    animal: str,
    quantity: int,
    current_day: int,
    config: RewardConfig,
) -> float:
    first_yield = int(game.ANIMALS[animal]["first_yield_day"])
    if current_day + first_yield > _last_actionable_day(config):
        return 0.0
    return float(quantity * game.ANIMALS[animal]["cost"])


def _one_time_growth_targets(crop: str) -> tuple[int, int]:
    """Gübresiz yield tavanı ve o tavana ulaşma yaşı.

    Değerler ``data/object_types.json`` üzerinden okunur. Sabit tablo yerine
    veri kullanmak, motorun ``ongoing`` bayrağı değişirse (ör. TOMATO tek
    seferliğe dönerse) sessiz bir KeyError yerine doğru sonuç verir.
    """
    item = OBJECT_TYPE_ITEMS.get(crop)
    if item is None:
        raise KeyError(f"{crop!r} object_types.json içinde tanımlı değil.")
    unfertilized_cap = item["max_yield_unfertilized"]
    target_age = item["time_to_max_yield_days"]
    if unfertilized_cap is None or target_age is None:
        raise ValueError(
            f"{crop!r} tek seferlik hasat için gerekli büyüme verisine sahip değil."
        )
    return int(unfertilized_cap), int(target_age)


def _one_time_crop_value(
    tile: Mapping[str, Any],
    current_day: int,
    market: Mapping[str, Any],
    config: RewardConfig,
) -> float:
    crop = tile["crop"]
    crop_data = game.CROPS[crop]
    age = current_day - int(tile["planted_day"])
    current_units = max(int(tile.get("yield_units", 0)), 0)
    first_yield_age = int(crop_data["first_yield_day"])

    if age < first_yield_age:
        days_to_harvest = first_yield_age - age
    else:
        days_to_harvest = 0
    harvest_day = current_day + days_to_harvest
    if harvest_day > _last_actionable_day(config):
        return 0.0

    # Sulaması aksatılmayan bitkinin mevcut sezonda erişebileceği muhafazakâr
    # (gübresiz) hedef. Mevcut yield gübreyle bunun üzerindeyse korunur.
    unfertilized_cap, target_age = _one_time_growth_targets(crop)
    remaining_growth_days = max(target_age - age, 0)
    reachable_units = min(
        unfertilized_cap,
        current_units + remaining_growth_days,
    )
    expected_units = max(current_units, reachable_units)

    preferred_harvest_day = min(
        int(tile["planted_day"]) + target_age,
        _last_actionable_day(config),
    )
    if preferred_harvest_day < current_day:
        preferred_harvest_day = current_day
    discount = _day_discount(preferred_harvest_day - current_day, config)
    return expected_units * _current_marginal_price(crop, market) * discount


def _ongoing_crop_value(
    tile: Mapping[str, Any],
    current_day: int,
    market: Mapping[str, Any],
    config: RewardConfig,
) -> float:
    crop = tile["crop"]
    crop_data = game.CROPS[crop]
    planted_day = int(tile["planted_day"])
    price = _current_marginal_price(crop, market)

    # Tile üzerinde birikmiş ürün bugün hasat edilebilir.
    value = max(int(tile.get("yield_units", 0)), 0) * price

    first_day = planted_day + int(crop_data["first_yield_day"])
    interval = int(crop_data["interval"])
    production_count = int(crop_data["max_yield"])
    fertilized_until = int(tile.get("fertilized_until_day", -1))

    for production_index in range(production_count):
        production_day = first_day + production_index * interval
        if production_day <= current_day:
            continue
        if production_day > _last_actionable_day(config):
            continue
        # End-of-day refresh production_day state'ini yaratır; gübre kontrolü
        # bir önceki gün üzerinden yapılır.
        units = 2 if fertilized_until >= production_day - 1 else 1
        value += (
            units
            * price
            * _day_discount(production_day - current_day, config)
        )
    return value


def _crop_tile_value(
    tile: Mapping[str, Any],
    current_day: int,
    market: Mapping[str, Any],
    config: RewardConfig,
) -> float:
    if game.CROPS[tile["crop"]]["ongoing"]:
        return _ongoing_crop_value(tile, current_day, market, config)
    return _one_time_crop_value(tile, current_day, market, config)


def _animal_tile_value(
    tile: Mapping[str, Any],
    current_day: int,
    market: Mapping[str, Any],
    config: RewardConfig,
) -> float:
    animal = tile["animal"]
    animal_data = game.ANIMALS[animal]
    product = animal_data["product"]
    product_price = _current_marginal_price(product, market)
    wheat_price = _current_marginal_price("WHEAT", market)

    value = max(int(tile.get("yield_units", 0)), 0) * product_price
    if tile.get("fertilizer_available", False):
        value += _current_marginal_price("FERTILIZER", market)

    placed_day = int(tile["placed_day"])
    first_day = placed_day + int(animal_data["first_yield_day"])
    interval = int(animal_data["interval"])
    pending_bonus = max(int(tile.get("pending_care_bonus", 0)), 0)

    production_days: list[int] = []
    production_day = first_day
    while production_day <= _last_actionable_day(config):
        if production_day > current_day:
            production_days.append(production_day)
        production_day += interval

    if not production_days:
        return value

    # Hayvanı hayatta tutmak için her gün bir wheat gerekir. Yalnızca son
    # değerlendirdiğimiz üretime kadar olan yem opportunity cost'unu düşüyoruz.
    last_production_day = production_days[-1]
    feeding_days = max(last_production_day - current_day, 0)
    feed_cost = feeding_days * wheat_price

    future_output = 0.0
    for index, day in enumerate(production_days):
        units = 1
        if index == 0:
            units += pending_bonus
        units = min(units, int(animal_data["max_held"]))
        future_output += (
            units
            * product_price
            * _day_discount(day - current_day, config)
        )

    # Negatif gelecek opsiyonunu sıfırda kesmek, hayvanın tile'daki mevcut
    # hasadının değerini yanlışlıkla yok etmez.
    return value + max(future_output - feed_cost, 0.0)


def economic_potential_breakdown(
    observation: Mapping[str, Any],
    config: RewardConfig | None = None,
) -> PotentialBreakdown:
    config = config or RewardConfig()
    farm = _player_farm(observation)
    market = observation["market"]
    private = observation["private"]
    current_day = int(observation["day"])
    owned = _owned_item_counts(observation)

    liquid_products = sum(
        liquidation_revenue(item, owned[item], market)
        for item in game.PRODUCTS
    )
    seeds = sum(
        _seed_option_value(
            crop,
            int(private["seeds"].get(crop, 0)),
            current_day,
            config,
        )
        for crop in game.CROPS
    )
    unplaced_animals = sum(
        _unplaced_animal_value(
            animal,
            owned[animal],
            current_day,
            config,
        )
        for animal in game.ANIMALS
    )

    growing_crops = 0.0
    placed_animals = 0.0
    for row in farm["tiles"]:
        for tile in row:
            if not isinstance(tile, Mapping):
                continue
            if tile.get("kind") == "PLANT":
                growing_crops += _crop_tile_value(
                    tile,
                    current_day,
                    market,
                    config,
                )
            elif "animal" in tile:
                placed_animals += _animal_tile_value(
                    tile,
                    current_day,
                    market,
                    config,
                )

    return PotentialBreakdown(
        cash=float(farm["money"]),
        liquid_products=float(liquid_products),
        seeds=float(seeds),
        unplaced_animals=float(unplaced_animals),
        growing_crops=float(growing_crops),
        placed_animals=float(placed_animals),
    )


def economic_potential(
    observation: Mapping[str, Any],
    config: RewardConfig | None = None,
) -> float:
    """Başlangıç parasına göre normalize ekonomik potansiyel."""
    config = config or RewardConfig()
    breakdown = economic_potential_breakdown(observation, config)
    return (breakdown.total - config.starting_money) / config.reward_scale


def terminal_gold_difference_reward(
    observation: Mapping[str, Any],
    done: bool,
    config: RewardConfig | None = None,
) -> float:
    """Episode sonunda rakibe karşı gold farkını normalize eder."""
    if not done:
        return 0.0
    config = config or RewardConfig()
    player = int(observation["player"])
    own_gold = float(observation["farms"][player]["money"])
    opponent_gold = float(observation["farms"][1 - player]["money"])
    return (own_gold - opponent_gold) / config.reward_scale


def terminal_profit_reward(
    observation: Mapping[str, Any],
    done: bool,
    config: RewardConfig | None = None,
) -> float:
    """Geriye uyumlu ad; artık rakibe karşı gold farkını döndürür."""
    return terminal_gold_difference_reward(observation, done, config)


def transition_reward(
    observation: Mapping[str, Any],
    next_observation: Mapping[str, Any],
    done: bool,
    config: RewardConfig | None = None,
) -> TransitionReward:
    """Bir environment geçişinin base, shaping ve toplam reward'ı."""
    config = config or RewardConfig()
    current_breakdown = economic_potential_breakdown(observation, config)
    current_potential = (
        current_breakdown.total - config.starting_money
    ) / config.reward_scale

    if done:
        next_breakdown = None
        next_potential = 0.0
    else:
        next_breakdown = economic_potential_breakdown(next_observation, config)
        next_potential = (
            next_breakdown.total - config.starting_money
        ) / config.reward_scale

    base = terminal_gold_difference_reward(next_observation, done, config)
    shaping = config.shaping_beta * (
        config.gamma * next_potential - current_potential
    )
    return TransitionReward(
        total=base + shaping,
        base=base,
        shaping=shaping,
        current_potential=current_potential,
        next_potential=next_potential,
        current_breakdown=current_breakdown,
        next_breakdown=next_breakdown,
        lifecycle_credit=0.0,
    )
