"""Market dışı, ilk satılabilir birimi shed'e koyana kadar gereken unit aksiyonları.

Sayaç motor takviminden: her yaş gününde bir WATER/FEED, artı PLANT/PLACE,
ilk hasat ve DROP. BUY/SELL yok. Gübre ve CARE isteğe bağlıdır; paydaya girmez.
"""
from kaggle_environments.envs.kaggriculture import kaggriculture as game

# Yetiştirme hamleleri: üzerinde durulan veya eldeki ürüne bağlanır.
GROW_OPS = frozenset({
    "PLANT", "WATER", "FERTILIZE", "HARVEST",
    "PLACE", "FEED", "CARE", "COLLECT_FERTILIZER",
    "DROP", "PICKUP", "BUILD_COOP", "BUILD_PASTURE",
})

ANIMAL_PRODUCT = {name: spec["product"] for name, spec in game.ANIMALS.items()}
PRODUCT_ANIMAL = {product: animal for animal, product in ANIMAL_PRODUCT.items()}


def _first_cycle_waters(crop: str) -> int:
    return int(game.CROPS[crop]["first_yield_day"]) + 1


def _first_cycle_feeds(animal: str) -> int:
    return int(game.ANIMALS[animal]["first_yield_day"]) + 1


def crop_unit_actions(crop: str) -> int:
    """PLANT + günlük WATER (yaş 0..first_yield) + HARVEST + DROP."""
    if crop not in game.CROPS:
        raise KeyError(crop)
    return 1 + _first_cycle_waters(crop) + 1 + 1


def animal_unit_actions(animal: str) -> int:
    """BUILD + PLACE + günlük FEED (yaş 0..first_yield) + HARVEST + DROP."""
    if animal not in game.ANIMALS:
        raise KeyError(animal)
    return 1 + 1 + _first_cycle_feeds(animal) + 1 + 1


def required_unit_actions(product: str) -> int:
    """Ödül paydası: satılan ürün adına, o türün ilk döngü unit sayısı."""
    if product in game.CROPS:
        return crop_unit_actions(product)
    if product in PRODUCT_ANIMAL:
        return animal_unit_actions(PRODUCT_ANIMAL[product])
    if product == "FERTILIZER":
        return 1
    raise KeyError(f"Bilinmeyen ürün: {product}")


def market_product(name: str) -> str | None:
    """Tohum/hayvan/ürün adını fiyatı olan market ürününe çevirir."""
    if name in game.CROPS or name in game.PRODUCTS:
        return name
    if name in ANIMAL_PRODUCT:
        return ANIMAL_PRODUCT[name]
    return None


CROP_UNIT_ACTIONS = {crop: crop_unit_actions(crop) for crop in game.CROPS}
ANIMAL_UNIT_ACTIONS = {animal: animal_unit_actions(animal) for animal in game.ANIMALS}
