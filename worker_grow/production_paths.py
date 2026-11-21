"""Single-product manager training paths.

ISSUE admission and production market buys/sells can be restricted to a
subset of TASK_ITEMS (for example only CARROT, or only COW). The manager
architecture is unchanged: item_score still has eight logits; the mask
closes the others. Omit --issue-items for the unrestricted eight-product game.

Animals may BUY_PRODUCT WHEAT as feed. They cannot BUY_SEED wheat or SELL
wheat, so the path is not a wheat-product flip. Crops cannot buy animals
or wheat product.
"""
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from action_encoding import ActionGrammarConfig
from worker_grow.contracts import TASK_ITEMS
from worker_grow.recipes import ANIMAL_PRODUCT

PATH_ITEMS = TASK_ITEMS


def parse_issue_items(raw: str | None | tuple[str, ...] | list[str]) -> tuple[str, ...] | None:
    """None / blank → unrestricted. Otherwise a unique ordered TASK_ITEMS tuple."""
    if raw is None:
        return None
    if isinstance(raw, (tuple, list)):
        parts = [str(piece).strip().upper() for piece in raw]
    else:
        text = str(raw).strip()
        if not text:
            return None
        parts = [piece.strip().upper() for piece in text.replace(";", ",").split(",")]
    parts = [piece for piece in parts if piece]
    if not parts:
        raise ValueError("--issue-items is empty")
    unknown = [piece for piece in parts if piece not in TASK_ITEMS]
    if unknown:
        raise ValueError(
            f"Unknown issue item(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(TASK_ITEMS)}"
        )
    unique: list[str] = []
    for piece in parts:
        if piece not in unique:
            unique.append(piece)
    return tuple(unique)


def config_issue_items(issue_items: tuple[str, ...] | None) -> list[str] | None:
    """JSON/checkpoint form: list or null. Unrestricted is null, not all eight names."""
    if issue_items is None:
        return None
    return list(issue_items)


def sell_products(item: str) -> tuple[str, ...]:
    if item in game.CROPS:
        return (item,)
    if item in game.ANIMALS:
        return (ANIMAL_PRODUCT[item], "FERTILIZER")
    raise KeyError(item)


def market_item_allowlist(issue_items: tuple[str, ...]) -> dict[str, frozenset[str]]:
    seeds = frozenset(item for item in issue_items if item in game.CROPS)
    animals = frozenset(item for item in issue_items if item in game.ANIMALS)
    sold: list[str] = []
    for item in issue_items:
        sold.extend(sell_products(item))
    return {
        "BUY_SEED": seeds,
        "BUY_ANIMAL": animals,
        "BUY_PRODUCT": frozenset({"WHEAT"} if animals else ()),
        "SELL": frozenset(sold),
    }


def grammar_for_issue_items(issue_items: tuple[str, ...] | None) -> ActionGrammarConfig | None:
    if issue_items is None:
        return None
    return ActionGrammarConfig(allowed_market_items=market_item_allowlist(issue_items))


def path_slug(item: str) -> str:
    if item not in TASK_ITEMS:
        raise KeyError(item)
    return item.lower()
