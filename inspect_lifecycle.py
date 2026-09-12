"""Kaydedilmiş plant lifecycle artifact'lerini terminalde incele."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def _resolve_episode_directory(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir():
        return path
    if path.name in {
        "plant_lifecycles.json",
        "lifecycle_events.jsonl.gz",
        "lifecycle_summary.json",
    }:
        return path.parent
    raise ValueError(f"Episode klasörü veya lifecycle artifact'i bekleniyordu: {path}")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_events(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed -> plant -> harvest -> sale zincirini incele.",
    )
    parser.add_argument("episode", type=Path)
    parser.add_argument("--plant-id")
    parser.add_argument("--crop", type=str.upper)
    parser.add_argument("--only-harvested", action="store_true")
    parser.add_argument(
        "--events",
        action="store_true",
        help="Seçilen plantlarla bağlantılı ham eventleri de göster.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episode_directory = _resolve_episode_directory(args.episode)
    plants = _read_json(episode_directory / "plant_lifecycles.json")
    selected = [
        plant
        for plant in plants
        if (args.plant_id is None or plant["plant_id"] == args.plant_id)
        and (args.crop is None or plant["crop"] == args.crop)
        and (not args.only_harvested or plant["harvested_units"] > 0)
    ]
    payload: dict[str, Any] = {
        "summary": _read_json(episode_directory / "lifecycle_summary.json"),
        "selected_plant_count": len(selected),
        "plants": selected,
    }
    if args.events:
        selected_ids = {plant["plant_id"] for plant in selected}
        events = _read_events(episode_directory / "lifecycle_events.jsonl.gz")
        payload["events"] = [
            event
            for event in events
            if event.get("plant_id") in selected_ids
            or any(
                allocation.get("source_plant_id") in selected_ids
                for allocation in event.get("data", {}).get("allocations", [])
            )
        ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
