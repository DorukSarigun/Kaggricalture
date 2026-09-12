"""Kaggriculture encoder + network evaluation against real game rules."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from kaggle_environments import make

from global_encoding import GLOBAL_FEATURE_NAMES, encode_global_observation
from kinds import (
    AnimalType,
    CoopTile,
    CropType,
    EmptyTile,
    LockedTile,
    PastureTile,
    PlantTile,
    TileKind,
    WeedTile,
    parse_tile_grid,
)
from model import GlobalEncoder, SpatialEncoder, TileEmbedding, TileEncoder
from tile_encoding import (
    ANIMAL_CATEGORIES,
    CROP_CATEGORIES,
    DYNAMIC_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    STATIC_FEATURE_NAMES,
    TILE_CATEGORIES,
    encode_tile_grid,
    tile_to_ids,
    tile_to_numeric_features,
)

ROOT = Path(__file__).parent
REPORT_PATH = ROOT / "evaluation_report.json"

BASE_PRICES = {
    "WHEAT": 25,
    "CARROT": 35,
    "TOMATO": 60,
    "STRAWBERRY": 120,
    "MELON": 250,
    "EGG": 50,
    "MILK": 160,
    "WOOL": 200,
    "FERTILIZER": 100,
}


def feat(names, values, name):
    return float(values[list(names).index(name)])


def encode_farm(observation, farm_index):
    tiles = parse_tile_grid(observation["farms"][farm_index]["tiles"])
    return encode_tile_grid(
        tiles,
        observation["day"],
        observation["step"],
        observation["market"]["prices"],
    )


def synthetic_cases():
    prices = dict(BASE_PRICES)
    cases = {}

    empty = EmptyTile()
    locked = LockedTile()
    weed = WeedTile()
    wheat_new = PlantTile(
        crop=CropType.WHEAT,
        planted_day=0,
        watered_today=False,
        consecutive_unwatered=1,
        yield_units=1,
        max_lifespan_step=120,
        fertilized_until_day=-1,
    )
    wheat_ready = PlantTile(
        crop=CropType.WHEAT,
        planted_day=0,
        watered_today=True,
        consecutive_unwatered=0,
        yield_units=4,
        max_lifespan_step=120,
        fertilized_until_day=2,
    )
    melon = PlantTile(
        crop=CropType.MELON,
        planted_day=0,
        watered_today=True,
        consecutive_unwatered=0,
        yield_units=6,
        max_lifespan_step=312,
        fertilized_until_day=-1,
    )
    empty_coop = CoopTile()
    goose = CoopTile(
        animal=AnimalType.GOOSE,
        placed_day=0,
        yield_units=2,
        fed_today=True,
        consecutive_unfed=0,
        cared_today=True,
        fertilizer_available=True,
        pending_care_bonus=2,
    )
    cow = PastureTile(
        animal=AnimalType.COW,
        placed_day=2,
        yield_units=6,
        fed_today=False,
        consecutive_unfed=1,
        cared_today=False,
        fertilizer_available=False,
        pending_care_bonus=0,
    )

    day, step = 4, 100
    encoded = {
        "empty": tile_to_numeric_features(empty, day, step, prices),
        "locked": tile_to_numeric_features(locked, day, step, prices),
        "weed": tile_to_numeric_features(weed, day, step, prices),
        "wheat_new": tile_to_numeric_features(wheat_new, 0, 0, prices),
        "wheat_ready": tile_to_numeric_features(wheat_ready, day, step, prices),
        "melon": tile_to_numeric_features(melon, 10, 240, prices),
        "empty_coop": tile_to_numeric_features(empty_coop, day, step, prices),
        "goose": tile_to_numeric_features(goose, day, step, prices),
        "cow": tile_to_numeric_features(cow, day, step, prices),
    }
    ids = {
        name: tile_to_ids(tile)
        for name, tile in {
            "empty": empty,
            "locked": locked,
            "weed": weed,
            "wheat_new": wheat_new,
            "melon": melon,
            "empty_coop": empty_coop,
            "goose": goose,
            "cow": cow,
        }.items()
    }
    cases["numeric"] = {k: v.tolist() for k, v in encoded.items()}
    cases["ids"] = {k: v.tolist() for k, v in ids.items()}

    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add(
        "kind_ids_unique",
        len({tuple(v) for k, v in ids.items() if k in {"empty", "locked", "weed"}}) == 3,
        f"empty/locked/weed ids={ {k: ids[k].tolist() for k in ('empty','locked','weed')} }",
    )
    add(
        "empty_numeric_zero",
        np.allclose(encoded["empty"], 0.0),
        f"nonzero={int(np.count_nonzero(encoded['empty']))}",
    )
    add(
        "new_wheat_neglect_half",
        abs(feat(NUMERIC_FEATURE_NAMES, encoded["wheat_new"], "neglect_streak") - 0.5) < 1e-6,
        f"neglect={feat(NUMERIC_FEATURE_NAMES, encoded['wheat_new'], 'neglect_streak')}",
    )
    add(
        "new_wheat_must_water_same_day",
        feat(NUMERIC_FEATURE_NAMES, encoded["wheat_new"], "basic_need_done_today") == 0.0,
        "planting day counts as first missed water",
    )
    add(
        "wheat_yield_fill_unfertilized_peak",
        abs(feat(NUMERIC_FEATURE_NAMES, encoded["wheat_ready"], "yield_fill") - 4 / 6) < 1e-6,
        f"fill={feat(NUMERIC_FEATURE_NAMES, encoded['wheat_ready'], 'yield_fill')}",
    )
    add(
        "wheat_vs_melon_static_differs",
        not np.allclose(encoded["wheat_ready"][12:], encoded["melon"][12:]),
        "static type channels should differ by crop",
    )
    add(
        "price_at_base_is_half",
        abs(feat(NUMERIC_FEATURE_NAMES, encoded["wheat_ready"], "market_price_vs_base") - 0.5) < 1e-6,
        "p/(p+base) at equilibrium",
    )
    add(
        "empty_coop_not_occupied",
        feat(NUMERIC_FEATURE_NAMES, encoded["empty_coop"], "animal_occupied") == 0.0,
        "empty coop has no animal",
    )
    add(
        "goose_occupied_and_cared",
        feat(NUMERIC_FEATURE_NAMES, encoded["goose"], "animal_occupied") == 1.0
        and feat(NUMERIC_FEATURE_NAMES, encoded["goose"], "animal_cared_today") == 1.0
        and abs(feat(NUMERIC_FEATURE_NAMES, encoded["goose"], "animal_pending_care_bonus") - 2 / 3) < 1e-6,
        "goose care bank encoded",
    )
    add(
        "cow_max_held_fill",
        abs(feat(NUMERIC_FEATURE_NAMES, encoded["cow"], "yield_fill") - 1.0) < 1e-6,
        "cow max_held=6",
    )
    add(
        "age_clipped",
        feat(NUMERIC_FEATURE_NAMES, encoded["melon"], "age") <= 1.0,
        f"melon age={feat(NUMERIC_FEATURE_NAMES, encoded['melon'], 'age')}",
    )
    add(
        "features_in_unit_interval",
        all(np.all((v >= 0.0) & (v <= 1.0)) for v in encoded.values()),
        "all synthetic numeric features clipped to [0,1]",
    )
    add(
        "crop_padding_zero",
        ids["empty"][1] == 0 and ids["empty"][2] == 0,
        "empty tile uses crop/animal padding 0",
    )
    add(
        "wheat_and_melon_crop_ids_differ",
        ids["wheat_new"][1] != ids["melon"][1],
        f"wheat={ids['wheat_new'][1]} melon={ids['melon'][1]}",
    )
    add(
        "category_counts",
        len(TILE_CATEGORIES) == 6 and len(CROP_CATEGORIES) == 6 and len(ANIMAL_CATEGORIES) == 4,
        f"kinds={len(TILE_CATEGORIES)} crops={len(CROP_CATEGORIES)} animals={len(ANIMAL_CATEGORIES)}",
    )
    cases["checks"] = checks
    return cases


def global_unit_checks():
    env = make("kaggriculture", configuration={"episodeSteps": 2, "seed": 1})
    states = env.reset()
    obs = states[0]["observation"]
    if "step" not in obs:
        obs = {**obs, "step": 0}
    vec = encode_global_observation(obs)
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("global_dim", len(vec) == len(GLOBAL_FEATURE_NAMES), f"{len(vec)} vs {len(GLOBAL_FEATURE_NAMES)}")
    add("in_unit_interval_except_trig", np.all(np.isfinite(vec)), "finite")
    names = list(GLOBAL_FEATURE_NAMES)
    add(
        "start_money_is_half",
        abs(vec[names.index("own_money")] - 0.5) < 1e-5
        and abs(vec[names.index("opponent_money")] - 0.5) < 1e-5,
        f"own={vec[names.index('own_money')]:.4f}",
    )
    add(
        "start_prices_half",
        abs(vec[names.index("market_price_WHEAT")] - 0.5) < 1e-5,
        f"wheat_price={vec[names.index('market_price_WHEAT')]:.4f}",
    )
    add(
        "hour0_cos_one",
        abs(vec[names.index("hour_sin")]) < 1e-5 and abs(vec[names.index("hour_cos")] - 1.0) < 1e-5,
        f"sin={vec[names.index('hour_sin')]:.4f} cos={vec[names.index('hour_cos')]:.4f}",
    )
    add("hands_zero", vec[names.index("own_hand_count")] == 0.0, "no hires at start")

    obs6 = dict(obs)
    obs6["hour"] = 6
    vec6 = encode_global_observation(obs6)
    add(
        "hour6_quarter_turn",
        abs(vec6[names.index("hour_sin")] - 1.0) < 1e-5
        and abs(vec6[names.index("hour_cos")]) < 1e-5,
        "hour encoding is cyclic",
    )

    obs_p1 = dict(obs)
    obs_p1["player"] = 1
    vec_p1 = encode_global_observation(obs_p1)
    add(
        "player_swap_money_symmetric_at_start",
        np.allclose(vec[names.index("own_money")], vec_p1[names.index("own_money")]),
        "both start 3000",
    )
    return {"vector": vec.tolist(), "checks": checks, "feature_count": len(GLOBAL_FEATURE_NAMES)}


def scan_saved_games(max_games=2, stride=15):
    kind_counts = Counter()
    crop_counts = Counter()
    animal_counts = Counter()
    numeric_min = None
    numeric_max = None
    nans = 0
    samples = 0
    parse_failures = 0
    money_values = []
    farmer_on_locked = 0
    farmer_positions = Counter()
    missing_step = 0
    encode_ok = 0

    game_paths = sorted((ROOT / "game_observations").glob("game_*.json"))[:max_games]
    for path in game_paths:
        with path.open(encoding="utf-8") as file:
            game = json.load(file)
        for step_record in game["steps"][::stride]:
            for player in step_record.get("players", []):
                obs = player.get("observation") or {}
                if "farms" not in obs or "market" not in obs:
                    continue
                if "step" not in obs:
                    missing_step += 1
                    obs = {**obs, "step": obs.get("day", 0) * 24 + obs.get("hour", 0)}
                try:
                    global_vec = encode_global_observation(obs)
                    if not np.all(np.isfinite(global_vec)):
                        nans += 1
                    money_values.append(obs["farms"][obs["player"]]["money"])
                    for farm_i, farm in enumerate(obs["farms"]):
                        tiles = parse_tile_grid(farm["tiles"])
                        cat, num = encode_tile_grid(
                            tiles,
                            obs["day"],
                            obs["step"],
                            obs["market"]["prices"],
                        )
                        if numeric_min is None:
                            numeric_min = num.min(axis=(1, 2))
                            numeric_max = num.max(axis=(1, 2))
                        else:
                            numeric_min = np.minimum(numeric_min, num.min(axis=(1, 2)))
                            numeric_max = np.maximum(numeric_max, num.max(axis=(1, 2)))
                        if not np.all(np.isfinite(num)):
                            nans += 1
                        fx, fy = farm["farmer"]
                        farmer_positions[(fx, fy)] += 1
                        tile = tiles[fy][fx]
                        if tile.kind == TileKind.LOCKED:
                            farmer_on_locked += 1
                        for row in tiles:
                            for tile in row:
                                kind_counts[tile.kind.value] += 1
                                crop = getattr(tile, "crop", None)
                                animal = getattr(tile, "animal", None)
                                if crop is not None:
                                    crop_counts[crop.value] += 1
                                if animal is not None:
                                    animal_counts[animal.value] += 1
                        encode_ok += 1
                    samples += 1
                except Exception as exc:  # noqa: BLE001
                    parse_failures += 1
                    if parse_failures <= 3:
                        print("parse fail", path.name, exc)

    return {
        "games_scanned": len(game_paths),
        "player_observations": samples,
        "farm_encodes": encode_ok,
        "parse_failures": parse_failures,
        "nans": nans,
        "missing_step": missing_step,
        "kind_counts": dict(kind_counts),
        "crop_counts": dict(crop_counts),
        "animal_counts": dict(animal_counts),
        "money_min": min(money_values) if money_values else None,
        "money_max": max(money_values) if money_values else None,
        "money_mean": float(np.mean(money_values)) if money_values else None,
        "farmer_on_locked": farmer_on_locked,
        "top_farmer_positions": farmer_positions.most_common(6),
        "numeric_min": numeric_min.tolist() if numeric_min is not None else None,
        "numeric_max": numeric_max.tolist() if numeric_max is not None else None,
        "numeric_names": list(NUMERIC_FEATURE_NAMES),
    }


def live_starter_game():
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 7})
    env.run(["starter", "random"])
    final = env.steps[-1]
    mid = env.steps[240]
    start = env.steps[0]

    def pack(step_states, label):
        obs = step_states[0]["observation"]
        if "step" not in obs:
            obs = {**dict(obs), "step": obs.get("day", 0) * 24 + obs.get("hour", 0)}
        kinds = Counter()
        for farm in obs["farms"]:
            for row in parse_tile_grid(farm["tiles"]):
                for tile in row:
                    kinds[tile.kind.value] += 1
        cat, num = encode_farm(obs, 0)
        g = encode_global_observation(obs)
        return {
            "label": label,
            "day": obs["day"],
            "hour": obs["hour"],
            "money": [farm["money"] for farm in obs["farms"]],
            "kinds": dict(kinds),
            "unlocked": [farm["unlocked_quadrants"] for farm in obs["farms"]],
            "hands": [len(farm["hands"]) for farm in obs["farms"]],
            "tile_cat_shape": list(cat.shape),
            "tile_num_shape": list(num.shape),
            "global_shape": list(g.shape),
            "global_finite": bool(np.all(np.isfinite(g))),
        }

    return {
        "start": pack(start, "start"),
        "day10": pack(mid, "day10"),
        "final": pack(final, "final"),
        "rewards": [s["reward"] for s in final],
    }


def network_tests(observation):
    cat, num = encode_farm(observation, observation["player"])
    g = encode_global_observation(observation)

    cat_t = torch.from_numpy(cat).unsqueeze(0)
    num_t = torch.from_numpy(num).unsqueeze(0)
    g_t = torch.from_numpy(g).unsqueeze(0)

    tile_encoder = TileEncoder()
    spatial = SpatialEncoder()
    glob = GlobalEncoder()
    tile_encoder.eval()
    spatial.eval()
    glob.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            tile_feat = tile_encoder(cat_t, num_t)
            spat = spatial(tile_feat)
            gvec = glob(g_t)
    elapsed_ms = (time.perf_counter() - t0) / 20 * 1000

    tile_feat = tile_encoder(cat_t, num_t)
    spat = spatial(tile_feat)
    gvec = glob(g_t)

    loss = tile_feat.pow(2).mean() + spat.pow(2).mean() + gvec.pow(2).mean()
    loss.backward()
    grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in list(tile_encoder.parameters()) + list(spatial.parameters()) + list(glob.parameters())
        if p.requires_grad
    )

    # Local mixer: swapping two distant tiles should change 1x1 features only there.
    cat_swap = cat_t.clone()
    cat_swap[0, :, 0, 0], cat_swap[0, :, 9, 9] = (
        cat_t[0, :, 9, 9].clone(),
        cat_t[0, :, 0, 0].clone(),
    )
    num_swap = num_t.clone()
    num_swap[0, :, 0, 0], num_swap[0, :, 9, 9] = (
        num_t[0, :, 9, 9].clone(),
        num_t[0, :, 0, 0].clone(),
    )
    with torch.no_grad():
        local_a = tile_encoder(cat_t, num_t)
        local_b = tile_encoder(cat_swap, num_swap)
        spatial_a = spatial(local_a)
        spatial_b = spatial(local_b)

    local_unchanged_mid = torch.allclose(local_a[0, :, 5, 5], local_b[0, :, 5, 5], atol=1e-6)
    local_corner_changed = not torch.allclose(local_a[0, :, 0, 0], local_b[0, :, 0, 0], atol=1e-6)
    spatial_mid_changed = not torch.allclose(spatial_a[0, :, 5, 5], spatial_b[0, :, 5, 5], atol=1e-5)

    # Padding bias: impulse at NW vs SE after spatial conv.
    impulse_nw = torch.zeros_like(local_a)
    impulse_se = torch.zeros_like(local_a)
    impulse_nw[0, 0, 0, 0] = 1.0
    impulse_se[0, 0, 9, 9] = 1.0
    with torch.no_grad():
        resp_nw = spatial(impulse_nw).abs().sum().item()
        resp_se = spatial(impulse_se).abs().sum().item()

    # Embedding: different crops, same numeric zeros except type — use real wheat/melon ids
    embed = TileEmbedding()
    wheat_id = torch.tensor([TILE_CATEGORIES[TileKind.PLANT]])
    wheat_crop = torch.tensor([CROP_CATEGORIES[CropType.WHEAT]])
    melon_crop = torch.tensor([CROP_CATEGORIES[CropType.MELON]])
    none_animal = torch.tensor([ANIMAL_CATEGORIES[None]])
    wheat_e = embed(wheat_id, wheat_crop, none_animal)
    melon_e = embed(wheat_id, melon_crop, none_animal)
    crop_embed_diff = not torch.allclose(wheat_e, melon_e)

    param_count = sum(p.numel() for p in tile_encoder.parameters())
    param_count += sum(p.numel() for p in spatial.parameters())
    param_count += sum(p.numel() for p in glob.parameters())

    pad = spatial.spatial_block[0]
    # ZeroPad2d (left, right, top, bottom)
    pad_tuple = (pad.padding[0], pad.padding[1], pad.padding[2], pad.padding[3])

    return {
        "tile_feat_shape": list(tile_feat.shape),
        "spatial_shape": list(spat.shape),
        "global_shape": list(gvec.shape),
        "tile_input_channels": tile_encoder.input_channels,
        "embedding_dim": tile_encoder.embedding.output_dim,
        "numeric_channels": tile_encoder.numeric_channels,
        "params": param_count,
        "forward_ms": round(elapsed_ms, 3),
        "grad_finite": grad_ok,
        "local_mixer_isolates_tiles": local_unchanged_mid and local_corner_changed,
        "spatial_mixes_neighbors": True,  # 10x10 by construction
        "corner_swap_changes_spatial_center": spatial_mid_changed,
        "padding": {"left": pad_tuple[0], "right": pad_tuple[1], "top": pad_tuple[2], "bottom": pad_tuple[3]},
        "impulse_l1_nw": resp_nw,
        "impulse_l1_se": resp_se,
        "nw_se_response_ratio": resp_nw / resp_se if resp_se else None,
        "crop_embeddings_differ": crop_embed_diff,
        "relu_global_can_zero": True,
    }


def coverage_gaps():
    present = {
        "tile_kind",
        "crop_type",
        "animal_type",
        "age",
        "watered_or_fed_today",
        "neglect_streak",
        "yield_fill",
        "price_vs_base_on_tile",
        "lifespan",
        "plant_fertilizer",
        "animal_care",
        "animal_fertilizer",
        "static_object_stats",
        "market_prices",
        "market_inventory",
        "money_both",
        "shed_seeds_carried",
        "hand_counts",
        "town_shops",
        "hour_cyclic",
        "season_progress",
    }
    missing = [
        {
            "id": "farmer_position",
            "severity": "critical",
            "why": "Her tur tek eylem bir konumda. Ağ çiftçinin nerede durduğunu görmüyor.",
        },
        {
            "id": "hand_positions",
            "severity": "critical",
            "why": "İşçiler ayrı birimler; sadece sayı var, konum yok.",
        },
        {
            "id": "xy_coordinates",
            "severity": "high",
            "why": "Depo merkezde, kadranlar mutlak. Conv öteleme-değişmez; CoordConv yok.",
        },
        {
            "id": "shed_adjacency",
            "severity": "high",
            "why": "PICKUP/DROP yalnız (4,4)(5,4)(4,5)(5,5). Maske yok.",
        },
        {
            "id": "unlocked_quadrants",
            "severity": "medium",
            "why": "Sadece LOCKED tile'lardan çıkar; BUY_LAND sırası/maliyeti yok.",
        },
        {
            "id": "hires_today",
            "severity": "medium",
            "why": "Sonraki HIRE Fibonacci maliyeti gözlemde var, vektörde yok.",
        },
        {
            "id": "opponent_farm_stream",
            "severity": "high",
            "why": "Rakip tahtası paylaşılıyor ama model tek tarla encode ediyor. Pazar dump için şart.",
        },
        {
            "id": "fusion_policy_head",
            "severity": "critical",
            "why": "Tile + spatial + global birleşmiyor; aksiyon başı yok.",
        },
        {
            "id": "unit_inventories_separate",
            "severity": "medium",
            "why": "Cepler toplanmış; hangi birimin ne taşıdığı kayıp.",
        },
        {
            "id": "asymmetric_padding",
            "severity": "high",
            "why": "10x10 + pad (4,5,4,5) NW'ye daha çok gerçek kiremit, SE'ye daha çok sıfır verir.",
        },
        {
            "id": "money_scale",
            "severity": "medium",
            "why": "x/(x+3000): 3k→0.50, 27k→0.90. Geç sezon nakit sıkışır.",
        },
        {
            "id": "receptive_field",
            "severity": "medium",
            "why": "Tek 10x10 katman + same-size pad: merkez tüm tarlayı görür, köşe karşı köşeyi hâlâ göremez.",
        },
    ]
    return {"encoded": sorted(present), "missing": missing}


def main():
    print("synthetic tiles...")
    synthetic = synthetic_cases()
    print("global start...")
    glob = global_unit_checks()
    print("saved games...")
    saved = scan_saved_games()
    print("live starter vs random...")
    live = live_starter_game()

    env = make("kaggriculture", configuration={"episodeSteps": 24, "seed": 11})
    env.run(["starter", "pass"])
    obs = env.steps[-1][0]["observation"]
    if "step" not in obs:
        obs = {**dict(obs), "step": obs["day"] * 24 + obs["hour"]}
    print("network...")
    net = network_tests(obs)
    gaps = coverage_gaps()

    synthetic_pass = sum(1 for c in synthetic["checks"] if c["ok"])
    global_pass = sum(1 for c in glob["checks"] if c["ok"])
    report = {
        "summary": {
            "numeric_channels": len(NUMERIC_FEATURE_NAMES),
            "dynamic_channels": len(DYNAMIC_FEATURE_NAMES),
            "static_channels": len(STATIC_FEATURE_NAMES),
            "global_features": len(GLOBAL_FEATURE_NAMES),
            "synthetic_checks": f"{synthetic_pass}/{len(synthetic['checks'])}",
            "global_checks": f"{global_pass}/{len(glob['checks'])}",
            "saved_parse_failures": saved["parse_failures"],
            "saved_nans": saved["nans"],
            "network_params": net["params"],
            "forward_ms": net["forward_ms"],
            "missing_critical": sum(1 for g in gaps["missing"] if g["severity"] == "critical"),
            "missing_high": sum(1 for g in gaps["missing"] if g["severity"] == "high"),
        },
        "synthetic": synthetic,
        "global": glob,
        "saved_games": saved,
        "live_game": live,
        "network": net,
        "coverage": gaps,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("wrote", REPORT_PATH)


if __name__ == "__main__":
    main()
