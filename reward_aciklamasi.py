"""Kaggriculture reward sistemini okuyarak öğrenmek ve denetlemek için araç.

Bu dosya EĞİTİM KODUNUN YERİNE GEÇMEZ ve reward'ı değiştirmez. Gerçek
hesaplama ``reward.py``, lifecycle stratejisi ``lifecycle_credit.py`` ve PPO'ya
bağlantı ``training.py`` içindedir. Buradaki amaç, formülü tek bir açıklamalı
dosyada toplamak ve kayıtlı trajectory üzerinden sayıları kontrol etmektir.


1. Terminal (base) reward
=========================

Oyun boyunca base reward sıfırdır. Episode bittiğinde:

    base = (our_final_gold - opponent_final_gold) / reward_scale

Biz 4200$, rakip 3600$ ile bitirirse:

    base = (4200 - 3600) / 3000 = 0.20

Bu, agent'ı yalnızca para kazanmaya değil rakipten daha fazla para kazanmaya
yöneltir. Satılmayan stok ve gereksiz harcamalar final gold'u azalttığı için
terminal reward tarafından cezalandırılır.


2. Ekonomik potential
======================

Terminal parayı 720 adım beklemek öğrenmeyi zorlaştırır. Ara state'in ekonomik
değerini Phi(s) ile yaklaşık olarak hesaplıyoruz:

    Phi(s) = (
        cash
        + liquid_products
        + usable_seeds
        + unplaced_animals
        + growing_crops
        + placed_animals
        - starting_money
    ) / reward_scale

Burada seed'in değeri sezon bitmeden hasat verebiliyorsa sayılır. Ürünler
mevcut dinamik market eğrisiyle tasfiye ediliyormuş gibi değerlenir. Bitki ve
hayvanlarda kalan sezon, üretim zamanı, sulama/gübre ve yem maliyeti hesaba girer.


3. Potential-based shaping (varsayılan kapalı)
===============================================

Her transition'daki ara reward:

    shaping = beta * (gamma * Phi(next_state) - Phi(state))

Yeni reward tasarımında ``shaping_beta = 0`` olduğu için bu değer loglama ve
eski deney uyumluluğu dışında reward'a girmez. Ara ekonomik bonus gizlice
hedefi etkilemez.


4. Lifecycle credit
====================

``harvest_value`` modunda yalnızca gerçekten oluşan HARVESTED event'i için:

    raw_credit = harvested_units * current_market_price / reward_scale
    lifecycle_credit = coefficient * raw_credit

Bu bonus potential-based değildir; dolayısıyla hedefi değiştirebilir ve hasat
değerini terminal kârın yanında ikinci kez ödüllendirebilir. Bu nedenle varsayılanı
kapalıdır ve deneylerde küçük katsayıyla karşılaştırılmalıdır.

``causal_chain`` varsayılan moddur. Lifecycle değeri global transition
reward'ina eklenmez. Split-credit v2 iki ayrı hedef kullanır.

Satın alma span'ı tam seed lot ekonomisini alır:

    purchase_credit = (
        lot_descendant_sale_revenue - full_seed_order_cost
    ) / reward_scale

Bu değer BUY_SEED + crop + quantity tokenlarına gider. 50 seed alıp yalnızca
birinden 51$ gelir elde eden 10$'lık seed order'ı ``(51-500)/3000`` kredi
alır. Hiç satış yoksa kredi ``-full_seed_order_cost/reward_scale`` olur.

Her satılan plant'in üretim kolu kardeş seed israfından bağımsızdır:

    plant_net = (plant_sale_revenue - one_seed_cost) / reward_scale
    production_action_credit = plant_net / production_action_count

Bu ikinci kredi yalnızca o plant'in PLANT, WATER/FERTILIZE, HARVEST, shed'e
DROP/PLACE ve SELL aksiyonlarına eşit yazılır. Satılmayan plant'in üretim
aksiyonları kredi almaz; maliyeti purchase span'inda kalır. Zorunlu yapısal
tokenlar ``decision_mask=False`` olduğundan lifecycle loss'a girmez.

``BUY_PRODUCT`` ile dışarıdan alınıp geri satılan ürünlerin
``source_plant_id`` değeri yoktur. Bu satışlar lifecycle SELL kredisi almaz;
aksi durumda agent zararına BUY_PRODUCT -> SELL yaparak bonus toplayabilirdi.
Bu ticaretin iyi/kötü sonucu terminal gold farkına bırakılır.


5. PPO'ya giren son reward
===========================

    total = terminal_gold_difference + shaping

Varsayılan ``shaping_beta=0`` ve ``causal_chain`` global transition'a
eklenmediği için episode içindeki global reward'lar sıfır, terminal adımdaki
global reward ise normalize gold farkıdır.

``EpisodeRollout.records()`` bu total dizisinden global GAE advantage ve critic
return target hesaplar. Global loss asıl terminal para hedefini korur.

``causal_chain`` aktifse buna ek olarak her kredili token için ayrı clipped PPO
oranı hesaplanır:

    token_ratio = exp(new_token_log_prob - old_token_log_prob)
    lifecycle_loss = -mean(min(token_ratio * token_advantage,
                               clipped_token_ratio * token_advantage))

Bu ikinci loss yalnızca ``lifecycle_token_credits.json`` içindeki step/span ile
eşleşen stochastic tokenlara gradient gönderir. Global advantage yine bütün
bileşik action'ın uzun vadeli ekonomik sonucunu öğretmeye devam eder.


Kullanım
========

Bir episode klasörünü özetle:

    .venv/bin/python reward_aciklamasi.py runs/RUN/episodes/EPISODE

Yalnızca lifecycle credit alan adımları göster:

    .venv/bin/python reward_aciklamasi.py EPISODE --lifecycle-only

Belirli bir environment adımını ayrıntılı göster:

    .venv/bin/python reward_aciklamasi.py EPISODE --step 123
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True, slots=True)
class SavedReward:
    """Trajectory'deki bir transition reward'inin okunabilir görünümü."""

    step: int
    day: int
    hour: int
    base: float
    shaping: float
    lifecycle_credit: float
    total: float
    current_potential: float
    next_potential: float
    action: Mapping[str, Any]
    lifecycle_assignments: tuple[Mapping[str, Any], ...]

    @property
    def reconstructed_total(self) -> float:
        """Loglanan bileşenlerden total'i yeniden kurar."""
        return self.base + self.shaping + self.lifecycle_credit

    @property
    def decomposition_is_valid(self) -> bool:
        """total = base + shaping + lifecycle_credit kontrolü."""
        return math.isclose(
            self.total,
            self.reconstructed_total,
            rel_tol=1e-7,
            abs_tol=1e-9,
        )

    @classmethod
    def from_trace_row(cls, row: Mapping[str, Any]) -> "SavedReward":
        reward = row["reward"]
        return cls(
            step=int(row["step"]),
            day=int(row["day"]),
            hour=int(row["hour"]),
            base=float(reward.get("base", 0.0)),
            shaping=float(reward.get("shaping", 0.0)),
            # Eski trace'lerde bu alan yoktur; onlar lifecycle bağlanmadan
            # önce kaydedildiği için doğru geriye-uyumlu değer 0'dır.
            lifecycle_credit=float(reward.get("lifecycle_credit", 0.0)),
            total=float(reward["total"]),
            current_potential=float(reward.get("current_potential", 0.0)),
            next_potential=float(reward.get("next_potential", 0.0)),
            action=row.get("learner_action", {}),
            lifecycle_assignments=tuple(
                row.get("lifecycle_credit_assignments", ())
            ),
        )


def _trajectory_path(path: Path) -> Path:
    """Episode klasörü veya doğrudan trajectory dosyası kabul eder."""
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        gzip_path = resolved / "trajectory.jsonl.gz"
        plain_path = resolved / "trajectory.jsonl"
        if gzip_path.exists():
            return gzip_path
        if plain_path.exists():
            return plain_path
        raise FileNotFoundError(f"Episode içinde trajectory bulunamadı: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(f"Trajectory bulunamadı: {resolved}")
    return resolved


def _open_trajectory(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_saved_rewards(path: Path) -> Iterator[SavedReward]:
    """Trajectory satırlarını belleğe yığmadan reward nesnelerine çevirir."""
    trajectory = _trajectory_path(path)
    with _open_trajectory(trajectory) as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield SavedReward.from_trace_row(json.loads(line))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Geçersiz trajectory satırı: {trajectory}:{line_number}"
                ) from error


def _short_action(action: Mapping[str, Any]) -> str:
    """Tabloyu taşırmadan action'ın operasyon isimlerini gösterir."""
    operations: list[str] = []
    farmer = action.get("farmer", ())
    if farmer:
        operations.append(f"F:{farmer[0]}")
    for index, hand in enumerate(action.get("hands", ())):
        if hand:
            operations.append(f"H{index}:{hand[0]}")
    for order in action.get("market", ()):
        if order:
            operations.append(f"M:{order[0]}")
    return ",".join(operations) or "-"


def print_reward_table(rows: list[SavedReward]) -> None:
    """Transition reward'lerini hizalı, kısa bir tablo halinde basar."""
    print(
        "step day hour       base    shaping  lifecycle      total "
        " phi_before  phi_after  action"
    )
    for row in rows:
        marker = "" if row.decomposition_is_valid else "  [TOPLAM HATALI]"
        print(
            f"{row.step:4d} {row.day:3d} {row.hour:4d} "
            f"{row.base:10.5f} {row.shaping:10.5f} "
            f"{row.lifecycle_credit:10.5f} {row.total:10.5f} "
            f"{row.current_potential:11.5f} {row.next_potential:10.5f} "
            f"{_short_action(row.action)}{marker}"
        )


def print_episode_summary(rows: list[SavedReward]) -> None:
    """Episode toplamını bileşenlerine ayırır ve tutarlılığı denetler."""
    base = sum(row.base for row in rows)
    shaping = sum(row.shaping for row in rows)
    lifecycle = sum(row.lifecycle_credit for row in rows)
    total = sum(row.total for row in rows)
    reconstructed = base + shaping + lifecycle
    invalid = [row.step for row in rows if not row.decomposition_is_valid]

    print("\nEpisode reward özeti")
    print(f"  transition sayısı : {len(rows)}")
    print(f"  base toplamı        : {base:.8f}")
    print(f"  shaping toplamı     : {shaping:.8f}")
    print(f"  lifecycle toplamı   : {lifecycle:.8f}")
    print(f"  total toplamı       : {total:.8f}")
    print(f"  yeniden kurulan total : {reconstructed:.8f}")
    print(f"  decomposition         : {'OK' if not invalid else 'HATALI'}")
    if invalid:
        print(f"  hatalı step'ler       : {invalid}")


def print_step_detail(row: SavedReward) -> None:
    """Tek transition'ı formül ve lifecycle kaynaklarıyla açıklar."""
    print(f"Step {row.step} (day={row.day}, hour={row.hour})")
    print(f"Action: {json.dumps(row.action, ensure_ascii=False)}")
    print()
    print(f"base             = {row.base:.8f}")
    print(f"shaping          = {row.shaping:.8f}")
    print(f"lifecycle_credit = {row.lifecycle_credit:.8f}")
    print(
        "total             = base + shaping + lifecycle_credit "
        f"= {row.reconstructed_total:.8f}"
    )
    print(f"loglanan total   = {row.total:.8f}")
    print(
        f"potential         = {row.current_potential:.8f} -> "
        f"{row.next_potential:.8f}"
    )
    print(
        "kontrol           = "
        f"{'OK' if row.decomposition_is_valid else 'HATALI'}"
    )

    if row.lifecycle_assignments:
        print("\nLifecycle credit kaynakları:")
        for assignment in row.lifecycle_assignments:
            print(
                "  - "
                f"event={assignment.get('event_id')} "
                f"type={assignment.get('event_type')} "
                f"plant={assignment.get('plant_id')} "
                f"token_span={assignment.get('token_start')}.."
                f"{assignment.get('token_end')} "
                f"amount={float(assignment.get('amount', 0.0)):.8f}"
            )
    else:
        print("\nBu transition'a lifecycle credit atanmamış.")


def print_token_credits(path: Path) -> None:
    """Episode sonunda üretilen geriye-dönük token kredilerini basar."""
    resolved = path.expanduser().resolve()
    episode_directory = resolved if resolved.is_dir() else resolved.parent
    credits_path = episode_directory / "lifecycle_token_credits.json"
    if not credits_path.exists():
        raise FileNotFoundError(
            f"Lifecycle token credit artifact'i bulunamadı: {credits_path}"
        )
    rows = json.loads(credits_path.read_text(encoding="utf-8"))
    print("step role       amount      token_span target_event_type reason")
    for row in rows:
        print(
            f"{int(row['step']):4d} {str(row['role']):10s} "
            f"{float(row['amount']):10.6f} "
            f"{int(row['token_start']):3d}..{int(row['token_end']):<3d} "
            f"{str(row['target_event_type']):18s} {row['reason']}"
        )
    print(f"\nassignment sayısı: {len(rows)}")
    print(
        "pozitif toplam: "
        f"{sum(max(float(row['amount']), 0.0) for row in rows):.8f}"
    )
    print(
        "negatif toplam: "
        f"{sum(min(float(row['amount']), 0.0) for row in rows):.8f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kayıtlı Kaggriculture reward bileşenlerini açıkla.",
    )
    parser.add_argument(
        "trajectory",
        type=Path,
        help="Episode klasörü, trajectory.jsonl veya trajectory.jsonl.gz",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Yalnızca bu environment step'ini ayrıntılı göster.",
    )
    parser.add_argument(
        "--lifecycle-only",
        action="store_true",
        help="Yalnızca lifecycle credit alan transition'ları listele.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Tabloda gösterilecek en fazla satır; 0 tümünü gösterir.",
    )
    parser.add_argument(
        "--token-credits",
        action="store_true",
        help="Causal-chain token credit artifact'ini göster.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit < 0:
        raise ValueError("--limit negatif olamaz.")

    if args.token_credits:
        print_token_credits(args.trajectory)
        return 0

    all_rows = list(iter_saved_rewards(args.trajectory))
    if args.step is not None:
        matching = [row for row in all_rows if row.step == args.step]
        if not matching:
            raise ValueError(f"Trajectory'de step={args.step} bulunamadı.")
        for row in matching:
            print_step_detail(row)
        return 0

    visible = all_rows
    if args.lifecycle_only:
        visible = [row for row in visible if row.lifecycle_credit != 0.0]
    if args.limit > 0:
        visible = visible[: args.limit]

    print_reward_table(visible)
    if args.limit > 0 and len(visible) < len(all_rows) and not args.lifecycle_only:
        print(f"... tablo ilk {len(visible)} transition ile sınırlandı.")
    print_episode_summary(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
