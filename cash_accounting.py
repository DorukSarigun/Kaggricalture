"""Tam nakit defteri; lifecycle ataması veya yeni bir reward üretmez.

Motorun bütün market execution'ları iki oyuncunun gerçek bakiye değişimiyle
her tur mutabık olmalıdır. Arazi, hayvan ve tüketilen girdiler de kapsamdadır.
Lot/FIFO bilgisi gerektirmez; eksik bir trace'i residual diye gizlemez.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Sequence, Any

from market_execution import MarketOrderExecution


@dataclass(frozen=True)
class CashEntry:
    step: int
    player: int
    order_index: int
    operation: str
    item: str | None
    quantity: int
    amount: float


class CashLedger:
    def __init__(self, initial_money: Sequence[float]) -> None:
        self.initial_money = tuple(float(x) for x in initial_money)
        if len(self.initial_money) != 2 or not all(map(math.isfinite, self.initial_money)):
            raise ValueError("CashLedger iki sonlu başlangıç bakiyesi gerektirir.")
        self.final_money = self.initial_money
        self.entries: list[CashEntry] = []
        self.transitions = 0
        self.last_step: int | None = None
        self.max_reconciliation_error = 0.0

    def observe(self, step: int, executions: Sequence[Sequence[MarketOrderExecution]],
                before_money: Sequence[float], after_money: Sequence[float]) -> None:
        if len(executions) != 2 or len(before_money) != 2 or len(after_money) != 2:
            raise ValueError("CashLedger iki oyuncu gerektirir.")
        if self.last_step is not None and step <= self.last_step:
            raise ValueError("Nakit transition adımları artmalı.")
        rows = []
        max_error = 0.0
        for player in range(2):
            before, after = float(before_money[player]), float(after_money[player])
            if not all(map(math.isfinite, (before, after))):
                raise ValueError("Sonlu olmayan bakiye.")
            if not math.isclose(before, self.final_money[player], abs_tol=1e-7, rel_tol=0):
                raise RuntimeError("Nakit defterinde eksik transition.")
            net = 0.0
            for order in executions[player]:
                if order.player != player:
                    raise ValueError("Execution oyuncusu uyuşmuyor.")
                if order.operation not in {"SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND"}:
                    raise ValueError(f"Bilinmeyen nakit operasyonu: {order.operation}")
                amount = order.total_value * (1 if order.operation == "SELL" else -1)
                if not math.isfinite(amount):
                    raise ValueError("Sonlu olmayan execution tutarı.")
                net += amount
                rows.append(CashEntry(step, player, order.order_index, order.operation,
                                      order.item, order.executed_quantity, amount))
            error = after - before - net
            max_error = max(max_error, abs(error))
            if abs(error) > 1e-7:
                raise RuntimeError(f"Cash reconciliation failed: step={step} player={player} "
                                   f"actual={after-before} traced={net} error={error}")
        self.entries.extend(rows)
        self.final_money = tuple(float(x) for x in after_money)
        self.transitions += 1
        self.last_step = step
        self.max_reconciliation_error = max(self.max_reconciliation_error, max_error)

    def summary(self, player: int) -> dict[str, Any]:
        by_operation: dict[str, float] = defaultdict(float)
        for entry in self.entries:
            if entry.player == player:
                by_operation[entry.operation] += entry.amount
        return {
            "initial_money": self.initial_money[player],
            "final_money": self.final_money[player],
            "net_cash": self.final_money[player] - self.initial_money[player],
            "by_operation": dict(by_operation),
            "transitions": self.transitions,
            "max_reconciliation_error": self.max_reconciliation_error,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "players": [self.summary(p) for p in range(2)],
                "entries": [asdict(entry) for entry in self.entries]}
