"""Backtesting con el MISMO pipeline que en Demo (Fase 4).

ticks -> estrategia -> ExecutionController -> RiskManager -> SimulatedBroker,
con reloj simulado (epoch del tick). Registra órdenes/trades en SQLite y
devuelve métricas netas de payout (no solo % de aciertos).
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Iterable

from broker.simulated_broker import SimulatedBroker
from config import Settings
from core import event_manager as ev
from core.event_manager import EventManager
from core.execution_controller import ExecutionController
from database.database import Database
from database.trades import TradeRepository
from market.market_data import Tick
from risk.risk_manager import RiskLimits, RiskManager
from strategy.strategy import Strategy


@dataclass
class BacktestReport:
    ticks: int = 0
    signals: int = 0
    rejected: int = 0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    pnl: float = 0.0
    staked: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    breakeven_win_rate: float = 0.0
    ev_per_trade: float = 0.0
    roi: float = 0.0
    final_balance: float = 0.0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    equity_curve: list[float] = field(default_factory=list)

    def to_dict(self, with_curve: bool = False) -> dict:
        d = asdict(self)
        if not with_curve:
            d.pop("equity_curve")
        return d


class Backtester:
    def __init__(self, strategy: Strategy, *, settings: Settings | None = None,
                 limits: RiskLimits | None = None, payout_ratio: float = 1.95,
                 latency_ticks: int = 0, initial_balance: float = 10_000.0,
                 history: int = 500, db: Database | None = None) -> None:
        self.strategy = strategy
        self.settings = settings or Settings()
        self.broker = SimulatedBroker(balance=initial_balance, payout_ratio=payout_ratio,
                                      latency_ticks=latency_ticks)
        self.risk = RiskManager(limits or RiskLimits(), clock=self.broker.clock)
        self.db = db or Database(":memory:")
        self.repo = TradeRepository(self.db, clock=self.broker.clock)
        self.events = EventManager()
        self.controller = ExecutionController(self.settings, self.broker, self.risk,
                                              repo=self.repo, events=self.events,
                                              clock=self.broker.clock)
        self.history = history
        self.payout_ratio = payout_ratio
        self.initial_balance = initial_balance
        self.report = BacktestReport(breakeven_win_rate=1 / payout_ratio)
        self._equity = 0.0
        self._peak = 0.0
        self.events.subscribe(ev.CONTRACT_SETTLED, self._on_settled)
        self.events.subscribe(ev.ORDER_REJECTED, self._on_rejected)

    def _on_settled(self, _topic: str, p: dict) -> None:
        r = self.report
        profit = p["profit"]
        r.trades += 1
        if profit > 0:
            r.wins += 1
        else:
            r.losses += 1
        self._equity += profit
        self._peak = max(self._peak, self._equity)
        r.max_drawdown = max(r.max_drawdown, self._peak - self._equity)
        r.equity_curve.append(round(self._equity, 2))
        self.strategy.on_settled(profit)

    def _on_rejected(self, _topic: str, p: dict) -> None:
        self.report.rejected += 1
        key = (p.get("reasons") or ["?"])[0].split(":")[0][:60]
        self.report.reject_reasons[key] = self.report.reject_reasons.get(key, 0) + 1

    def run(self, ticks: Iterable[Tick], symbol: str | None = None) -> BacktestReport:
        window: deque[Tick] = deque(maxlen=self.history)
        for tick in ticks:
            if symbol and tick.symbol != symbol:
                continue
            self.report.ticks += 1
            self.broker.on_tick(tick)           # primero liquida con el tick actual
            window.append(tick)
            decision = self.strategy.evaluate(tick.symbol, list(window),
                                              self.broker.balance, self.broker.now)
            if decision.order is not None:
                self.report.signals += 1
                self.controller.submit(decision.order)
        return self._finalize()

    def _finalize(self) -> BacktestReport:
        r = self.report
        ties = sum(1 for u in self.broker.settled
                   if u.entry_spot is not None and u.exit_spot == u.entry_spot)
        r.ties = ties
        r.pnl = round(self._equity, 2)
        r.staked = round(sum(u.buy_price or 0 for u in self.broker.settled), 2)
        r.win_rate = r.wins / r.trades if r.trades else 0.0
        r.ev_per_trade = r.pnl / r.trades if r.trades else 0.0
        r.roi = r.pnl / r.staked if r.staked else 0.0
        r.final_balance = round(self.broker.balance, 2)
        return r

    def save(self, name: str, params: dict | None = None) -> int:
        """Guarda el resumen en ``audit_events`` (los trades ya están en orders/trades)."""
        payload = {"name": name, "params": params or {}, "report": self.report.to_dict(),
                   "ts": time.time()}
        self.db.log_event("backtest", "run", json.dumps(payload, default=str))
        return self.report.trades


def max_drawdown(curve: list[float]) -> float:
    peak, mdd = -math.inf, 0.0
    for x in curve:
        peak = max(peak, x)
        mdd = max(mdd, peak - x)
    return mdd
