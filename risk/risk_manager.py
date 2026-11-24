"""Gestión de riesgo (PDF §7 «Límites»).

Comprueba importe máximo, pérdida diaria, contratos simultáneos, frecuencia,
pausa tras errores repetidos/pérdidas consecutivas y parada de emergencia.
Thread-safe. El día de trading se cuenta en UTC.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from strategy.signals import OrderRequest


@dataclass(frozen=True)
class RiskLimits:
    max_stake: float = 5.0
    max_daily_loss: float = 25.0
    max_open_contracts: int = 1
    max_trades_per_hour: int = 30
    min_seconds_between_trades: float = 5.0
    max_consecutive_errors: int = 3
    error_pause_s: float = 300.0
    max_consecutive_losses: int = 5
    loss_streak_pause_s: float = 900.0
    min_payout_ratio: float = 1.70       # payout/ask mínimo aceptable
    max_price_slippage: float = 0.02     # ask_price <= stake * (1 + 2 %)
    min_balance: float = 10.0


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reasons: tuple[str, ...] = ()


@dataclass
class RiskState:
    day: str = ""
    realized_pnl: float = 0.0
    trades_today: int = 0
    open_contracts: dict[str, float] = field(default_factory=dict)  # order_id -> stake
    trade_times: list[float] = field(default_factory=list)
    consecutive_errors: int = 0
    consecutive_losses: int = 0
    paused_until: float = 0.0
    pause_reason: str = ""
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.limits = limits or RiskLimits()
        self._clock = clock
        self._lock = threading.RLock()
        self.state = RiskState()

    # ----------------------------------------------------------------- helpers
    def _roll_day(self, now: float) -> None:
        day = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
        if day != self.state.day:
            self.state.day = day
            self.state.realized_pnl = 0.0
            self.state.trades_today = 0

    def update_limits(self, limits: RiskLimits) -> None:
        with self._lock:
            self.limits = limits

    # ------------------------------------------------------------ comprobación
    def check(self, order: OrderRequest, balance: float) -> RiskDecision:
        now = self._clock()
        lim = self.limits
        reasons: list[str] = []
        with self._lock:
            self._roll_day(now)
            st = self.state
            if st.halted:
                reasons.append(f"parada de emergencia activa: {st.halt_reason}")
            if now < st.paused_until:
                reasons.append(f"en pausa {st.paused_until - now:.0f}s: {st.pause_reason}")
            if order.spec.stake > lim.max_stake:
                reasons.append(f"stake {order.spec.stake} > máximo {lim.max_stake}")
            if st.realized_pnl <= -lim.max_daily_loss:
                reasons.append(f"pérdida diaria {st.realized_pnl:.2f} alcanzó el límite")
            # Pérdida potencial: lo perdido hoy + stakes abiertos + este stake.
            exposure = -st.realized_pnl + sum(st.open_contracts.values()) + order.spec.stake
            if exposure > lim.max_daily_loss + 1e-9:
                reasons.append(f"exposición potencial {exposure:.2f} superaría la pérdida "
                               f"diaria máxima {lim.max_daily_loss}")
            if len(st.open_contracts) >= lim.max_open_contracts:
                reasons.append(f"{len(st.open_contracts)} contratos abiertos (máx "
                               f"{lim.max_open_contracts})")
            recent = [t for t in st.trade_times if now - t < 3600]
            if len(recent) >= lim.max_trades_per_hour:
                reasons.append("límite de operaciones por hora alcanzado")
            if st.trade_times and now - st.trade_times[-1] < lim.min_seconds_between_trades:
                reasons.append("frecuencia mínima entre operaciones no cumplida")
            if balance - order.spec.stake < lim.min_balance:
                reasons.append(f"saldo {balance:.2f} insuficiente (mínimo {lim.min_balance})")
        return RiskDecision(not reasons, tuple(reasons))

    def can_open_more(self) -> tuple[bool, str]:
        """Comprobación barata previa a generar órdenes (no sustituye a ``check``)."""
        now = self._clock()
        with self._lock:
            st = self.state
            if st.halted:
                return False, "parada de emergencia"
            if now < st.paused_until:
                return False, f"pausa: {st.pause_reason}"
            if len(st.open_contracts) >= self.limits.max_open_contracts:
                return False, "máximo de contratos abiertos"
        return True, ""

    def check_proposal(self, stake: float, ask_price: float, payout: float) -> RiskDecision:
        lim = self.limits
        reasons = []
        if ask_price > stake * (1 + lim.max_price_slippage) + 1e-9:
            reasons.append(f"ask_price {ask_price} excede stake {stake} + tolerancia")
        if ask_price > lim.max_stake + 1e-9:
            reasons.append(f"ask_price {ask_price} > stake máximo {lim.max_stake}")
        ratio = payout / ask_price if ask_price > 0 else 0.0
        if ratio < lim.min_payout_ratio:
            reasons.append(f"payout ratio {ratio:.3f} < mínimo {lim.min_payout_ratio}")
        return RiskDecision(not reasons, tuple(reasons))

    # ---------------------------------------------------------------- registro
    def record_open(self, order_id: str, stake: float) -> None:
        now = self._clock()
        with self._lock:
            self._roll_day(now)
            self.state.open_contracts[order_id] = stake
            self.state.trade_times = [t for t in self.state.trade_times if now - t < 3600]
            self.state.trade_times.append(now)
            self.state.trades_today += 1

    def record_settlement(self, order_id: str, profit: float) -> None:
        now = self._clock()
        with self._lock:
            self._roll_day(now)
            self.state.open_contracts.pop(order_id, None)
            self.state.realized_pnl += profit
            if profit < 0:
                self.state.consecutive_losses += 1
                if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
                    self._pause(self.limits.loss_streak_pause_s,
                                f"{self.state.consecutive_losses} pérdidas seguidas")
            else:
                self.state.consecutive_losses = 0

    def release(self, order_id: str) -> None:
        """La compra fue rechazada por el broker: no hay contrato ni exposición."""
        with self._lock:
            self.state.open_contracts.pop(order_id, None)

    def record_error(self, reason: str = "") -> None:
        with self._lock:
            self.state.consecutive_errors += 1
            if self.state.consecutive_errors >= self.limits.max_consecutive_errors:
                self._pause(self.limits.error_pause_s,
                            f"{self.state.consecutive_errors} errores seguidos: {reason}")

    def record_success(self) -> None:
        with self._lock:
            self.state.consecutive_errors = 0

    def _pause(self, seconds: float, reason: str) -> None:
        self.state.paused_until = self._clock() + seconds
        self.state.pause_reason = reason

    def halt(self, reason: str) -> None:
        with self._lock:
            self.state.halted = True
            self.state.halt_reason = reason

    def resume(self) -> None:
        """Rearme manual tras parada de emergencia (nunca automático)."""
        with self._lock:
            self.state.halted = False
            self.state.halt_reason = ""
            self.state.paused_until = 0.0
            self.state.consecutive_errors = 0

    def snapshot(self) -> dict:
        with self._lock:
            self._roll_day(self._clock())
            st = self.state
            return {"day": st.day, "realized_pnl": round(st.realized_pnl, 2),
                    "trades_today": st.trades_today, "open_contracts": len(st.open_contracts),
                    "consecutive_errors": st.consecutive_errors,
                    "consecutive_losses": st.consecutive_losses,
                    "paused_for_s": max(0.0, st.paused_until - self._clock()),
                    "pause_reason": st.pause_reason, "halted": st.halted,
                    "halt_reason": st.halt_reason}
