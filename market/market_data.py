"""Modelos de datos de mercado y validación de ticks (independiente del broker).

Controles del PDF §7 «Datos»: timestamps, duplicados, desconexiones (frescura),
saltos de precio y cotizaciones obsoletas.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from config import TickValidationSettings


@dataclass(frozen=True)
class SymbolInfo:
    """Símbolo según ``active_symbols`` (campos del esquema oficial)."""

    underlying_symbol: str
    underlying_symbol_name: str
    market: str
    submarket: str
    pip_size: float
    exchange_is_open: bool
    is_trading_suspended: bool

    @property
    def is_tradeable(self) -> bool:
        return self.exchange_is_open and not self.is_trading_suspended


@dataclass(frozen=True)
class Tick:
    symbol: str
    epoch: int
    quote: float
    pip_size: float | None
    received_at: float


class TickIssue(str, Enum):
    MALFORMED = "malformed"
    SYMBOL_MISMATCH = "symbol_mismatch"
    INVALID_PRICE = "invalid_price"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"
    FUTURE_TIMESTAMP = "future_timestamp"
    STALE_QUOTE = "stale_quote"
    PRICE_JUMP = "price_jump"


@dataclass(frozen=True)
class TickValidationResult:
    tick: Tick | None
    accepted: bool
    issue: TickIssue | None = None
    detail: str = ""


def parse_tick(payload: dict[str, Any], received_at: float | None = None) -> Tick:
    """Convierte el objeto ``tick`` de la API en ``Tick``. Lanza ValueError si es inválido."""
    try:
        symbol = payload["symbol"]
        epoch = payload["epoch"]
        quote = payload["quote"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"tick sin campo obligatorio: {exc}") from exc
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("symbol inválido")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError("epoch no es entero")
    if isinstance(quote, bool) or not isinstance(quote, (int, float)):
        raise ValueError("quote no es numérico")
    pip = payload.get("pip_size")
    return Tick(symbol=symbol, epoch=epoch, quote=float(quote),
                pip_size=float(pip) if isinstance(pip, (int, float)) else None,
                received_at=received_at if received_at is not None else time.time())


class TickValidator:
    """Valida un flujo de ticks de UN símbolo. No es thread-safe: un validador por stream."""

    def __init__(self, symbol: str, settings: TickValidationSettings | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.symbol = symbol
        self.settings = settings or TickValidationSettings()
        self._clock = clock
        self.last_tick: Tick | None = None
        self.last_received_at: float | None = None

    def reset(self) -> None:
        """Tras reconexión: se conserva el último tick para detectar duplicados/huecos."""
        self.last_received_at = None

    def validate(self, payload: dict[str, Any]) -> TickValidationResult:
        now = self._clock()
        try:
            tick = parse_tick(payload, received_at=now)
        except ValueError as exc:
            return TickValidationResult(None, False, TickIssue.MALFORMED, str(exc))

        if tick.symbol != self.symbol:
            return TickValidationResult(tick, False, TickIssue.SYMBOL_MISMATCH,
                                        f"esperado {self.symbol}, recibido {tick.symbol}")
        if not math.isfinite(tick.quote) or tick.quote <= 0:
            return TickValidationResult(tick, False, TickIssue.INVALID_PRICE,
                                        f"quote={tick.quote}")
        if tick.epoch > now + self.settings.max_future_skew_s:
            return TickValidationResult(tick, False, TickIssue.FUTURE_TIMESTAMP,
                                        f"epoch={tick.epoch} now={now:.0f}")
        if now - tick.epoch > self.settings.stale_after_s:
            return TickValidationResult(tick, False, TickIssue.STALE_QUOTE,
                                        f"edad={now - tick.epoch:.1f}s")

        last = self.last_tick
        if last is not None:
            if tick.epoch == last.epoch:
                return TickValidationResult(tick, False, TickIssue.DUPLICATE,
                                            f"epoch={tick.epoch}")
            if tick.epoch < last.epoch:
                return TickValidationResult(tick, False, TickIssue.OUT_OF_ORDER,
                                            f"epoch={tick.epoch} < {last.epoch}")
            jump = abs(tick.quote - last.quote) / last.quote
            if jump > self.settings.max_relative_jump:
                # Se acepta como nuevo último precio para no bloquear el stream,
                # pero se marca para que estrategia/riesgo puedan pausar.
                self.last_tick, self.last_received_at = tick, now
                return TickValidationResult(tick, False, TickIssue.PRICE_JUMP,
                                            f"salto={jump:.2%}")

        self.last_tick, self.last_received_at = tick, now
        return TickValidationResult(tick, True)

    def seconds_since_last_tick(self) -> float | None:
        if self.last_received_at is None:
            return None
        return self._clock() - self.last_received_at

    def is_stale(self) -> bool:
        age = self.seconds_since_last_tick()
        return age is not None and age > self.settings.stale_after_s
