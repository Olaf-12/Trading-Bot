"""Datos de mercado de Deriv: ``active_symbols`` y streams de ``ticks``.

Campos verificados contra deriv-com/deriv-api-schemas:
- active_symbols: request ``{"active_symbols": "brief"|"full"}``; cada símbolo trae
  ``underlying_symbol``, ``underlying_symbol_name``, ``market``, ``submarket``,
  ``pip_size``, ``exchange_is_open`` (0/1), ``is_trading_suspended`` (0/1).
- ticks: request ``{"ticks": <symbol>, "subscribe": 1}``; respuesta ``msg_type: "tick"``
  con ``tick.symbol``, ``tick.epoch``, ``tick.quote``, ``tick.pip_size`` y
  ``subscription.id`` para ``forget``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from broker.broker_interface import IssueCallback, TickCallback
from broker.deriv_ws_client import DerivWSClient
from config import TickValidationSettings
from market.market_data import (SymbolInfo, Tick, TickIssue, TickValidator,
                                TickValidationResult)

log = logging.getLogger(__name__)


class UnknownSymbolError(ValueError):
    pass


def parse_active_symbols(message: dict[str, Any]) -> list[SymbolInfo]:
    out: list[SymbolInfo] = []
    for raw in message.get("active_symbols") or []:
        try:
            out.append(SymbolInfo(
                underlying_symbol=str(raw["underlying_symbol"]),
                underlying_symbol_name=str(raw.get("underlying_symbol_name", "")),
                market=str(raw.get("market", "")),
                submarket=str(raw.get("submarket", "")),
                pip_size=float(raw.get("pip_size") or 0),
                exchange_is_open=bool(raw.get("exchange_is_open")),
                is_trading_suspended=bool(raw.get("is_trading_suspended")),
            ))
        except (KeyError, TypeError, ValueError):
            log.warning("Símbolo con formato inesperado descartado")
    return out


class DerivMarketData:
    def __init__(self, client: DerivWSClient,
                 settings: TickValidationSettings | None = None,
                 on_issue: IssueCallback | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.client = client
        self.settings = settings or TickValidationSettings()
        self._on_issue = on_issue
        self._clock = clock
        self._symbols: dict[str, SymbolInfo] = {}
        self._validators: dict[str, TickValidator] = {}
        self._subscribed_at: dict[str, float] = {}
        self._lock = threading.Lock()
        self._last_stale_reconnect = 0.0
        client.add_reconnect_hook(self._on_reconnect)

    # ---------------------------------------------------------- active_symbols
    def get_active_symbols(self, refresh: bool = False) -> list[SymbolInfo]:
        if refresh or not self._symbols:
            msg = self.client.request({"active_symbols": "brief"})
            symbols = parse_active_symbols(msg)
            if not symbols:
                raise RuntimeError("active_symbols devolvió una lista vacía")
            with self._lock:
                self._symbols = {s.underlying_symbol: s for s in symbols}
        return list(self._symbols.values())

    def symbol_info(self, symbol: str) -> SymbolInfo:
        if not self._symbols:
            self.get_active_symbols()
        try:
            return self._symbols[symbol]
        except KeyError:
            raise UnknownSymbolError(
                f"'{symbol}' no figura en active_symbols para esta conexión") from None

    # --------------------------------------------------------- ticks_history
    def get_ticks_history(self, symbol: str, count: int = 5000,
                          end: str | int = "latest") -> list[Tick]:
        """Histórico de ticks (``ticks_history`` style=ticks -> history.prices/times).
        Deriv limita ``count`` por petición; para más datos, paginar con ``end``."""
        self.symbol_info(symbol)
        msg = self.client.request({"ticks_history": symbol, "end": str(end),
                                   "count": int(count), "style": "ticks"},
                                  timeout=max(self.client.settings.request_timeout_s, 30))
        hist = msg.get("history") or {}
        prices, times = hist.get("prices") or [], hist.get("times") or []
        if len(prices) != len(times):
            raise ValueError("ticks_history: prices y times de distinta longitud")
        pip = msg.get("pip_size")
        now = self._clock()
        out: list[Tick] = []
        last_epoch = -1
        for price, epoch in zip(prices, times):
            if not isinstance(epoch, int) or epoch <= last_epoch:
                continue                      # duplicados / desorden
            if not isinstance(price, (int, float)) or price <= 0:
                continue
            out.append(Tick(symbol, epoch, float(price),
                            float(pip) if isinstance(pip, (int, float)) else None, now))
            last_epoch = epoch
        return out

    # ------------------------------------------------------------------- ticks
    @staticmethod
    def _key(symbol: str) -> str:
        return f"ticks:{symbol}"

    def subscribe_ticks(self, symbol: str, on_tick: TickCallback) -> None:
        info = self.symbol_info(symbol)            # nunca asumir disponibilidad
        if not info.exchange_is_open:
            log.warning("%s: mercado cerrado según active_symbols", symbol)
        validator = TickValidator(symbol, self.settings, clock=self._clock)

        def handle(message: dict[str, Any]) -> None:
            if message.get("msg_type") != "tick":
                return
            result = validator.validate(message.get("tick") or {})
            if result.accepted and result.tick is not None:
                on_tick(result.tick)
            else:
                self._report(symbol, result)

        with self._lock:
            self._validators[symbol] = validator
            self._subscribed_at[symbol] = self._clock()
        try:
            self.client.subscribe(self._key(symbol), {"ticks": symbol}, handle)
        except BaseException:
            with self._lock:
                self._validators.pop(symbol, None)
                self._subscribed_at.pop(symbol, None)
            raise

    def unsubscribe_ticks(self, symbol: str) -> None:
        self.client.unsubscribe(self._key(symbol))
        with self._lock:
            self._validators.pop(symbol, None)
            self._subscribed_at.pop(symbol, None)

    def _report(self, symbol: str, result: TickValidationResult) -> None:
        log.warning("Tick rechazado %s: %s %s", symbol,
                    result.issue.value if result.issue else "?", result.detail)
        if self._on_issue:
            try:
                self._on_issue(symbol, result)
            except Exception:
                log.exception("on_issue falló")

    # --------------------------------------------------------------- frescura
    def stale_ages(self) -> dict[str, float]:
        """Símbolos suscritos sin ticks recientes -> segundos sin datos.

        Incluye streams que nunca recibieron un tick desde la (re)suscripción.
        """
        now = self._clock()
        out: dict[str, float] = {}
        with self._lock:
            for symbol, validator in self._validators.items():
                last = validator.last_received_at or self._subscribed_at.get(symbol, now)
                if now - last > self.settings.stale_after_s:
                    out[symbol] = now - last
        return out

    def check_health(self) -> list[str]:
        """Llamar periódicamente (scheduler del TradingEngine o timer de Qt).

        Notifica streams obsoletos y fuerza reconexión si superan el umbral.
        """
        ages = self.stale_ages()
        stale = list(ages)
        now = self._clock()
        for symbol in stale:
            self._report(symbol, TickValidationResult(
                None, False, TickIssue.STALE_QUOTE, f"stream sin ticks {ages[symbol]:.1f}s"))
        if stale and self.client.is_connected:
            worst = max(ages.values())
            if (worst > self.settings.reconnect_after_stale_s
                    and now - self._last_stale_reconnect > self.settings.reconnect_after_stale_s):
                self._last_stale_reconnect = now
                self.client.force_reconnect(f"ticks obsoletos: {', '.join(stale)}")
        return stale

    def _on_reconnect(self) -> None:
        now = self._clock()
        with self._lock:
            for symbol, validator in self._validators.items():
                validator.reset()
                self._subscribed_at[symbol] = now
