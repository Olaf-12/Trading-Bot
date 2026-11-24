"""Estado observable de la sesión de trading (lo lee la GUI)."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SessionState:
    mode: str = "public"                 # public (solo datos) | demo | backtest
    environment: str = "demo"
    account_id: str | None = None
    currency: str | None = None
    connection: str = "disconnected"
    symbol: str | None = None
    auto_trading: bool = False
    status: str = "stopped"              # stopped | running | paused | halted
    halt_reason: str = ""
    balance: float | None = None
    last_quote: float | None = None
    last_epoch: int | None = None
    last_prediction: float | None = None
    model_version: str | None = None
    market_ok: bool = True
    market_issue: str = ""
    ticks_processed: int = 0
    signals: int = 0
    orders_open: int = 0
    started_at: float = field(default_factory=time.time)


class TradingSession:
    def __init__(self, **initial: Any) -> None:
        self._state = SessionState(**initial)
        self._lock = threading.Lock()

    def update(self, **changes: Any) -> None:
        with self._lock:
            for k, v in changes.items():
                if not hasattr(self._state, k):
                    raise AttributeError(k)
                setattr(self._state, k, v)

    def incr(self, name: str, by: int = 1) -> None:
        with self._lock:
            setattr(self._state, name, getattr(self._state, name) + by)

    def get(self, name: str) -> Any:
        with self._lock:
            return getattr(self._state, name)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self._state)
