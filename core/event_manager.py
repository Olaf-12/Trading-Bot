"""Bus de eventos pub/sub thread-safe. La GUI se suscribe a través de un puente Qt."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable

log = logging.getLogger(__name__)

Handler = Callable[[str, dict[str, Any]], None]

# Temas publicados
TICK = "tick"
SIGNAL = "signal"
PREDICTION = "prediction"
ORDER_REJECTED = "order_rejected"
ORDER_PLACED = "order_placed"
CONTRACT_UPDATE = "contract_update"
CONTRACT_SETTLED = "contract_settled"
BALANCE = "balance"
CONNECTION = "connection"
DATA_ISSUE = "data_issue"
RISK = "risk"
ENGINE_STATE = "engine_state"
EMERGENCY_STOP = "emergency_stop"
RECONCILIATION = "reconciliation"
LOG = "log"
ALL = "*"


class EventManager:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, handler: Handler) -> None:
        with self._lock:
            self._handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        with self._lock:
            if handler in self._handlers.get(topic, []):
                self._handlers[topic].remove(handler)

    def publish(self, topic: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        with self._lock:
            handlers = list(self._handlers.get(topic, [])) + list(self._handlers.get(ALL, []))
        for h in handlers:
            try:
                h(topic, payload)
            except Exception:
                log.exception("Handler de evento '%s' falló", topic)
