"""Monitor de estabilidad para la sesión Demo prolongada (Fase 7).

Agrega métricas a partir del bus de eventos y genera informes periódicos:
uptime, reconexiones, ticks, anomalías de datos, órdenes, liquidaciones, P&L,
errores, pausas de riesgo y resultado de las reconciliaciones.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from typing import Any, Callable

from core import event_manager as ev
from core.event_manager import EventManager
from database.database import Database


class SoakMonitor:
    def __init__(self, events: EventManager, db: Database | None = None,
                 connection_count: Callable[[], int] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self.started = clock()
        self.db = db
        self._connection_count = connection_count
        self._lock = threading.Lock()
        self.counts: Counter[str] = Counter()
        self.data_issues: Counter[str] = Counter()
        self.reject_reasons: Counter[str] = Counter()
        self.pnl = 0.0
        self.recon_runs = 0
        self.recon_failures = 0
        self.last_tick_at: float | None = None
        self.max_tick_gap = 0.0
        events.subscribe(ev.ALL, self._on_event)

    def _on_event(self, topic: str, p: dict[str, Any]) -> None:
        now = self._clock()
        with self._lock:
            self.counts[topic] += 1
            if topic == ev.TICK:
                if self.last_tick_at is not None:
                    self.max_tick_gap = max(self.max_tick_gap, now - self.last_tick_at)
                self.last_tick_at = now
            elif topic == ev.DATA_ISSUE:
                self.data_issues[p.get("issue", "?")] += 1
            elif topic == ev.ORDER_REJECTED:
                self.reject_reasons[(p.get("reasons") or ["?"])[0][:50]] += 1
            elif topic == ev.CONTRACT_SETTLED:
                self.pnl += p.get("profit", 0.0)
            elif topic == ev.RECONCILIATION:
                self.recon_runs += 1
                self.recon_failures += 0 if p.get("ok") else 1

    def report(self) -> dict[str, Any]:
        with self._lock:
            uptime = self._clock() - self.started
            return {
                "uptime_h": round(uptime / 3600, 3),
                "reconnections": max(0, (self._connection_count() if self._connection_count
                                         else 1) - 1),
                "ticks": self.counts[ev.TICK],
                "max_tick_gap_s": round(self.max_tick_gap, 1),
                "data_issues": dict(self.data_issues),
                "predictions": self.counts[ev.PREDICTION],
                "orders_placed": self.counts[ev.ORDER_PLACED],
                "orders_rejected": self.counts[ev.ORDER_REJECTED],
                "top_reject_reasons": dict(self.reject_reasons.most_common(5)),
                "settled": self.counts[ev.CONTRACT_SETTLED],
                "pnl": round(self.pnl, 2),
                "emergency_stops": self.counts[ev.EMERGENCY_STOP],
                "reconciliations": self.recon_runs,
                "reconciliation_failures": self.recon_failures,
            }

    def save(self) -> dict[str, Any]:
        rep = self.report()
        if self.db:
            self.db.log_event("soak", "report", json.dumps(rep))
        return rep

    def passed(self, min_hours: float = 24.0) -> tuple[bool, list[str]]:
        """Criterio de salida orientativo de la Fase 7."""
        r = self.report()
        problems = []
        if r["uptime_h"] < min_hours:
            problems.append(f"duración {r['uptime_h']}h < {min_hours}h")
        if r["reconciliation_failures"]:
            problems.append(f"{r['reconciliation_failures']} reconciliaciones con discrepancias")
        if r["emergency_stops"]:
            problems.append(f"{r['emergency_stops']} paradas de emergencia")
        if r["reconciliations"] == 0:
            problems.append("no se ejecutó ninguna reconciliación")
        return not problems, problems
