"""Reconciliación local <-> broker (Fase 7). El broker es la fuente de verdad.

Detecta y corrige:
- contratos liquidados en el broker sin liquidar en local (stream perdido);
- órdenes ``unknown`` (buy sin confirmación): se emparejan con contratos remotos
  o, pasado un margen, se marcan como no ejecutadas y se libera la exposición;
- diferencias de profit entre local y broker (se corrige al valor del broker);
- contratos remotos desconocidos (compras fuera de este bot) -> parada de
  emergencia por defecto, porque invalidan los límites de riesgo.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from broker.broker_interface import ContractUpdate, ExecutionPort, RemoteContract
from core import event_manager as ev
from core.execution_controller import ExecutionController
from database.trades import TradeRepository

log = logging.getLogger(__name__)


@dataclass
class ReconciliationReport:
    ts: float
    local_open: int = 0
    remote_open: int = 0
    settled_missed: list[int] = field(default_factory=list)
    unknown_resolved: list[str] = field(default_factory=list)
    unknown_not_executed: list[str] = field(default_factory=list)
    profit_corrected: list[int] = field(default_factory=list)
    unknown_remote: list[int] = field(default_factory=list)
    missing_remote: list[int] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not (self.unknown_remote or self.missing_remote or self.error)


class Reconciler:
    def __init__(self, port: ExecutionPort, repo: TradeRepository,
                 controller: ExecutionController, *, halt_on_unknown_remote: bool = True,
                 unknown_grace_s: float = 120.0, lookback_s: float = 86_400.0,
                 clock: Callable[[], float] = time.time) -> None:
        self.port = port
        self.repo = repo
        self.controller = controller
        self.halt_on_unknown_remote = halt_on_unknown_remote
        self.unknown_grace_s = unknown_grace_s
        self.lookback_s = lookback_s
        self._clock = clock
        self._lock = threading.Lock()
        self.last_report: ReconciliationReport | None = None

    def run_safe(self) -> ReconciliationReport | None:
        if not self._lock.acquire(blocking=False):
            return None                       # ya hay una reconciliación en curso
        try:
            with self.controller.exclusive():        # sin compras en curso
                return self._run()
        except Exception as exc:
            log.exception("Reconciliación falló")
            rep = ReconciliationReport(self._clock(), error=f"{type(exc).__name__}: {exc}")
            self._finish(rep)
            return rep
        finally:
            self._lock.release()

    run = run_safe

    # ------------------------------------------------------------------------
    def _run(self) -> ReconciliationReport:
        now = self._clock()
        rep = ReconciliationReport(now)
        remote_open = {c.contract_id: c for c in self.port.open_contracts()}
        since = int(now - self.lookback_s)
        remote_settled = {c.contract_id: c for c in self.port.settled_contracts(since, 500)}
        rep.remote_open = len(remote_open)
        local_orders = self.repo.open_orders()
        rep.local_open = len(local_orders)
        known_ids = {r["contract_id"] for r in self.repo.db.query(
            "SELECT contract_id FROM orders WHERE contract_id IS NOT NULL")}

        # 1) Órdenes unknown: emparejar con contratos remotos no conocidos.
        unmatched_remote = {cid: c for cid, c in {**remote_settled, **remote_open}.items()
                            if cid not in known_ids}
        for o in [o for o in local_orders if o["contract_id"] is None]:
            match = self._match(o, unmatched_remote.values())
            if match is not None:
                unmatched_remote.pop(match.contract_id)
                self.repo.attach_contract(o["client_order_id"], match.contract_id,
                                          match.transaction_id, match.buy_price, match.payout,
                                          match.purchase_time)
                known_ids.add(match.contract_id)
                rep.unknown_resolved.append(o["client_order_id"])
                log.warning("Orden %s resuelta: contrato %s", o["client_order_id"][:8],
                            match.contract_id)
                if match.contract_id in remote_open:
                    self.controller.track(o["client_order_id"], match.contract_id,
                                          match.buy_price or o["stake"], o["model_version"])
            elif now - o["created_at"] > self.unknown_grace_s and o["status"] in ("unknown",
                                                                                  "buying"):
                self.repo.set_status(o["client_order_id"], "failed",
                                     error="no ejecutada (verificado por reconciliación)")
                self.controller.risk.release(o["client_order_id"])
                rep.unknown_not_executed.append(o["client_order_id"])

        # 2) Contratos locales abiertos: ¿liquidados en el broker?
        for o in self.repo.open_orders():
            cid = o["contract_id"]
            if cid is None:
                continue
            if cid in remote_settled:
                rc = remote_settled[cid]
                if not self.controller.is_settled(cid):
                    profit = rc.profit if rc.profit is not None else -(rc.buy_price or 0)
                    self._store_remote_settlement(o, rc, profit)
                    self.controller.settle(o["client_order_id"], cid, profit,
                                           "won" if profit > 0 else "lost")
                    rep.settled_missed.append(cid)
                else:
                    self.repo.set_status(o["client_order_id"], "settled")
            elif cid in remote_open:
                self.controller.track(o["client_order_id"], cid, o["buy_price"] or o["stake"],
                                      o["model_version"])
            elif now - (o["purchase_time"] or o["created_at"]) > self.lookback_s:
                pass                                   # fuera de la ventana consultada
            else:
                rep.missing_remote.append(cid)

        # 3) Profit local vs broker en liquidados.
        for cid, rc in remote_settled.items():
            t = self.repo.trade(cid)
            if t is None or t["profit"] is None or rc.profit is None:
                continue
            if abs(t["profit"] - rc.profit) > 0.01:
                self.repo.db.execute("UPDATE trades SET profit = ?, sell_price = ? WHERE "
                                     "contract_id = ?", (rc.profit, rc.sell_price, cid))
                rep.profit_corrected.append(cid)

        # 4) Contratos remotos que este bot no compró.
        rep.unknown_remote = sorted(unmatched_remote)
        if rep.unknown_remote and self.halt_on_unknown_remote:
            self.controller.emergency_stop(
                f"reconciliación: {len(rep.unknown_remote)} contrato(s) en la cuenta no "
                f"comprados por este bot: {rep.unknown_remote[:5]}")
        self._finish(rep)
        return rep

    @staticmethod
    def _match(order, candidates) -> RemoteContract | None:
        best = None
        for c in candidates:
            if c.underlying_symbol != order["symbol"] or c.contract_type != order["contract_type"]:
                continue
            if c.buy_price is not None and abs(c.buy_price - (order["proposal_ask"] or
                                                              order["stake"])) > 0.011:
                continue
            if c.purchase_time is not None and c.purchase_time < order["created_at"] - 5:
                continue
            if best is None or (c.purchase_time or 0) < (best.purchase_time or 0):
                best = c
        return best

    def _store_remote_settlement(self, o, rc: RemoteContract, profit: float) -> None:
        u = ContractUpdate(contract_id=rc.contract_id, status="won" if profit > 0 else "lost",
                           is_sold=True, is_expired=True, buy_price=rc.buy_price,
                           payout=rc.payout, profit=profit, sell_price=rc.sell_price,
                           entry_spot=None, entry_spot_time=None, exit_spot=None,
                           exit_spot_time=None, date_expiry=None, sell_time=rc.sell_time,
                           underlying_symbol=rc.underlying_symbol,
                           contract_type=rc.contract_type, account_id=self.port.account_id,
                           currency=None)
        self.repo.upsert_trade(o["client_order_id"], u, o["model_version"])

    def _finish(self, rep: ReconciliationReport) -> None:
        self.last_report = rep
        payload = {**asdict(rep), "ok": rep.ok}
        self.repo.db.execute("INSERT INTO reconciliation_runs(ts, ok, report) VALUES (?,?,?)",
                             (rep.ts, int(rep.ok), json.dumps(payload)))
        self.controller.events.publish(ev.RECONCILIATION, payload)
        level = logging.INFO if rep.ok else logging.WARNING
        log.log(level, "Reconciliación ok=%s %s", rep.ok,
                {k: v for k, v in payload.items() if v and k not in ("ts", "ok")})
