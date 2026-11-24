"""ExecutionController: ÚNICO punto autorizado para aprobar o rechazar una orden.

Flujo (pasos 7–11 del PDF), serializado con un lock:
  1. Parada de emergencia / entorno Demo / moneda.
  2. Revalida la cuenta (``ensure_account_valid``: loginid y moneda).
  3. RiskManager.check (importe, pérdida diaria, exposición, simultáneos, frecuencia, pausas).
  4. proposal -> validación de precio, payout y frescura.
  5. Emite ``OrderApproval`` de un solo uso -> buy(price = ask aprobado).
  6. Suscribe proposal_open_contract y registra la liquidación (trades, riesgo, eventos).

Reglas de seguridad:
- ``buy`` nunca se reintenta. Un timeout o desconexión deja la orden en ``unknown``:
  la exposición sigue contando y la reconciliación (Fase 7) la resuelve.
- Si un contrato llega con un ``account_id`` distinto, o la cuenta no coincide, se
  activa la parada de emergencia.
- La parada de emergencia bloquea nuevas compras, mantiene el seguimiento de los
  contratos abiertos y solo se rearma manualmente.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from broker.broker_interface import (BrokerRejection, ContractUpdate, ExecutionPort,
                                     Proposal)
from config import AccountEnvironment, Settings
from core import event_manager as ev
from core.event_manager import EventManager
from core.order_approval import _MINT_KEY, OrderApproval
from database.trades import TradeRepository
from risk.risk_manager import RiskManager
from strategy.signals import OrderRequest

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionResult:
    client_order_id: str
    status: str                       # rejected | open | unknown | failed
    reasons: tuple[str, ...] = ()
    contract_id: int | None = None
    proposal: Proposal | None = None


@dataclass
class _Tracked:
    client_order_id: str
    stake: float
    model_version: str | None
    settled: bool = False
    updates: list[ContractUpdate] = field(default_factory=list)


class ExecutionController:
    PROPOSAL_MAX_AGE_S = 15.0

    def __init__(self, settings: Settings, port: ExecutionPort, risk: RiskManager, *,
                 repo: TradeRepository | None = None, events: EventManager | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.settings = settings
        self.port = port
        self.risk = risk
        self.repo = repo
        self.events = events or EventManager()
        self._clock = clock
        self._submit_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._halted = False
        self._halt_reason = ""
        self._tracked: dict[int, _Tracked] = {}            # contract_id -> seguimiento
        self.on_unknown_order: Callable[[str], None] | None = None

    # ------------------------------------------------------------------ estado
    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def emergency_stop(self, reason: str) -> None:
        with self._state_lock:
            if self._halted:
                return
            self._halted = True
            self._halt_reason = reason
        self.risk.halt(reason)
        log.error("PARADA DE EMERGENCIA: %s", reason)
        self.events.publish(ev.EMERGENCY_STOP, {"reason": reason,
                                                "open_contracts": self.open_contract_ids()})

    def resume(self) -> None:
        """Rearme manual. Nunca se llama automáticamente."""
        with self._state_lock:
            self._halted = False
            self._halt_reason = ""
        self.risk.resume()
        self.events.publish(ev.RISK, {"resumed": True, **self.risk.snapshot()})

    def exclusive(self):
        """Context manager: sin compras en curso (lo usa la reconciliación)."""
        return self._submit_lock

    def open_contract_ids(self) -> list[int]:
        with self._state_lock:
            return [cid for cid, t in self._tracked.items() if not t.settled]

    # ----------------------------------------------------------------- órdenes
    def _reject(self, order: OrderRequest, reasons: tuple[str, ...],
                account_id: str | None = None, status: str = "rejected") -> ExecutionResult:
        log.info("Orden %s %s: %s", order.client_order_id[:8], status, "; ".join(reasons))
        if self.repo:
            if self.repo.order(order.client_order_id) is None:
                self.repo.create_order(order, account_id, status, reasons)
            else:
                self.repo.set_status(order.client_order_id, status, reasons=reasons)
        self.events.publish(ev.ORDER_REJECTED, {"client_order_id": order.client_order_id,
                                                "status": status, "reasons": list(reasons),
                                                "symbol": order.spec.symbol})
        return ExecutionResult(order.client_order_id, status, reasons)

    def submit(self, order: OrderRequest) -> ExecutionResult:
        with self._submit_lock:
            return self._submit(order)

    def _submit(self, order: OrderRequest) -> ExecutionResult:
        spec = order.spec
        # 1. Estado y entorno ------------------------------------------------
        if self._halted:
            return self._reject(order, (f"parada de emergencia: {self._halt_reason}",))
        if self.settings.environment is not AccountEnvironment.DEMO:
            return self._reject(order, ("solo se permite operar en Demo",))

        # 2. Cuenta ---------------------------------------------------------
        try:
            balance = self.port.ensure_account_valid()
            account_id = self.port.account_id
        except PermissionError as exc:
            self.emergency_stop(f"validación de cuenta: {exc}")
            return self._reject(order, (f"cuenta inválida: {exc}",))
        except Exception as exc:
            self.risk.record_error(f"balance: {exc}")
            return self._reject(order, (f"no se pudo validar la cuenta: {exc}",))
        if balance.loginid != account_id:
            self.emergency_stop(f"loginid {balance.loginid} != cuenta {account_id}")
            return self._reject(order, ("discrepancia de cuenta",), account_id)
        if spec.currency != balance.currency:
            return self._reject(order, (f"moneda {spec.currency} != cuenta "
                                        f"{balance.currency}",), account_id)

        # 3. Riesgo ---------------------------------------------------------
        decision = self.risk.check(order, balance.balance)
        if not decision.approved:
            return self._reject(order, decision.reasons, account_id)

        if self.repo:
            self.repo.create_order(order, account_id, "proposing")

        # 4. Propuesta ------------------------------------------------------
        try:
            proposal = self.port.get_proposal(spec)
        except Exception as exc:
            self.risk.record_error(f"proposal: {exc}")
            return self._reject(order, (f"proposal falló: {exc}",), account_id, "failed")
        if self.repo:
            self.repo.record_proposal(order.client_order_id, proposal)
        reasons = list(self.risk.check_proposal(spec.stake, proposal.ask_price,
                                                proposal.payout).reasons)
        age = self._clock() - proposal.spot_time
        if age > self.PROPOSAL_MAX_AGE_S:
            reasons.append(f"propuesta obsoleta ({age:.0f}s)")
        if reasons:
            return self._reject(order, tuple(reasons), account_id)

        # 5. Aprobación + compra -------------------------------------------
        approval = OrderApproval(key=_MINT_KEY, client_order_id=order.client_order_id,
                                 proposal_id=proposal.id, max_price=proposal.ask_price,
                                 account_id=account_id, clock=self._clock)
        self.risk.record_open(order.client_order_id, proposal.ask_price)
        if self.repo:
            self.repo.set_status(order.client_order_id, "buying")
        try:
            receipt = self.port.buy(proposal, approval)
        except BrokerRejection as exc:
            # Rechazo explícito: no se compró nada.
            self.risk.release(order.client_order_id)
            self.risk.record_error(f"buy rechazado: {exc}")
            return self._reject(order, (f"buy rechazado: {exc}",), account_id, "failed")
        except Exception as exc:
            # Timeout/desconexión: estado desconocido. NO reintentar.
            self.risk.record_error(f"buy sin confirmación: {exc}")
            if self.repo:
                self.repo.set_status(order.client_order_id, "unknown",
                                     error=f"{type(exc).__name__}: {exc}")
            log.error("Compra %s sin confirmación (%s): requiere reconciliación",
                      order.client_order_id[:8], type(exc).__name__)
            self.events.publish(ev.ORDER_REJECTED, {
                "client_order_id": order.client_order_id, "status": "unknown",
                "reasons": [f"sin confirmación: {exc}"], "symbol": spec.symbol})
            if self.on_unknown_order:
                self.on_unknown_order(order.client_order_id)
            return ExecutionResult(order.client_order_id, "unknown",
                                   (f"sin confirmación: {exc}",), proposal=proposal)

        self.risk.record_success()
        if self.repo:
            self.repo.record_buy(order.client_order_id, receipt)
        self.events.publish(ev.ORDER_PLACED, {
            "client_order_id": order.client_order_id, "contract_id": receipt.contract_id,
            "symbol": spec.symbol, "contract_type": spec.contract_type,
            "buy_price": receipt.buy_price, "payout": receipt.payout,
            "balance_after": receipt.balance_after, "longcode": receipt.longcode})

        # 6. Seguimiento ----------------------------------------------------
        model_version = order.signal.model_version if order.signal else None
        self.track(order.client_order_id, receipt.contract_id, receipt.buy_price,
                   model_version)
        return ExecutionResult(order.client_order_id, "open", (), receipt.contract_id,
                               proposal)

    # ------------------------------------------------------------ seguimiento
    def track(self, client_order_id: str, contract_id: int, stake: float,
              model_version: str | None) -> None:
        with self._state_lock:
            if contract_id in self._tracked:
                return
            self._tracked[contract_id] = _Tracked(client_order_id, stake, model_version)
        try:
            self.port.track_contract(contract_id, self._on_contract_update)
        except Exception as exc:
            log.error("No se pudo seguir el contrato %s: %s (lo resolverá la reconciliación)",
                      contract_id, exc)

    def _on_contract_update(self, u: ContractUpdate) -> None:
        with self._state_lock:
            t = self._tracked.get(u.contract_id)
        if t is None:
            return
        expected = self.port.account_id
        if u.account_id and u.account_id != expected:
            self.emergency_stop(f"contrato {u.contract_id} de la cuenta {u.account_id}, "
                                f"esperada {expected}")
        t.updates.append(u)
        if self.repo:
            self.repo.upsert_trade(t.client_order_id, u, t.model_version)
        self.events.publish(ev.CONTRACT_UPDATE, {
            "contract_id": u.contract_id, "status": u.status, "profit": u.profit,
            "entry_spot": u.entry_spot, "exit_spot": u.exit_spot, "symbol": u.underlying_symbol})
        if u.is_settled:
            profit = u.profit
            if profit is None and u.sell_price is not None and u.buy_price is not None:
                profit = u.sell_price - u.buy_price
            self.settle(t.client_order_id, u.contract_id, profit if profit is not None
                        else -t.stake, u.status)

    def settle(self, client_order_id: str, contract_id: int, profit: float,
               status: str | None = None) -> bool:
        """Registra la liquidación una sola vez (también desde la reconciliación)."""
        with self._state_lock:
            t = self._tracked.get(contract_id)
            if t is None:
                t = self._tracked[contract_id] = _Tracked(client_order_id, 0.0, None)
            if t.settled:
                return False
            t.settled = True
        self.risk.record_settlement(client_order_id, profit)
        self.events.publish(ev.CONTRACT_SETTLED, {
            "client_order_id": client_order_id, "contract_id": contract_id,
            "profit": round(profit, 2), "status": status, **self.risk.snapshot()})
        log.info("Contrato %s liquidado: %s %.2f", contract_id, status, profit)
        return True

    def is_settled(self, contract_id: int) -> bool:
        with self._state_lock:
            t = self._tracked.get(contract_id)
            return bool(t and t.settled)
