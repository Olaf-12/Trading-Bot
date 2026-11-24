"""Ejecución de contratos Deriv: proposal -> buy -> proposal_open_contract (Fase 3).

Solo lo invoca ``core.execution_controller.ExecutionController``: ``buy`` exige
un ``OrderApproval`` de un solo uso ligado a la propuesta, al precio máximo y a
la cuenta. Campos según deriv-com/deriv-api-schemas:

- proposal (req): proposal=1, amount, basis, contract_type, currency, duration,
  duration_unit, underlying_symbol. (resp) proposal.{id, ask_price, payout, spot,
  spot_time, date_start, date_expiry?, longcode}.
- buy (req): buy=<proposal_id>, price=<máximo>. (resp) buy.{contract_id,
  transaction_id, buy_price, payout, balance_after, purchase_time, start_time, ...}.
- proposal_open_contract (req): proposal_open_contract=1, contract_id, subscribe=1.
  (resp) importes como STRING; incluye account_id, status, is_sold, entry/exit_spot.
- portfolio / profit_table para reconciliación.

``buy`` NUNCA se reintenta automáticamente: un timeout puede significar que la
compra sí se ejecutó. El controlador marca la orden como ``unknown`` y la
reconciliación (portfolio/profit_table) la resuelve.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from broker.broker_interface import (Balance, BuyReceipt, ContractCallback, ContractUpdate,
                                     ExecutionPort, Proposal, RemoteContract)
from core.order_approval import ApprovalError, OrderApproval
from strategy.signals import ContractSpec

if TYPE_CHECKING:
    from broker.deriv_adapter import DerivDemoAdapter

log = logging.getLogger(__name__)


def parse_proposal(message: dict[str, Any], spec: ContractSpec) -> Proposal:
    p = message.get("proposal")
    if not isinstance(p, dict):
        raise ValueError("Respuesta sin objeto 'proposal'")
    try:
        return Proposal(id=str(p["id"]), ask_price=float(p["ask_price"]),
                        payout=float(p["payout"]), spot=float(p["spot"]),
                        spot_time=int(p["spot_time"]), date_start=int(p["date_start"]),
                        date_expiry=int(p["date_expiry"]) if p.get("date_expiry") else None,
                        longcode=str(p.get("longcode", "")), spec=spec)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"proposal con formato inesperado: {exc}") from None


def parse_buy(message: dict[str, Any]) -> BuyReceipt:
    b = message.get("buy")
    if not isinstance(b, dict):
        raise ValueError("Respuesta sin objeto 'buy'")
    try:
        return BuyReceipt(contract_id=int(b["contract_id"]),
                          transaction_id=int(b["transaction_id"]),
                          buy_price=float(b["buy_price"]), payout=float(b["payout"]),
                          balance_after=float(b["balance_after"]),
                          purchase_time=int(b["purchase_time"]),
                          start_time=int(b["start_time"]),
                          longcode=str(b.get("longcode", "")),
                          shortcode=str(b.get("shortcode", "")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"buy con formato inesperado: {exc}") from None


def _f(v: Any) -> float | None:
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


class DerivExecution(ExecutionPort):
    def __init__(self, adapter: "DerivDemoAdapter") -> None:
        self.adapter = adapter
        self.client = adapter.client

    @property
    def account_id(self) -> str:
        if self.adapter.account is None:
            raise RuntimeError("Cuenta no seleccionada")
        return self.adapter.account.account_id

    def ensure_account_valid(self) -> Balance:
        return self.adapter.ensure_account_valid()

    def get_proposal(self, spec: ContractSpec) -> Proposal:
        self.adapter._ensure_not_locked()
        return parse_proposal(self.client.request(spec.to_proposal_request()), spec)

    def buy(self, proposal: Proposal, approval: OrderApproval) -> BuyReceipt:
        self.adapter._ensure_not_locked()
        if not isinstance(approval, OrderApproval):
            raise ApprovalError("Compra sin aprobación de ExecutionController")
        approval.consume(proposal_id=proposal.id, price=proposal.ask_price,
                         account_id=self.account_id)
        # price = precio máximo aceptado (ask de la propuesta aprobada).
        msg = self.client.request({"buy": proposal.id, "price": proposal.ask_price})
        return parse_buy(msg)

    def track_contract(self, contract_id: int, on_update: ContractCallback) -> None:
        key = f"poc:{contract_id}"

        def handle(message: dict[str, Any]) -> None:
            poc = message.get("proposal_open_contract")
            if not isinstance(poc, dict) or "contract_id" not in poc:
                return
            update = ContractUpdate.from_api(poc)
            try:
                on_update(update)
            finally:
                if update.is_settled:
                    # Desde el hilo receptor: baja sin esperar respuesta.
                    self.client.unsubscribe(key, wait=False)

        self.client.subscribe(key, {"proposal_open_contract": 1,
                                    "contract_id": int(contract_id)}, handle)

    def open_contracts(self) -> list[RemoteContract]:
        msg = self.client.request({"portfolio": 1})
        out = []
        for c in (msg.get("portfolio") or {}).get("contracts") or []:
            out.append(RemoteContract(
                contract_id=int(c["contract_id"]), transaction_id=c.get("transaction_id"),
                underlying_symbol=c.get("underlying_symbol"),
                contract_type=c.get("contract_type"), buy_price=_f(c.get("buy_price")),
                payout=_f(c.get("payout")), purchase_time=c.get("purchase_time")))
        return out

    def settled_contracts(self, since_epoch: int, limit: int = 100) -> list[RemoteContract]:
        msg = self.client.request({"profit_table": 1, "description": 1,
                                   "date_from": str(int(since_epoch)), "limit": limit,
                                   "sort": "DESC"})
        out = []
        for t in (msg.get("profit_table") or {}).get("transactions") or []:
            out.append(RemoteContract(
                contract_id=int(t["contract_id"]), transaction_id=t.get("transaction_id"),
                underlying_symbol=t.get("underlying_symbol"),
                contract_type=t.get("contract_type"), buy_price=_f(t.get("buy_price")),
                payout=_f(t.get("payout")), purchase_time=t.get("purchase_time"),
                sell_price=_f(t.get("sell_price")), sell_time=t.get("sell_time")))
        return out
