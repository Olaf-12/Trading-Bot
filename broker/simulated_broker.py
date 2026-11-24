"""Broker simulado para backtesting y pruebas (Fase 4).

Implementa ``ExecutionPort`` igual que Deriv, de modo que ExecutionController y
RiskManager se ejercitan sin cambios. Reglas de liquidación Rise/Fall:
- Entrada (``entry_spot``): primer tick POSTERIOR a la compra (+ latencia simulada).
- Duración en ticks (``t``): salida = N-ésimo tick tras la entrada.
- Duración en tiempo (s/m/h): salida = último tick con epoch <= vencimiento
  (fecha de vencimiento = epoch del tick de compra + duración).
- CALL gana si salida > entrada; PUT si salida < entrada. Empate = pérdida.
- Ganancia = payout - precio; pérdida = -precio.
El payout real de Deriv varía por símbolo/duración: ``payout_ratio`` es un supuesto.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from broker.broker_interface import (Balance, BrokerRejection, BuyReceipt, ContractCallback,
                                     ContractUpdate, ExecutionPort, Proposal, RemoteContract)
from core.order_approval import ApprovalError, OrderApproval
from market.market_data import Tick
from strategy.signals import ContractSpec


@dataclass
class _SimContract:
    contract_id: int
    spec: ContractSpec
    price: float
    payout: float
    purchase_epoch: int
    ticks_to_entry: int
    entry: Tick | None = None
    ticks_after_entry: int = 0
    last_before_expiry: Tick | None = None
    exit: Tick | None = None
    status: str = "open"
    callbacks: list[ContractCallback] = field(default_factory=list)

    @property
    def expiry(self) -> int | None:
        secs = self.spec.duration_seconds()
        return None if secs is None else self.purchase_epoch + secs


class SimulatedBroker(ExecutionPort):
    def __init__(self, *, balance: float = 10_000.0, currency: str = "USD",
                 payout_ratio: float = 1.95, latency_ticks: int = 0,
                 account_id: str = "SIM0000001") -> None:
        self._account_id = account_id
        self.currency = currency
        self.balance = balance
        self.payout_ratio = payout_ratio
        self.latency_ticks = latency_ticks
        self.now = 0.0
        self.last_tick: Tick | None = None
        self.contracts: dict[int, _SimContract] = {}
        self.settled: list[ContractUpdate] = []
        self._ids = itertools.count(1)
        self._proposals: dict[str, Proposal] = {}

    # --------------------------------------------------------------- reloj
    def clock(self) -> float:
        return self.now

    # --------------------------------------------------------- ExecutionPort
    @property
    def account_id(self) -> str:
        return self._account_id

    def ensure_account_valid(self) -> Balance:
        return Balance(round(self.balance, 2), self.currency, self._account_id)

    def get_proposal(self, spec: ContractSpec) -> Proposal:
        if self.last_tick is None:
            raise BrokerRejection("Sin precio de mercado")
        if spec.symbol != self.last_tick.symbol:
            raise BrokerRejection(f"Símbolo {spec.symbol} no disponible")
        if spec.currency != self.currency:
            raise BrokerRejection("Moneda no válida")
        ask = round(spec.stake, 2)
        pid = f"sim-prop-{next(self._ids)}"
        secs = spec.duration_seconds()
        p = Proposal(id=pid, ask_price=ask, payout=round(ask * self.payout_ratio, 2),
                     spot=self.last_tick.quote, spot_time=self.last_tick.epoch,
                     date_start=int(self.now),
                     date_expiry=int(self.now) + secs if secs else None,
                     longcode=f"Sim {spec.contract_type} {spec.symbol}", spec=spec)
        self._proposals[pid] = p
        return p

    def buy(self, proposal: Proposal, approval: OrderApproval) -> BuyReceipt:
        if not isinstance(approval, OrderApproval):
            raise ApprovalError("Compra sin aprobación de ExecutionController")
        approval.consume(proposal_id=proposal.id, price=proposal.ask_price,
                         account_id=self._account_id)
        if self._proposals.pop(proposal.id, None) is None:
            raise BrokerRejection("Propuesta inexistente o ya usada")
        if proposal.ask_price > self.balance:
            raise BrokerRejection("Saldo insuficiente")
        cid = 10_000 + next(self._ids)
        self.balance -= proposal.ask_price
        self.contracts[cid] = _SimContract(cid, proposal.spec, proposal.ask_price,
                                           proposal.payout, int(self.now),
                                           ticks_to_entry=1 + self.latency_ticks)
        return BuyReceipt(contract_id=cid, transaction_id=cid * 10, buy_price=proposal.ask_price,
                          payout=proposal.payout, balance_after=round(self.balance, 2),
                          purchase_time=int(self.now), start_time=int(self.now),
                          longcode=proposal.longcode, shortcode=f"SIM_{cid}")

    def track_contract(self, contract_id: int, on_update: ContractCallback) -> None:
        c = self.contracts.get(contract_id)
        if c is None:
            raise KeyError(contract_id)
        c.callbacks.append(on_update)
        on_update(self._update(c))

    def open_contracts(self) -> list[RemoteContract]:
        return [RemoteContract(c.contract_id, c.contract_id * 10, c.spec.symbol,
                               c.spec.contract_type, c.price, c.payout, c.purchase_epoch)
                for c in self.contracts.values() if c.status == "open"]

    def settled_contracts(self, since_epoch: int, limit: int = 100) -> list[RemoteContract]:
        out = []
        for c in self.contracts.values():
            if c.status != "open" and c.purchase_epoch >= since_epoch:
                sell = c.payout if c.status == "won" else 0.0
                out.append(RemoteContract(c.contract_id, c.contract_id * 10, c.spec.symbol,
                                          c.spec.contract_type, c.price, c.payout,
                                          c.purchase_epoch, sell,
                                          c.exit.epoch if c.exit else None))
        return out[-limit:]

    # ---------------------------------------------------------- simulación
    def on_tick(self, tick: Tick) -> None:
        """Avanza el mercado: fija entradas y liquida contratos vencidos."""
        self.now = float(tick.epoch)
        for c in list(self.contracts.values()):
            if c.status != "open":
                continue
            if c.entry is None:
                c.ticks_to_entry -= 1
                if c.ticks_to_entry <= 0:
                    c.entry = tick
                    c.last_before_expiry = tick
                    self._notify(c)
                continue
            if c.spec.duration_unit == "t":
                c.ticks_after_entry += 1
                if c.ticks_after_entry >= c.spec.duration:
                    self._settle(c, tick)
            else:
                expiry = c.expiry
                if tick.epoch == expiry:
                    self._settle(c, tick)
                elif tick.epoch > expiry:
                    self._settle(c, c.last_before_expiry or c.entry)
                else:
                    c.last_before_expiry = tick
        self.last_tick = tick

    def _settle(self, c: _SimContract, exit_tick: Tick) -> None:
        c.exit = exit_tick
        assert c.entry is not None
        if c.spec.contract_type == "CALL":
            won = exit_tick.quote > c.entry.quote
        else:
            won = exit_tick.quote < c.entry.quote
        c.status = "won" if won else "lost"
        if won:
            self.balance += c.payout
        update = self._update(c)
        self.settled.append(update)
        self._notify(c, update)

    def _notify(self, c: _SimContract, update: ContractUpdate | None = None) -> None:
        update = update or self._update(c)
        for cb in list(c.callbacks):
            cb(update)

    def _update(self, c: _SimContract) -> ContractUpdate:
        settled = c.status != "open"
        profit = (c.payout - c.price if c.status == "won" else -c.price) if settled else 0.0
        return ContractUpdate(
            contract_id=c.contract_id, status=c.status, is_sold=settled, is_expired=settled,
            buy_price=c.price, payout=c.payout, profit=round(profit, 2),
            sell_price=(c.payout if c.status == "won" else 0.0) if settled else None,
            entry_spot=c.entry.quote if c.entry else None,
            entry_spot_time=c.entry.epoch if c.entry else None,
            exit_spot=c.exit.quote if c.exit else None,
            exit_spot_time=c.exit.epoch if c.exit else None,
            date_expiry=c.expiry, sell_time=c.exit.epoch if c.exit else None,
            underlying_symbol=c.spec.symbol, contract_type=c.spec.contract_type,
            account_id=self._account_id, currency=self.currency)
