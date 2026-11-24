"""Contratos abstractos de broker: permiten sustituir Deriv por el simulador (Fase 4)
sin cambiar estrategia, riesgo ni ExecutionController."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from market.market_data import SymbolInfo, Tick, TickValidationResult

if TYPE_CHECKING:  # evita import circular en tiempo de ejecución
    from core.order_approval import OrderApproval
    from strategy.signals import ContractSpec

class BrokerRejection(RuntimeError):
    """El broker rechazó explícitamente la petición (la operación NO se ejecutó)."""


TickCallback = Callable[[Tick], None]
IssueCallback = Callable[[str, TickValidationResult], None]


# ----------------------------------------------------------------------------
# Tipos de datos de trading (campos según esquemas oficiales de Deriv)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Balance:
    balance: float
    currency: str
    loginid: str


@dataclass(frozen=True)
class Proposal:
    id: str
    ask_price: float
    payout: float
    spot: float
    spot_time: int
    date_start: int
    date_expiry: int | None
    longcode: str
    spec: "ContractSpec"

    @property
    def payout_ratio(self) -> float:
        return self.payout / self.ask_price if self.ask_price > 0 else 0.0


@dataclass(frozen=True)
class BuyReceipt:
    contract_id: int
    transaction_id: int
    buy_price: float
    payout: float
    balance_after: float
    purchase_time: int
    start_time: int
    longcode: str
    shortcode: str


def _num(value: Any) -> float | None:
    """Los importes de proposal_open_contract llegan como string en el esquema nuevo."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ContractUpdate:
    contract_id: int
    status: str | None          # open | sold | won | lost | cancelled
    is_sold: bool
    is_expired: bool
    buy_price: float | None
    payout: float | None
    profit: float | None
    sell_price: float | None
    entry_spot: float | None
    entry_spot_time: int | None
    exit_spot: float | None
    exit_spot_time: int | None
    date_expiry: int | None
    sell_time: int | None
    underlying_symbol: str | None
    contract_type: str | None
    account_id: str | None
    currency: str | None

    @property
    def is_settled(self) -> bool:
        return self.is_sold or self.status in ("won", "lost", "sold", "cancelled")

    @classmethod
    def from_api(cls, poc: dict[str, Any]) -> "ContractUpdate":
        def _int(v: Any) -> int | None:
            return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

        return cls(
            contract_id=int(poc["contract_id"]),
            status=poc.get("status"),
            is_sold=bool(poc.get("is_sold")),
            is_expired=bool(poc.get("is_expired")),
            buy_price=_num(poc.get("buy_price")),
            payout=_num(poc.get("payout")),
            profit=_num(poc.get("profit")),
            sell_price=_num(poc.get("sell_price")),
            entry_spot=_num(poc.get("entry_spot")),
            entry_spot_time=_int(poc.get("entry_spot_time")),
            exit_spot=_num(poc.get("exit_spot")),
            exit_spot_time=_int(poc.get("exit_spot_time")),
            date_expiry=_int(poc.get("date_expiry")),
            sell_time=_int(poc.get("sell_time")),
            underlying_symbol=poc.get("underlying_symbol"),
            contract_type=poc.get("contract_type"),
            account_id=poc.get("account_id"),
            currency=poc.get("currency"),
        )


@dataclass(frozen=True)
class RemoteContract:
    """Contrato según portfolio/profit_table (reconciliación, Fase 7)."""

    contract_id: int
    transaction_id: int | None
    underlying_symbol: str | None
    contract_type: str | None
    buy_price: float | None
    payout: float | None
    purchase_time: int | None
    sell_price: float | None = None
    sell_time: int | None = None

    @property
    def profit(self) -> float | None:
        if self.sell_price is None or self.buy_price is None:
            return None
        return self.sell_price - self.buy_price


ContractCallback = Callable[[ContractUpdate], None]


# ----------------------------------------------------------------------------
# Interfaces
# ----------------------------------------------------------------------------
class MarketDataBroker(ABC):
    """Capacidades de datos de mercado (Fase 1)."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def get_active_symbols(self, refresh: bool = False) -> list[SymbolInfo]: ...

    @abstractmethod
    def subscribe_ticks(self, symbol: str, on_tick: TickCallback) -> None: ...

    @abstractmethod
    def unsubscribe_ticks(self, symbol: str) -> None: ...


class TradingBroker(MarketDataBroker, ABC):
    """Capacidades de cuenta (Fase 2)."""

    @abstractmethod
    def get_balance(self) -> Balance: ...


class ExecutionPort(ABC):
    """Operaciones de trading. SOLO ExecutionController puede invocarlas:
    ``buy`` exige un ``OrderApproval`` de un solo uso emitido por el controlador."""

    @property
    @abstractmethod
    def account_id(self) -> str: ...

    @abstractmethod
    def ensure_account_valid(self) -> Balance: ...

    @abstractmethod
    def get_proposal(self, spec: "ContractSpec") -> Proposal: ...

    @abstractmethod
    def buy(self, proposal: Proposal, approval: "OrderApproval") -> BuyReceipt: ...

    @abstractmethod
    def track_contract(self, contract_id: int, on_update: ContractCallback) -> None: ...

    @abstractmethod
    def open_contracts(self) -> list[RemoteContract]: ...

    @abstractmethod
    def settled_contracts(self, since_epoch: int, limit: int = 100) -> list[RemoteContract]: ...
