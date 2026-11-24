"""Fachada de alto nivel sobre Deriv (encapsula REST + WebSocket).

- ``DerivPublicAdapter`` (Fase 1): datos públicos sin autenticar.
- ``DerivDemoAdapter``   (Fase 2): cuenta Demo autenticada por OTP + saldo.

Separación estricta de cuentas: la cuenta se valida por REST (existe, ``demo``,
``active``) antes de conectar, y por WS (``balance.loginid`` y moneda) al conectar,
tras cada reconexión y bajo demanda (``ensure_account_valid``, que
ExecutionController llamará antes de cada compra en Fase 3). Cualquier
discrepancia bloquea el adaptador y cierra la conexión.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable

from broker.broker_interface import (Balance, IssueCallback, MarketDataBroker,
                                     TickCallback, TradingBroker)
from broker.deriv_auth import (ENV_ACCOUNT_ID, DerivAuthError, DerivRestClient,
                               InvalidOtpUrlError, OptionsAccount, OtpUrlProvider,
                               select_demo_account)
from broker.deriv_market import DerivMarketData
from broker.deriv_ws_client import ConnectionFactory, DerivWSClient, default_connection_factory
from config import AccountEnvironment, RealModeLockedError, Settings
from database.database import Database
from market.market_data import SymbolInfo, Tick, TickValidationResult

log = logging.getLogger(__name__)


class AccountMismatchError(PermissionError):
    """La cuenta conectada no coincide con la seleccionada. Bloquea el adaptador."""


class AdapterLockedError(PermissionError):
    """El adaptador quedó bloqueado por un fallo de seguridad; requiere reinicio manual."""


def parse_balance(message: dict[str, Any]) -> Balance:
    raw = message.get("balance")
    if not isinstance(raw, dict):
        raise ValueError("Respuesta 'balance' sin objeto balance")
    try:
        return Balance(balance=float(raw["balance"]), currency=str(raw["currency"]),
                       loginid=str(raw["loginid"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"balance con formato inesperado: {exc}") from None


def _is_fatal_connect_error(exc: BaseException) -> bool:
    """Errores que no se arreglan reintentando: se detiene la reconexión."""
    if isinstance(exc, (PermissionError, InvalidOtpUrlError)):
        return True
    return isinstance(exc, DerivAuthError) and exc.status in (401, 403, 404)


class _DerivMarketAdapter(MarketDataBroker):
    """Base común: cliente WS + datos de mercado + persistencia/auditoría."""

    audit_source = "deriv_ws"

    def __init__(self, settings: Settings, url_provider: Callable[[], str],
                 db: Database | None, connection_factory: ConnectionFactory,
                 on_state_change: Callable[[str], None] | None,
                 on_issue: IssueCallback | None) -> None:
        self.settings = settings
        self.db = db
        self._user_on_issue = on_issue
        self._user_on_state = on_state_change
        self.client = DerivWSClient(
            url_provider=url_provider,
            settings=settings.connection,
            connection_factory=connection_factory,
            on_state_change=self._state_changed,
            is_fatal_connect_error=_is_fatal_connect_error,
        )
        self.market = DerivMarketData(self.client, settings.ticks, on_issue=self._issue)

    def _audit(self, event: str, payload: Any = None, level: str = "INFO") -> None:
        if self.db:
            self.db.log_event(self.audit_source, event, payload, level)

    def _state_changed(self, state: str) -> None:
        self._audit(f"state:{state}")
        if self._user_on_state:
            self._user_on_state(state)

    def _issue(self, symbol: str, result: TickValidationResult) -> None:
        if self.db:
            self.db.log_data_issue(symbol, result.issue.value if result.issue else "unknown",
                                   result.detail)
        if self._user_on_issue:
            self._user_on_issue(symbol, result)

    # ------------------------------------------------------ MarketDataBroker
    def connect(self) -> None:
        self.client.start(wait=True)

    def disconnect(self) -> None:
        self.client.stop()

    @property
    def is_connected(self) -> bool:
        return self.client.is_connected

    def get_active_symbols(self, refresh: bool = False) -> list[SymbolInfo]:
        return self.market.get_active_symbols(refresh)

    def subscribe_ticks(self, symbol: str, on_tick: TickCallback) -> None:
        def persist_then_forward(tick: Tick) -> None:
            if self.db:
                self.db.insert_tick(tick.symbol, tick.epoch, tick.quote, tick.pip_size,
                                    tick.received_at)
            on_tick(tick)

        self.market.subscribe_ticks(symbol, persist_then_forward)

    def unsubscribe_ticks(self, symbol: str) -> None:
        self.market.unsubscribe_ticks(symbol)

    def download_history(self, symbol: str, count: int = 5000) -> int:
        """Descarga histórico y lo guarda en SQLite. Devuelve ticks nuevos insertados."""
        ticks = self.market.get_ticks_history(symbol, count)
        if not self.db:
            return 0
        return sum(self.db.insert_tick(t.symbol, t.epoch, t.quote, t.pip_size, t.received_at)
                   for t in ticks)

    def check_health(self) -> list[str]:
        return self.market.check_health()


class DerivPublicAdapter(_DerivMarketAdapter):
    """Datos de mercado públicos de Deriv. No puede operar ni leer saldos."""

    audit_source = "deriv_public_ws"

    def __init__(self, settings: Settings, db: Database | None = None,
                 connection_factory: ConnectionFactory = default_connection_factory,
                 on_state_change: Callable[[str], None] | None = None,
                 on_issue: IssueCallback | None = None) -> None:
        public_url = settings.connection.public_ws_url
        super().__init__(settings, lambda: public_url, db, connection_factory,
                         on_state_change, on_issue)


class DerivDemoAdapter(_DerivMarketAdapter, TradingBroker):
    """Cuenta Demo autenticada (OTP nuevo por conexión). Real: siempre rechazado."""

    audit_source = "deriv_demo_ws"
    MAX_REVERIFY_FAILURES = 3

    def __init__(self, settings: Settings, account_id: str | None = None, *,
                 db: Database | None = None, rest: DerivRestClient | None = None,
                 connection_factory: ConnectionFactory = default_connection_factory,
                 on_state_change: Callable[[str], None] | None = None,
                 on_issue: IssueCallback | None = None,
                 on_locked: Callable[[str], None] | None = None) -> None:
        self.rest = rest or DerivRestClient(settings.connection.rest_base_url,
                                            timeout_s=settings.connection.request_timeout_s)
        # account_id no es secreto: argumento o variable de entorno. Nunca de la BD.
        self.requested_account_id = account_id or os.environ.get(ENV_ACCOUNT_ID) or None
        self.account: OptionsAccount | None = None
        self._otp: OtpUrlProvider | None = None
        self._on_locked = on_locked
        self._locked_reason: str | None = None
        self._reverify_failures = 0
        self._lock = threading.Lock()
        super().__init__(settings, self._next_ws_url, db, connection_factory,
                         on_state_change, on_issue)
        self.client.add_reconnect_hook(self._verify_after_reconnect)

    # ----------------------------------------------------------------- estado
    @property
    def locked(self) -> bool:
        return self._locked_reason is not None

    @property
    def locked_reason(self) -> str | None:
        return self._locked_reason

    def _lock_adapter(self, reason: str) -> None:
        with self._lock:
            if self._locked_reason is not None:
                return
            self._locked_reason = reason
        log.error("Adaptador Demo BLOQUEADO: %s", reason)
        self._audit("adapter_locked", {"reason": reason}, level="ERROR")
        self.client.stop()
        if self._on_locked:
            try:
                self._on_locked(reason)
            except Exception:
                log.exception("on_locked falló")

    def _ensure_not_locked(self) -> None:
        if self._locked_reason is not None:
            raise AdapterLockedError(f"Adaptador bloqueado: {self._locked_reason}")

    def _next_ws_url(self) -> str:
        self._ensure_not_locked()
        if self._otp is None:
            raise RuntimeError("Cuenta no seleccionada: llama a connect()")
        return self._otp()                       # OTP nuevo y validado en cada intento

    # --------------------------------------------------------------- conexión
    def connect(self) -> None:
        self._ensure_not_locked()
        if self.settings.environment is not AccountEnvironment.DEMO:
            raise RealModeLockedError("DerivDemoAdapter solo opera en entorno Demo")
        accounts = self.rest.list_accounts()
        account = select_demo_account(accounts, self.requested_account_id)
        self.account = account
        self._otp = OtpUrlProvider(self.rest, account.account_id, AccountEnvironment.DEMO,
                                   self.settings.connection.allowed_ws_hosts)
        self._audit("account_selected", {"account_id": account.account_id,
                                         "account_type": account.account_type,
                                         "currency": account.currency})
        log.info("Cuenta Demo seleccionada: %s (%s)", account.account_id, account.currency)
        self.client.start(wait=True)
        try:
            self.ensure_account_valid()
        except BaseException:
            self.client.stop()
            raise

    @property
    def execution(self):
        """Puerto de ejecución (Fase 3). Solo debe entregarse a ExecutionController."""
        if getattr(self, "_execution", None) is None:
            from broker.deriv_execution import DerivExecution
            self._execution = DerivExecution(self)
        return self._execution

    # ------------------------------------------------------------------ saldo
    def get_balance(self) -> Balance:
        self._ensure_not_locked()
        balance = parse_balance(self.client.request({"balance": 1}))
        self._check_balance_account(balance)
        return balance

    def _check_balance_account(self, balance: Balance) -> None:
        acc = self.account
        if acc is None:
            raise RuntimeError("Cuenta no seleccionada")
        problems = []
        if balance.loginid != acc.account_id:
            problems.append(f"loginid={balance.loginid} esperado={acc.account_id}")
        if balance.currency and balance.currency != acc.currency:
            problems.append(f"moneda={balance.currency} esperada={acc.currency}")
        if problems:
            reason = "Cuenta conectada no coincide: " + "; ".join(problems)
            self._lock_adapter(reason)
            raise AccountMismatchError(reason)

    def ensure_account_valid(self) -> Balance:
        """Revalida que la sesión WS pertenece a la cuenta Demo elegida.

        Llamar al iniciar y antes de cada compra (ExecutionController, Fase 3).
        """
        balance = self.get_balance()
        self._audit("account_verified", {"account_id": balance.loginid,
                                         "currency": balance.currency})
        return balance

    def _verify_after_reconnect(self) -> None:
        if self.locked:
            return
        try:
            self.ensure_account_valid()
            self._reverify_failures = 0
        except AccountMismatchError:
            pass                                # ya bloqueado y auditado
        except Exception as exc:
            # Sin verificación no se puede garantizar la cuenta.
            self._reverify_failures += 1
            log.warning("No se pudo revalidar la cuenta tras reconectar (%d/%d): %s",
                        self._reverify_failures, self.MAX_REVERIFY_FAILURES, exc)
            if self._reverify_failures >= self.MAX_REVERIFY_FAILURES:
                self._lock_adapter("La cuenta no pudo revalidarse tras reconectar")
            else:
                self.client.force_reconnect("revalidación de cuenta fallida")
