"""Autenticación Deriv: PAT por REST -> URL WebSocket con OTP (Fase 2).

Contrato verificado en deriv-com/deriv-api-schemas (rest-api-openapi.json, v1.1.0):
    GET  /trading/v1/options/accounts                 -> {"data": [OptionsAccount], "meta"}
    POST /trading/v1/options/accounts/{accountId}/otp -> {"data": {"url"}, "meta"}
    Cabecera obligatoria ``Deriv-App-ID``; seguridad oauth2, scope ``trade``.
    OptionsAccount: account_id, balance, currency, group, status (active|inactive),
                    account_type (demo|real).
    Error: {"errors": [{"status", "code", "message", "field"?}], "meta"}
    OTP: válido 120 s y de un solo uso -> se pide uno NUEVO en cada (re)conexión.

Reglas aplicadas:
- Token y app_id se leen en cada uso con ``config.get_secret`` (entorno o keyring);
  no se guardan en atributos, BD ni logs. ``repr`` nunca los muestra.
- La URL OTP no se registra, imprime ni persiste: se valida y se entrega al WS.
- Solo cuentas ``demo`` y ``active``. Una cuenta ``real`` lanza RealModeLockedError.
- Una URL OTP que no apunte a ``/ws/demo`` en un host permitido se rechaza.
- Scope mínimo: el bot solo necesita ``trade``. Nunca crear el PAT con
  ``payment`` ni ``account_manage``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from urllib.parse import parse_qs, urlsplit

from config import (DERIV_API_BASE, REAL_MODE_AVAILABLE, AccountEnvironment,
                    RealModeLockedError, get_secret, redact)

log = logging.getLogger(__name__)

ENV_TOKEN = "DERIV_TOKEN"
ENV_APP_ID = "DERIV_APP_ID"
ENV_ACCOUNT_ID = "DERIV_ACCOUNT_ID"

ACCOUNTS_PATH = "/trading/v1/options/accounts"
OTP_PATH = "/trading/v1/options/accounts/{account_id}/otp"
WS_PATHS = {
    AccountEnvironment.DEMO: "/trading/v1/options/ws/demo",
    AccountEnvironment.REAL: "/trading/v1/options/ws/real",
}
DEFAULT_ALLOWED_WS_HOSTS = frozenset({"api.derivws.com"})


# ----------------------------------------------------------------------------
# Errores
# ----------------------------------------------------------------------------
class DerivAuthError(RuntimeError):
    """Error de autenticación o de la API REST (mensaje ya redactado)."""

    def __init__(self, message: str, *, status: int | None = None,
                 code: str | None = None) -> None:
        self.status = status
        self.code = code
        super().__init__(redact(message))


class AccountValidationError(DerivAuthError):
    """La cuenta no existe, no es demo, no está activa o no coincide."""


class InvalidOtpUrlError(DerivAuthError):
    """La URL devuelta por el endpoint OTP no es la esperada."""


def assert_environment_allowed(env: AccountEnvironment) -> None:
    if AccountEnvironment(env) is AccountEnvironment.REAL and not REAL_MODE_AVAILABLE:
        raise RealModeLockedError("Entorno Real bloqueado")


# ----------------------------------------------------------------------------
# Modelo de cuenta
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OptionsAccount:
    account_id: str
    balance: float
    currency: str
    group: str
    status: str
    account_type: str

    @property
    def is_demo(self) -> bool:
        return self.account_type == AccountEnvironment.DEMO.value

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "OptionsAccount":
        try:
            return cls(account_id=str(raw["account_id"]), balance=float(raw["balance"]),
                       currency=str(raw["currency"]), group=str(raw["group"]),
                       status=str(raw["status"]), account_type=str(raw["account_type"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise DerivAuthError(f"Cuenta con formato inesperado: falta/incorrecto {exc}") \
                from None


# ----------------------------------------------------------------------------
# Cliente REST
# ----------------------------------------------------------------------------
class HttpSession(Protocol):
    def get(self, url: str, **kw: Any) -> Any: ...
    def post(self, url: str, **kw: Any) -> Any: ...


SecretReader = Callable[[str], str | None]


def _default_session() -> HttpSession:
    import requests

    return requests.Session()


class DerivRestClient:
    def __init__(self, base_url: str = DERIV_API_BASE, *,
                 session: HttpSession | None = None, timeout_s: float = 15.0,
                 secret_reader: SecretReader = get_secret) -> None:
        if urlsplit(base_url).scheme != "https":
            raise ValueError("La API REST de Deriv debe usarse sobre HTTPS")
        self.base_url = base_url.rstrip("/")
        self._session = session or _default_session()
        self.timeout_s = timeout_s
        self._read_secret = secret_reader

    def __repr__(self) -> str:  # nunca exponer credenciales
        return f"DerivRestClient(base_url={self.base_url!r})"

    def _headers(self) -> dict[str, str]:
        # Se leen en cada llamada: no quedan como atributo del objeto.
        token = self._read_secret(ENV_TOKEN)
        app_id = self._read_secret(ENV_APP_ID)
        return {"Authorization": f"Bearer {token}", "Deriv-App-ID": str(app_id),
                "Accept": "application/json"}

    def _call(self, method: str, path: str) -> dict[str, Any]:
        url = self.base_url + path
        sender = self._session.get if method == "GET" else self._session.post
        headers = self._headers()          # SecretNotFoundError se propaga tal cual
        try:
            # allow_redirects=False: nunca reenviar el Bearer a otro destino.
            resp = sender(url, headers=headers, timeout=self.timeout_s,
                          allow_redirects=False)
        except Exception as exc:
            raise DerivAuthError(f"{method} {path}: fallo de red ({type(exc).__name__})") \
                from None
        status = getattr(resp, "status_code", 0)
        try:
            body = resp.json()
        except Exception:
            body = None
        if status >= 300 or not isinstance(body, dict):
            code, msg = None, f"HTTP {status}"
            errors = (body or {}).get("errors") if isinstance(body, dict) else None
            if errors and isinstance(errors, list) and isinstance(errors[0], dict):
                code = errors[0].get("code")
                msg = f"HTTP {status} {code}: {errors[0].get('message', '')}"
            # Por si el servidor reflejara credenciales en el mensaje.
            for value in (headers.get("Authorization", "")[7:], headers.get("Deriv-App-ID")):
                if value:
                    msg = msg.replace(value, "<redacted>")
            raise DerivAuthError(f"{method} {path}: {msg}", status=status, code=code)
        return body

    def list_accounts(self) -> list[OptionsAccount]:
        body = self._call("GET", ACCOUNTS_PATH)
        data = body.get("data")
        if not isinstance(data, list):
            raise DerivAuthError("Respuesta de cuentas sin lista 'data'")
        return [OptionsAccount.from_api(a) for a in data]

    def request_otp_url(self, account_id: str) -> str:
        """Devuelve la URL WS con OTP. NO registrar ni imprimir el valor devuelto."""
        body = self._call("POST", OTP_PATH.format(account_id=account_id))
        url = (body.get("data") or {}).get("url")
        if not isinstance(url, str) or not url:
            raise DerivAuthError("Respuesta OTP sin 'data.url'")
        return url


# ----------------------------------------------------------------------------
# Validaciones
# ----------------------------------------------------------------------------
def validate_otp_url(url: str, env: AccountEnvironment = AccountEnvironment.DEMO,
                     allowed_hosts: frozenset[str] = DEFAULT_ALLOWED_WS_HOSTS) -> str:
    """Comprueba esquema, host, ruta del entorno y presencia de ``otp``.

    Los mensajes de error nunca incluyen la URL.
    """
    assert_environment_allowed(env)
    parts = urlsplit(url)
    if parts.scheme != "wss":
        raise InvalidOtpUrlError("La URL OTP no usa wss://")
    if (parts.hostname or "").lower() not in allowed_hosts:
        raise InvalidOtpUrlError(f"Host WS no permitido: {parts.hostname}")
    path = parts.path.rstrip("/")
    if path == WS_PATHS[AccountEnvironment.REAL]:
        raise RealModeLockedError("El OTP apunta al endpoint Real; conexión rechazada")
    if path != WS_PATHS[env]:
        raise InvalidOtpUrlError(f"Ruta WS inesperada para {env.value}: {path}")
    if not parse_qs(parts.query).get("otp"):
        raise InvalidOtpUrlError("La URL OTP no contiene el parámetro 'otp'")
    return url


def select_demo_account(accounts: list[OptionsAccount],
                        account_id: str | None = None) -> OptionsAccount:
    """Elige la cuenta Demo. Nunca selecciona una cuenta Real automáticamente.

    - Con ``account_id``: debe existir, ser demo y estar activa.
    - Sin él: solo si existe exactamente UNA cuenta demo activa.
    """
    if account_id:
        match = [a for a in accounts if a.account_id == account_id]
        if not match:
            raise AccountValidationError(f"La cuenta {account_id} no pertenece a este token")
        acc = match[0]
        if not acc.is_demo:
            raise RealModeLockedError(
                f"La cuenta {account_id} es de tipo '{acc.account_type}': Real bloqueado")
        if not acc.is_active:
            raise AccountValidationError(f"La cuenta demo {account_id} no está activa")
        return acc

    demos = [a for a in accounts if a.is_demo and a.is_active]
    if len(demos) == 1:
        return demos[0]
    if not demos:
        raise AccountValidationError("No hay ninguna cuenta Demo activa para este token")
    ids = ", ".join(a.account_id for a in demos)
    raise AccountValidationError(
        f"Hay varias cuentas Demo activas ({ids}); indica una con {ENV_ACCOUNT_ID} "
        "o --account-id")


class OtpUrlProvider:
    """``url_provider`` para DerivWSClient: pide y valida un OTP NUEVO en cada llamada."""

    def __init__(self, rest: DerivRestClient, account_id: str,
                 env: AccountEnvironment = AccountEnvironment.DEMO,
                 allowed_hosts: frozenset[str] = DEFAULT_ALLOWED_WS_HOSTS) -> None:
        assert_environment_allowed(env)
        self.rest = rest
        self.account_id = account_id
        self.env = env
        self.allowed_hosts = allowed_hosts
        self.issued = 0

    def __repr__(self) -> str:
        return f"OtpUrlProvider(account_id={self.account_id!r}, env={self.env.value})"

    def __call__(self) -> str:
        url = validate_otp_url(self.rest.request_otp_url(self.account_id), self.env,
                               self.allowed_hosts)
        self.issued += 1
        log.info("OTP emitido para %s (n.º %d)", self.account_id, self.issued)
        return url
