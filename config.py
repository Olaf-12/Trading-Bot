"""Configuración central de TradingBotML.

Reglas de seguridad aplicadas aquí:
- El entorno es SIEMPRE ``DEMO`` al arrancar. ``REAL`` no se puede seleccionar
  por variable de entorno, ni por fichero, ni por configuración persistida.
- Los secretos (token PAT, app_id) se leen del entorno o del almacén seguro del
  SO (``keyring``). Nunca se definen en código ni se guardan en SQLite.
- Las URLs OTP y los tokens se redactan en todos los logs.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_DB_PATH = DATA_DIR / "tradingbot.sqlite3"

# Endpoints verificados contra deriv-com/deriv-api-schemas (rest-api-openapi.json):
#   GET  /trading/v1/options/ws/public              (WS sin autenticación)
#   GET  /trading/v1/options/ws/demo?otp=...        (Fase 2)
#   GET  /trading/v1/options/ws/real?otp=...        (bloqueado)
#   POST /trading/v1/options/accounts/{accountId}/otp  (cabecera Deriv-App-ID)
# El host procede del PDF de arquitectura; no figura en el fragmento del spec revisado.
DERIV_API_BASE = "https://api.derivws.com"
DERIV_PUBLIC_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"


class AccountEnvironment(str, Enum):
    DEMO = "demo"
    REAL = "real"


class RealModeLockedError(PermissionError):
    """Se intentó seleccionar el entorno Real mientras está bloqueado."""


class SecretNotFoundError(LookupError):
    """No se encontró un secreto requerido en el entorno ni en el keyring."""


# Real permanece bloqueado hasta la Fase 8 (revisión manual, legal y de seguridad).
# Es una constante de código, no un ajuste: no puede activarse desde config/BD/entorno.
REAL_MODE_AVAILABLE = False


@dataclass(frozen=True)
class ConnectionSettings:
    public_ws_url: str = DERIV_PUBLIC_WS_URL
    rest_base_url: str = DERIV_API_BASE
    # Hosts aceptados en la URL OTP devuelta por la API (wss://<host>/.../ws/demo).
    allowed_ws_hosts: frozenset[str] = frozenset({"api.derivws.com"})
    connect_timeout_s: float = 15.0
    request_timeout_s: float = 15.0
    recv_poll_s: float = 1.0            # timeout de recv() para revisar la señal de parada
    ping_interval_s: float = 30.0       # keepalive si no hubo tráfico saliente
    reconnect_initial_delay_s: float = 1.0
    reconnect_max_delay_s: float = 60.0
    reconnect_max_attempts: int = 0     # 0 = ilimitado


@dataclass(frozen=True)
class TickValidationSettings:
    stale_after_s: float = 10.0         # sin ticks durante este tiempo -> stream obsoleto
    reconnect_after_stale_s: float = 30.0
    max_future_skew_s: float = 5.0      # epoch del tick demasiado en el futuro
    max_relative_jump: float = 0.05     # salto > 5 % entre ticks consecutivos -> anomalía


@dataclass
class Settings:
    connection: ConnectionSettings = field(default_factory=ConnectionSettings)
    ticks: TickValidationSettings = field(default_factory=TickValidationSettings)
    db_path: Path = DEFAULT_DB_PATH
    log_level: str = "INFO"
    # El entorno no se lee del entorno/BD: siempre arranca en DEMO.
    _environment: AccountEnvironment = field(default=AccountEnvironment.DEMO, init=False)

    @property
    def environment(self) -> AccountEnvironment:
        return self._environment

    def select_environment(self, env: AccountEnvironment) -> None:
        """Cambia de entorno en memoria. Real está bloqueado hasta la Fase 8."""
        env = AccountEnvironment(env)
        if env is AccountEnvironment.REAL and not REAL_MODE_AVAILABLE:
            raise RealModeLockedError(
                "El entorno Real está bloqueado. Solo se habilitará tras la Fase 8 "
                "y con confirmación explícita del usuario en cada sesión."
            )
        self._environment = env


def load_settings() -> Settings:
    """Carga ajustes no secretos. Solo se aceptan overrides inocuos desde el entorno."""
    db_path = os.environ.get("TRADINGBOT_DB_PATH")
    return Settings(
        db_path=Path(db_path) if db_path else DEFAULT_DB_PATH,
        log_level=os.environ.get("TRADINGBOT_LOG_LEVEL", "INFO"),
    )


# ----------------------------------------------------------------------------
# Secretos
# ----------------------------------------------------------------------------
KEYRING_SERVICE = "TradingBotML"


def get_secret(name: str, *, required: bool = True) -> str | None:
    """Lee un secreto desde variable de entorno o, en su defecto, del keyring del SO.

    No cachea el valor ni lo escribe en disco/BD/log.
    """
    value = os.environ.get(name)
    if not value:
        try:
            import keyring  # import diferido: opcional en entornos sin backend

            value = keyring.get_password(KEYRING_SERVICE, name)
        except Exception:  # backend no disponible
            value = None
    if not value and required:
        raise SecretNotFoundError(
            f"Secreto '{name}' no encontrado en variables de entorno ni en el keyring."
        )
    return value or None


# ----------------------------------------------------------------------------
# Logging con redacción de secretos
# ----------------------------------------------------------------------------
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(otp=)[^&\s\"']+", re.IGNORECASE), r"\1<redacted>"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE), r"\1<redacted>"),
    (re.compile(r"(\"?(?:token|authorize|password|secret|otp)\"?\s*[:=]\s*\"?)[^\"',\s}]+",
                re.IGNORECASE), r"\1<redacted>"),
    (re.compile(r"wss://[^\s\"']*/ws/(?:demo|real)[^\s\"']*", re.IGNORECASE),
     "<redacted-authenticated-ws-url>"),
)


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
