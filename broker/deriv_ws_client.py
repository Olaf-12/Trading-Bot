"""Cliente WebSocket de bajo nivel para Deriv con bucle receptor central.

Principios (PDF §6.3):
- Un ÚNICO hilo lee del socket. Nadie más llama a ``recv()``.
- Cada petición lleva ``req_id`` propio; la respuesta se entrega a su ``Future``.
- Los streams se enrutan por ``subscription.id`` y, como respaldo, por el
  ``req_id`` de la petición que los creó. Los mensajes no solicitados se
  enrutan por ``msg_type`` o se descartan con registro.
- Al caer la conexión: se fallan las peticiones pendientes, se reconecta con
  backoff exponencial + jitter y se vuelven a enviar las suscripciones activas.

La URL de conexión se obtiene mediante ``url_provider`` en CADA intento, de modo
que en Fase 2 se pueda pedir una URL OTP nueva (de un solo uso) al reconectar.
La URL nunca se registra.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from broker.broker_interface import BrokerRejection
from config import ConnectionSettings, redact

log = logging.getLogger(__name__)

MessageHandler = Callable[[dict[str, Any]], None]


class WSConnection(Protocol):
    """Interfaz mínima de conexión (websocket-client o un fake de pruebas)."""

    def send(self, payload: str) -> Any: ...
    def recv(self) -> str: ...
    def close(self) -> None: ...
    def settimeout(self, timeout: float | None) -> None: ...


ConnectionFactory = Callable[[str, float], WSConnection]


class DerivAPIError(BrokerRejection):
    def __init__(self, error: dict[str, Any], msg_type: str | None = None) -> None:
        self.code = error.get("code")
        self.error_message = error.get("message", "")
        self.details = error.get("details")
        self.msg_type = msg_type
        super().__init__(f"[{self.code}] {self.error_message}")


class ConnectionLostError(ConnectionError):
    """La conexión se cerró con peticiones pendientes."""


class NotConnectedError(ConnectionError):
    pass


class RecvTimeout(Exception):
    """Timeout normal de recv(); el fake de pruebas también lo lanza."""


def default_connection_factory(url: str, timeout: float) -> WSConnection:
    from websocket import create_connection

    return create_connection(url, timeout=timeout)


def _is_recv_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (RecvTimeout, TimeoutError)):
        return True
    try:
        from websocket import WebSocketTimeoutException
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, WebSocketTimeoutException)


@dataclass
class _Pending:
    future: Future
    request: dict[str, Any]


@dataclass
class Subscription:
    """Stream lógico que sobrevive a reconexiones (cambian req_id y subscription.id)."""

    key: str
    request: dict[str, Any]           # sin req_id
    on_message: MessageHandler
    req_id: int | None = None
    stream_id: str | None = None
    active: bool = True
    first_response: Future = field(default_factory=Future)


class DerivWSClient:
    def __init__(
        self,
        url_provider: Callable[[], str],
        settings: ConnectionSettings | None = None,
        connection_factory: ConnectionFactory = default_connection_factory,
        on_state_change: Callable[[str], None] | None = None,
        is_fatal_connect_error: Callable[[BaseException], bool] | None = None,
    ) -> None:
        self._url_provider = url_provider
        # Errores al conectar que NO se reintentan (credenciales, entorno bloqueado...).
        self._is_fatal = is_fatal_connect_error or (lambda e: isinstance(e, PermissionError))
        self.last_error: BaseException | None = None
        self.settings = settings or ConnectionSettings()
        self._factory = connection_factory
        self._on_state_change = on_state_change

        self._conn: WSConnection | None = None
        self._send_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._req_ids = itertools.count(1)
        self._pending: dict[int, _Pending] = {}
        self._subs: dict[str, Subscription] = {}
        self._subs_by_req: dict[int, Subscription] = {}
        self._subs_by_stream: dict[str, Subscription] = {}
        self._type_handlers: dict[str, list[MessageHandler]] = {}
        self._reconnect_hooks: list[Callable[[], None]] = []

        self._stop = threading.Event()
        self._connected = threading.Event()
        self._force_reconnect = threading.Event()
        self._supervisor: threading.Thread | None = None
        self._last_send = 0.0
        self.connection_count = 0
        self.state = "disconnected"

    # ----------------------------------------------------------------- estado
    def _set_state(self, state: str) -> None:
        self.state = state
        log.info("WS estado -> %s", state)
        if self._on_state_change:
            try:
                self._on_state_change(state)
            except Exception:
                log.exception("on_state_change falló")

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def on_msg_type(self, msg_type: str, handler: MessageHandler) -> None:
        self._type_handlers.setdefault(msg_type, []).append(handler)

    def add_reconnect_hook(self, hook: Callable[[], None]) -> None:
        self._reconnect_hooks.append(hook)

    # ------------------------------------------------------------- ciclo vida
    def start(self, wait: bool = True, timeout: float | None = None) -> None:
        if self._supervisor and self._supervisor.is_alive():
            return
        self._stop.clear()
        self._supervisor = threading.Thread(target=self._run, name="deriv-ws-supervisor",
                                            daemon=True)
        self._supervisor.start()
        if not wait:
            return
        deadline = time.monotonic() + (timeout or self.settings.connect_timeout_s)
        # Sale antes si el supervisor termina (error no recuperable).
        while not self._connected.is_set() and self._supervisor.is_alive() \
                and time.monotonic() < deadline:
            self._connected.wait(0.05)
        if not self._connected.is_set():
            self.stop()   # no dejar reintentos huérfanos en segundo plano
            cause = self.last_error
            if cause is not None and self._is_fatal(cause):
                raise cause
            raise NotConnectedError("No se pudo conectar al WebSocket de Deriv") from cause

    def stop(self) -> None:
        self._stop.set()
        self._close_conn()
        if self._supervisor and self._supervisor is not threading.current_thread():
            self._supervisor.join(timeout=5)
        self._fail_pending(ConnectionLostError("cliente detenido"))
        self._set_state("stopped")

    def force_reconnect(self, reason: str = "") -> None:
        """Cierra el socket para que el supervisor reconecte (p. ej. datos obsoletos)."""
        log.warning("Reconexión forzada: %s", reason)
        self._force_reconnect.set()
        self._close_conn()

    def _close_conn(self) -> None:
        conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                self._set_state("connecting")
                url = self._url_provider()          # nueva URL (OTP) en cada intento
                conn = self._factory(url, self.settings.connect_timeout_s)
                del url
                conn.settimeout(self.settings.recv_poll_s)
                self._conn = conn
            except Exception as exc:
                self.last_error = exc
                if self._is_fatal(exc):
                    log.error("Error no recuperable al conectar: %s", redact(str(exc)))
                    self._set_state("failed")
                    break
                attempt += 1
                log.warning("Fallo al conectar (intento %d): %s", attempt, redact(str(exc)))
                if not self._backoff(attempt):
                    break
                continue

            attempt = 0
            self._last_send = time.monotonic()
            self.connection_count += 1
            self._force_reconnect.clear()
            self._connected.set()
            self._set_state("connected")
            if self.connection_count > 1:
                self._resubscribe_all()
                # En otro hilo: un hook puede hacer request() (p. ej. revalidar la
                # cuenta con 'balance') y necesita que el bucle receptor ya corra.
                threading.Thread(target=self._run_reconnect_hooks,
                                 name="deriv-ws-reconnect-hooks", daemon=True).start()

            self._receive_loop(conn)

            self._connected.clear()
            self._conn = None
            self._fail_pending(ConnectionLostError("conexión WebSocket perdida"))
            for sub in self._subs.values():
                sub.stream_id = None
            if self._stop.is_set():
                break
            self._set_state("reconnecting")
            attempt += 1
            # También tras reconexión forzada: nunca reconectar en bucle cerrado.
            if not self._backoff(attempt):
                break
        self._connected.clear()

    def _run_reconnect_hooks(self) -> None:
        for hook in list(self._reconnect_hooks):
            try:
                hook()
            except Exception:
                log.exception("reconnect hook falló")

    def _backoff(self, attempt: int) -> bool:
        max_attempts = self.settings.reconnect_max_attempts
        if max_attempts and attempt > max_attempts:
            log.error("Máximo de reintentos alcanzado (%d)", max_attempts)
            self._set_state("failed")
            return False
        delay = min(self.settings.reconnect_max_delay_s,
                    self.settings.reconnect_initial_delay_s * (2 ** (attempt - 1)))
        delay *= random.uniform(0.8, 1.2)
        return not self._stop.wait(delay)

    # ----------------------------------------------------------- bucle central
    def _receive_loop(self, conn: WSConnection) -> None:
        while not self._stop.is_set():
            if time.monotonic() - self._last_send > self.settings.ping_interval_s:
                try:
                    self._send_raw({"ping": 1, "req_id": next(self._req_ids)})
                except Exception:
                    return
            try:
                raw = conn.recv()
            except Exception as exc:
                if _is_recv_timeout(exc):
                    continue
                if not self._stop.is_set():
                    log.warning("Conexión cerrada: %s", type(exc).__name__)
                return
            if not raw:          # websocket-client devuelve '' al cerrar
                return
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                log.warning("Mensaje no JSON descartado")
                continue
            if isinstance(message, dict):
                self.dispatch(message)

    def dispatch(self, message: dict[str, Any]) -> None:
        """Enruta un mensaje entrante. Público para facilitar pruebas."""
        req_id = message.get("req_id")
        msg_type = message.get("msg_type")
        stream_id = (message.get("subscription") or {}).get("id")
        handled = False

        with self._state_lock:
            pending = self._pending.pop(req_id, None) if isinstance(req_id, int) else None
            sub = None
            if stream_id:
                sub = self._subs_by_stream.get(stream_id)
            if sub is None and isinstance(req_id, int):
                sub = self._subs_by_req.get(req_id)
            if sub is not None and stream_id and sub.stream_id != stream_id:
                sub.stream_id = stream_id
                self._subs_by_stream[stream_id] = sub

        if pending is not None:
            handled = True
            if "error" in message:
                pending.future.set_exception(DerivAPIError(message["error"], msg_type))
            else:
                pending.future.set_result(message)

        if sub is not None:
            handled = True
            if not sub.first_response.done():
                if "error" in message:
                    sub.first_response.set_exception(DerivAPIError(message["error"], msg_type))
                else:
                    sub.first_response.set_result(message)
            if "error" not in message and sub.active:
                try:
                    sub.on_message(message)
                except Exception:
                    log.exception("Handler de suscripción '%s' falló", sub.key)

        for handler in self._type_handlers.get(msg_type or "", []):
            handled = True
            try:
                handler(message)
            except Exception:
                log.exception("Handler msg_type=%s falló", msg_type)

        if not handled and msg_type != "ping":
            log.debug("Mensaje no solicitado descartado: msg_type=%s req_id=%s",
                      msg_type, req_id)

    # ----------------------------------------------------------------- envío
    def _send_raw(self, payload: dict[str, Any]) -> None:
        conn = self._conn
        if conn is None or not self._connected.is_set():
            raise NotConnectedError("WebSocket no conectado")
        with self._send_lock:
            conn.send(json.dumps(payload))
            self._last_send = time.monotonic()

    def request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """Envía una petición y espera SU respuesta (correlada por req_id)."""
        if "subscribe" in payload:
            raise ValueError("Usa subscribe() para peticiones con 'subscribe'")
        req_id = next(self._req_ids)
        msg = {**payload, "req_id": req_id}
        fut: Future = Future()
        with self._state_lock:
            self._pending[req_id] = _Pending(fut, msg)
        try:
            self._send_raw(msg)
            return fut.result(timeout or self.settings.request_timeout_s)
        except FutureTimeout as exc:
            raise TimeoutError(f"Sin respuesta a req_id={req_id}") from exc
        finally:
            with self._state_lock:
                self._pending.pop(req_id, None)

    def subscribe(self, key: str, payload: dict[str, Any], on_message: MessageHandler,
                  timeout: float | None = None) -> dict[str, Any]:
        """Crea un stream persistente (se re-suscribe solo tras reconectar)."""
        with self._state_lock:
            if key in self._subs and self._subs[key].active:
                raise ValueError(f"Suscripción '{key}' ya existe")
            sub = Subscription(key=key, request={**payload, "subscribe": 1},
                               on_message=on_message)
            self._subs[key] = sub
        try:
            self._send_subscription(sub)
            return sub.first_response.result(timeout or self.settings.request_timeout_s)
        except BaseException:
            self._drop_subscription(sub)
            raise

    def _send_subscription(self, sub: Subscription) -> None:
        req_id = next(self._req_ids)
        with self._state_lock:
            if sub.req_id is not None:
                self._subs_by_req.pop(sub.req_id, None)
            sub.req_id = req_id
            sub.first_response = Future()
            self._subs_by_req[req_id] = sub
        self._send_raw({**sub.request, "req_id": req_id})

    def _resubscribe_all(self) -> None:
        with self._state_lock:
            self._subs_by_stream.clear()
            subs = [s for s in self._subs.values() if s.active]
        for sub in subs:
            try:
                self._send_subscription(sub)
                log.info("Re-suscrito '%s'", sub.key)
            except Exception:
                log.exception("No se pudo re-suscribir '%s'", sub.key)

    def unsubscribe(self, key: str, wait: bool = True) -> None:
        """Da de baja un stream. ``wait=False`` es obligatorio si se llama desde un
        handler (hilo receptor): no se puede esperar una respuesta que ese mismo
        hilo tendría que leer."""
        with self._state_lock:
            sub = self._subs.get(key)
        if sub is None:
            return
        stream_id = sub.stream_id
        self._drop_subscription(sub)
        if not stream_id or not self.is_connected:
            return
        if wait and threading.current_thread() is not self._supervisor:
            try:
                self.request({"forget": stream_id})
            except Exception as exc:
                log.warning("forget %s falló: %s", key, exc)
        else:
            try:
                self._send_raw({"forget": stream_id, "req_id": next(self._req_ids)})
            except Exception as exc:
                log.warning("forget %s falló: %s", key, exc)

    def _drop_subscription(self, sub: Subscription) -> None:
        with self._state_lock:
            sub.active = False
            self._subs.pop(sub.key, None)
            if sub.req_id is not None:
                self._subs_by_req.pop(sub.req_id, None)
            if sub.stream_id:
                self._subs_by_stream.pop(sub.stream_id, None)

    def active_subscriptions(self) -> list[str]:
        with self._state_lock:
            return [k for k, s in self._subs.items() if s.active]

    def _fail_pending(self, exc: Exception) -> None:
        with self._state_lock:
            pending, self._pending = self._pending, {}
        for p in pending.values():
            if not p.future.done():
                p.future.set_exception(exc)
