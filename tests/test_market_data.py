"""Fase 1: active_symbols, ticks, validación, enrutado WS y reconexión.

Usa un servidor WebSocket falso en memoria: no requiere red ni credenciales.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from broker.deriv_adapter import DerivPublicAdapter
from broker.deriv_market import DerivMarketData, UnknownSymbolError, parse_active_symbols
from broker.deriv_ws_client import DerivAPIError, DerivWSClient
from config import (AccountEnvironment, RealModeLockedError, Settings,
                    TickValidationSettings, redact)
from database.database import Database, ForbiddenSettingError
from market.market_data import TickIssue, TickValidator
from tests.fakes import ACTIVE_SYMBOLS, FAST, FakeServer, tick_msg, wait_until

@pytest.fixture
def server():
    return FakeServer()


@pytest.fixture
def client(server):
    c = DerivWSClient(lambda: "wss://example.test/ws/public", FAST, server.factory)
    c.start()
    yield c
    c.stop()


# --------------------------------------------------------------------------
# active_symbols
# --------------------------------------------------------------------------
def test_parse_active_symbols_uses_official_fields():
    symbols = {s.underlying_symbol: s for s in parse_active_symbols(ACTIVE_SYMBOLS)}
    assert set(symbols) == {"R_100", "frxEURUSD"}
    assert symbols["R_100"].is_tradeable
    assert not symbols["frxEURUSD"].exchange_is_open
    assert symbols["R_100"].underlying_symbol_name == "Volatility 100 Index"


def test_parse_active_symbols_skips_malformed():
    msg = {"active_symbols": [{"underlying_symbol_name": "sin id"},
                              ACTIVE_SYMBOLS["active_symbols"][0]]}
    assert [s.underlying_symbol for s in parse_active_symbols(msg)] == ["R_100"]


def test_get_active_symbols_over_ws(client, server):
    md = DerivMarketData(client)
    symbols = md.get_active_symbols()
    assert {s.underlying_symbol for s in symbols} == {"R_100", "frxEURUSD"}
    sent = [m for m in server.received if "active_symbols" in m]
    assert sent[0]["active_symbols"] == "brief" and isinstance(sent[0]["req_id"], int)


def test_subscribe_unknown_symbol_rejected(client):
    md = DerivMarketData(client)
    with pytest.raises(UnknownSymbolError):
        md.subscribe_ticks("NOPE", lambda t: None)


# --------------------------------------------------------------------------
# Enrutado central por req_id / subscription.id / msg_type
# --------------------------------------------------------------------------
def test_responses_routed_by_req_id_even_out_of_order(client, server):
    server.auto_reply = False
    results: dict[str, Any] = {}

    def call(name, payload):
        results[name] = client.request(payload)

    t1 = threading.Thread(target=call, args=("a", {"active_symbols": "brief"}))
    t2 = threading.Thread(target=call, args=("b", {"ping": 1}))
    t1.start()
    assert wait_until(lambda: len(server.received) >= 1)
    t2.start()
    assert wait_until(lambda: len(server.received) >= 2)
    req_a = next(m for m in server.received if "active_symbols" in m)["req_id"]
    req_b = next(m for m in server.received if "ping" in m)["req_id"]
    # Ruido no solicitado + respuestas en orden inverso.
    server.conn.push({"msg_type": "website_status", "website_status": {}})
    server.conn.push({"msg_type": "ping", "ping": "pong", "req_id": req_b})
    server.conn.push({**ACTIVE_SYMBOLS, "req_id": req_a})
    t1.join(2), t2.join(2)
    assert results["a"]["msg_type"] == "active_symbols"
    assert results["b"]["ping"] == "pong"


def test_api_error_raised_on_matching_request(client, server):
    server.auto_reply = False
    out: dict[str, Any] = {}

    def call():
        try:
            client.request({"active_symbols": "brief"})
        except DerivAPIError as e:
            out["err"] = e

    t = threading.Thread(target=call)
    t.start()
    assert wait_until(lambda: server.received)
    rid = server.received[-1]["req_id"]
    server.conn.push({"msg_type": "active_symbols", "req_id": rid,
                      "error": {"code": "RateLimit", "message": "slow down"}})
    t.join(2)
    assert out["err"].code == "RateLimit"


def test_request_timeout(client, server):
    server.auto_reply = False
    with pytest.raises(TimeoutError):
        client.request({"active_symbols": "brief"}, timeout=0.1)


def test_tick_stream_routed_by_subscription_id(client, server):
    md = DerivMarketData(client)
    got = []
    md.subscribe_ticks("R_100", got.append)
    assert wait_until(lambda: len(got) == 1)
    sid = server.last_stream_id("R_100")
    now = int(time.time())
    # Mensajes de stream sin req_id: deben enrutarse por subscription.id.
    server.conn.push(tick_msg("R_100", now + 1, 1000.5, sid))
    server.conn.push(tick_msg("R_100", now + 2, 1001.0, sid))
    assert wait_until(lambda: len(got) == 3)
    assert [t.quote for t in got] == [1000.0, 1000.5, 1001.0]


def test_unsubscribe_sends_forget(client, server):
    md = DerivMarketData(client)
    md.subscribe_ticks("R_100", lambda t: None)
    sid = server.last_stream_id("R_100")
    md.unsubscribe_ticks("R_100")
    assert any(m.get("forget") == sid for m in server.received)
    assert client.active_subscriptions() == []


# --------------------------------------------------------------------------
# Validación de ticks
# --------------------------------------------------------------------------
class Clock:
    def __init__(self, t=1_700_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make_validator(clock=None, **kw):
    return TickValidator("R_100", TickValidationSettings(**kw), clock=clock or Clock())


def test_validator_accepts_normal_sequence():
    c = Clock()
    v = make_validator(c)
    e = int(c.t)
    assert v.validate({"symbol": "R_100", "epoch": e, "quote": 100.0}).accepted
    c.t += 1
    assert v.validate({"symbol": "R_100", "epoch": e + 1, "quote": 100.1}).accepted


@pytest.mark.parametrize("payload, issue", [
    ({"symbol": "R_100", "quote": 1.0}, TickIssue.MALFORMED),
    ({"symbol": "R_100", "epoch": "x", "quote": 1.0}, TickIssue.MALFORMED),
    ({"symbol": "R_100", "epoch": 1_700_000_000, "quote": "1"}, TickIssue.MALFORMED),
    ({"symbol": "R_50", "epoch": 1_700_000_000, "quote": 1.0}, TickIssue.SYMBOL_MISMATCH),
    ({"symbol": "R_100", "epoch": 1_700_000_000, "quote": 0}, TickIssue.INVALID_PRICE),
    ({"symbol": "R_100", "epoch": 1_700_000_000, "quote": float("nan")}, TickIssue.INVALID_PRICE),
    ({"symbol": "R_100", "epoch": 1_700_000_100, "quote": 1.0}, TickIssue.FUTURE_TIMESTAMP),
    ({"symbol": "R_100", "epoch": 1_699_999_900, "quote": 1.0}, TickIssue.STALE_QUOTE),
])
def test_validator_rejects_bad_ticks(payload, issue):
    r = make_validator().validate(payload)
    assert not r.accepted and r.issue is issue


def test_validator_duplicates_and_out_of_order():
    c = Clock()
    v = make_validator(c)
    e = int(c.t)
    v.validate({"symbol": "R_100", "epoch": e, "quote": 100.0})
    assert v.validate({"symbol": "R_100", "epoch": e, "quote": 100.0}).issue is TickIssue.DUPLICATE
    assert v.validate({"symbol": "R_100", "epoch": e - 1, "quote": 100.0}).issue \
        is TickIssue.OUT_OF_ORDER


def test_validator_flags_price_jump():
    c = Clock()
    v = make_validator(c, max_relative_jump=0.01)
    e = int(c.t)
    v.validate({"symbol": "R_100", "epoch": e, "quote": 100.0})
    r = v.validate({"symbol": "R_100", "epoch": e + 1, "quote": 110.0})
    assert not r.accepted and r.issue is TickIssue.PRICE_JUMP
    # El nuevo nivel pasa a ser la referencia: el siguiente tick normal se acepta.
    assert v.validate({"symbol": "R_100", "epoch": e + 2, "quote": 110.1}).accepted


def test_stale_stream_detection_and_forced_reconnect(client, server):
    clock = Clock(time.time())
    md = DerivMarketData(client, TickValidationSettings(stale_after_s=5,
                                                        reconnect_after_stale_s=10),
                         clock=clock)
    issues = []
    md._on_issue = lambda s, r: issues.append(r.issue)
    md.subscribe_ticks("R_100", lambda t: None)
    assert md.check_health() == []
    clock.t += 6
    assert md.check_health() == ["R_100"]
    assert issues[-1] is TickIssue.STALE_QUOTE
    assert server.conn.closed is False
    clock.t += 6
    md.check_health()
    assert wait_until(lambda: len(server.connections) == 2 and client.is_connected)


# --------------------------------------------------------------------------
# Reconexión
# --------------------------------------------------------------------------
def test_reconnect_resubscribes_and_fails_pending(client, server):
    md = DerivMarketData(client)
    got = []
    md.subscribe_ticks("R_100", got.append)
    assert wait_until(lambda: len(got) == 1)

    server.auto_reply = False
    errors = []

    def pending_call():
        try:
            client.request({"active_symbols": "brief"}, timeout=2)
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=pending_call)
    t.start()
    assert wait_until(lambda: any("active_symbols" in m for m in server.received[-1:]))
    server.auto_reply = True
    server.fail_next_connects = 2            # el servidor sigue caído 2 intentos
    server.conn.drop()
    t.join(3)
    assert errors and isinstance(errors[0], ConnectionError)

    assert wait_until(lambda: len(server.connections) == 2 and client.is_connected)
    ticks_requests = [m for m in server.received if m.get("ticks") == "R_100"]
    assert len(ticks_requests) == 2
    assert ticks_requests[0]["req_id"] != ticks_requests[1]["req_id"]
    # Nuevo stream tras reconectar entrega ticks al mismo callback.
    assert wait_until(lambda: len(got) >= 2)
    assert client.connection_count == 2
    assert len(server.urls) == 4             # 1 inicial + 2 fallidos + 1 éxito


def test_url_requested_fresh_on_each_connect(server):
    calls = []

    def provider():
        calls.append(1)
        return "wss://example.test/ws/public"

    c = DerivWSClient(provider, FAST, server.factory)
    c.start()
    server.conn.drop()
    assert wait_until(lambda: len(server.connections) == 2 and c.is_connected)
    c.stop()
    assert len(calls) == 2


def test_stop_is_clean(server):
    c = DerivWSClient(lambda: "wss://x", FAST, server.factory)
    c.start()
    c.stop()
    assert not c.is_connected and c.state == "stopped"
    time.sleep(0.1)
    assert len(server.connections) == 1


# --------------------------------------------------------------------------
# Adaptador + persistencia
# --------------------------------------------------------------------------
def test_public_adapter_persists_valid_ticks(server):
    db = Database(":memory:")
    s = Settings()
    adapter = DerivPublicAdapter(
        Settings(connection=FAST, ticks=s.ticks), db=db, connection_factory=server.factory)
    adapter.connect()
    try:
        got = []
        adapter.subscribe_ticks("R_100", got.append)
        assert wait_until(lambda: len(got) == 1)
        sid = server.last_stream_id("R_100")
        e = got[0].epoch
        server.conn.push(tick_msg("R_100", e, 1000.0, sid))          # duplicado
        server.conn.push(tick_msg("R_100", e + 1, 1000.2, sid))
        assert wait_until(lambda: len(got) == 2)
        assert wait_until(lambda: len(db.data_issues("R_100")) == 1)
        assert [r["quote"] for r in db.recent_ticks("R_100")] == [1000.0, 1000.2]
        assert db.data_issues("R_100")[0]["issue"] == "duplicate"
        assert any(r["event"] == "state:connected" for r in db.audit_events())
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------
# Seguridad (reglas no negociables)
# --------------------------------------------------------------------------
def test_demo_is_default_and_real_is_locked(monkeypatch):
    monkeypatch.setenv("TRADINGBOT_ENV", "real")
    from config import load_settings
    s = load_settings()
    assert s.environment is AccountEnvironment.DEMO
    with pytest.raises(RealModeLockedError):
        s.select_environment(AccountEnvironment.REAL)
    assert s.environment is AccountEnvironment.DEMO


def test_db_rejects_secrets_and_environment():
    db = Database(":memory:")
    for key in ("deriv_token", "OTP_url", "api_key", "password"):
        with pytest.raises(ForbiddenSettingError):
            db.set_setting(key, "x")
    with pytest.raises(ForbiddenSettingError):
        db.set_setting("environment", "real")
    db.set_setting("default_symbol", "R_100")
    assert db.get_setting("default_symbol") == "R_100"


def test_redaction_of_otp_urls_and_tokens():
    text = ("connecting wss://api.derivws.com/trading/v1/options/ws/demo?otp=abc123 "
            "Authorization: Bearer pat_SECRET {\"token\": \"s3cr3t\"}")
    out = redact(text)
    for secret in ("abc123", "pat_SECRET", "s3cr3t"):
        assert secret not in out


def test_audit_log_redacts_payload():
    db = Database(":memory:")
    db.log_event("auth", "otp", {"url": "wss://h/trading/v1/options/ws/demo?otp=ZZZ"})
    assert "ZZZ" not in db.audit_events()[0]["payload"]


def test_ticks_history_download_filters_duplicates(server):
    db = Database(":memory:")
    adapter = DerivPublicAdapter(Settings(connection=FAST), db=db,
                                 connection_factory=server.factory)
    adapter.connect()
    try:
        assert adapter.download_history("R_100", 20) == 19
        req = next(m for m in server.received if "ticks_history" in m)
        assert req["style"] == "ticks" and req["end"] == "latest"
        assert len(db.recent_ticks("R_100", 100)) == 19
        with pytest.raises(UnknownSymbolError):
            adapter.market.get_ticks_history("NOPE", 10)
    finally:
        adapter.disconnect()
