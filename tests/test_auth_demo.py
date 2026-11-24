"""Fase 2: autenticación Demo (PAT -> OTP -> WS) y saldo Demo.

Sin red ni credenciales reales: REST y WebSocket son dobles en memoria.
"""

from __future__ import annotations

import logging

import pytest

from broker.deriv_adapter import (AccountMismatchError, AdapterLockedError, DerivDemoAdapter,
                                  parse_balance)
from broker.deriv_auth import (AccountValidationError, DerivAuthError, DerivRestClient,
                               InvalidOtpUrlError, OptionsAccount, OtpUrlProvider,
                               select_demo_account, validate_otp_url)
from config import AccountEnvironment, RealModeLockedError, SecretNotFoundError, Settings
from database.database import Database
from tests.fakes import (DEMO_ACCOUNT, FAST, REAL_ACCOUNT, FakeHttpSession, FakeResponse,
                         FakeServer, fake_secrets, wait_until)

OTP_SECRET_PREFIX = "SECRETOTP"
TOKEN = "pat_FAKE_TOKEN_123"


def rest_client(session: FakeHttpSession) -> DerivRestClient:
    return DerivRestClient("https://api.derivws.com", session=session,
                           secret_reader=fake_secrets)


def make_adapter(server: FakeServer, session: FakeHttpSession, db=None, account_id=None):
    return DerivDemoAdapter(Settings(connection=FAST), account_id, db=db,
                            rest=rest_client(session), connection_factory=server.factory)


@pytest.fixture(autouse=True)
def _no_env_account(monkeypatch):
    monkeypatch.delenv("DERIV_ACCOUNT_ID", raising=False)


# --------------------------------------------------------------------------
# REST
# --------------------------------------------------------------------------
def test_rest_sends_bearer_and_app_id_without_redirects():
    s = FakeHttpSession()
    accounts = rest_client(s).list_accounts()
    assert accounts[0] == OptionsAccount.from_api(DEMO_ACCOUNT)
    call = s.calls[0]
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert call["headers"]["Deriv-App-ID"] == "12345"
    assert call["allow_redirects"] is False
    assert call["timeout"] > 0


def test_rest_requires_https():
    with pytest.raises(ValueError):
        DerivRestClient("http://api.derivws.com", session=FakeHttpSession())


def test_rest_error_schema_parsed_and_redacted():
    s = FakeHttpSession()
    s.next_error = FakeResponse(401, {"errors": [{"status": 401, "code": "Unauthorized",
                                                  "message": f"bad token {TOKEN}"}],
                                      "meta": {}})
    with pytest.raises(DerivAuthError) as ei:
        rest_client(s).list_accounts()
    assert ei.value.status == 401 and ei.value.code == "Unauthorized"
    # Si el servidor refleja el token en el mensaje, no debe aparecer en la excepción.
    assert "Unauthorized" in str(ei.value) and TOKEN not in str(ei.value)


def test_rest_non_json_error():
    s = FakeHttpSession()
    s.next_error = FakeResponse(502, ValueError("not json"))
    with pytest.raises(DerivAuthError) as ei:
        rest_client(s).list_accounts()
    assert ei.value.status == 502


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("DERIV_TOKEN", raising=False)
    monkeypatch.setattr("keyring.get_password", lambda *a: None)
    client = DerivRestClient("https://api.derivws.com", session=FakeHttpSession())
    with pytest.raises(SecretNotFoundError):
        client.list_accounts()


def test_rest_client_repr_and_attrs_hold_no_secret():
    c = rest_client(FakeHttpSession())
    c.list_accounts()
    assert TOKEN not in repr(c)
    assert all(TOKEN not in str(v) for v in vars(c).values())


# --------------------------------------------------------------------------
# Selección de cuenta
# --------------------------------------------------------------------------
def accounts(*raw):
    return [OptionsAccount.from_api(a) for a in raw]


def test_auto_selects_single_demo_never_real():
    acc = select_demo_account(accounts(REAL_ACCOUNT, DEMO_ACCOUNT))
    assert acc.account_id == DEMO_ACCOUNT["account_id"]


def test_explicit_real_account_is_locked():
    with pytest.raises(RealModeLockedError):
        select_demo_account(accounts(REAL_ACCOUNT, DEMO_ACCOUNT), REAL_ACCOUNT["account_id"])


def test_only_real_account_fails():
    with pytest.raises(AccountValidationError):
        select_demo_account(accounts(REAL_ACCOUNT))


def test_multiple_demos_require_explicit_choice():
    other = {**DEMO_ACCOUNT, "account_id": "DOT90009999"}
    with pytest.raises(AccountValidationError):
        select_demo_account(accounts(DEMO_ACCOUNT, other))
    assert select_demo_account(accounts(DEMO_ACCOUNT, other), "DOT90009999").account_id \
        == "DOT90009999"


def test_unknown_or_inactive_account_rejected():
    with pytest.raises(AccountValidationError):
        select_demo_account(accounts(DEMO_ACCOUNT), "DOT00000000")
    inactive = {**DEMO_ACCOUNT, "status": "inactive"}
    with pytest.raises(AccountValidationError):
        select_demo_account(accounts(inactive), DEMO_ACCOUNT["account_id"])
    with pytest.raises(AccountValidationError):
        select_demo_account(accounts(inactive))


# --------------------------------------------------------------------------
# Validación de la URL OTP
# --------------------------------------------------------------------------
GOOD = "wss://api.derivws.com/trading/v1/options/ws/demo?otp=abc"


@pytest.mark.parametrize("url, exc", [
    ("ws://api.derivws.com/trading/v1/options/ws/demo?otp=abc", InvalidOtpUrlError),
    ("wss://evil.example.com/trading/v1/options/ws/demo?otp=abc", InvalidOtpUrlError),
    ("wss://api.derivws.com/trading/v1/options/ws/public?otp=abc", InvalidOtpUrlError),
    ("wss://api.derivws.com/trading/v1/options/ws/demo", InvalidOtpUrlError),
    ("wss://api.derivws.com/trading/v1/options/ws/real?otp=abc", RealModeLockedError),
])
def test_bad_otp_urls_rejected_without_leaking(url, exc):
    with pytest.raises(exc) as ei:
        validate_otp_url(url)
    assert "abc" not in str(ei.value)


def test_good_otp_url_and_real_env_locked():
    assert validate_otp_url(GOOD) == GOOD
    with pytest.raises(RealModeLockedError):
        validate_otp_url(GOOD, AccountEnvironment.REAL)
    with pytest.raises(RealModeLockedError):
        OtpUrlProvider(rest_client(FakeHttpSession()), "DOT1", AccountEnvironment.REAL)


def test_otp_provider_requests_fresh_otp_each_call():
    s = FakeHttpSession()
    p = OtpUrlProvider(rest_client(s), DEMO_ACCOUNT["account_id"])
    u1, u2 = p(), p()
    assert u1 != u2 and s.otp_count == 2
    assert s.calls[0]["url"].endswith(f"/accounts/{DEMO_ACCOUNT['account_id']}/otp")
    assert OTP_SECRET_PREFIX not in repr(p)


# --------------------------------------------------------------------------
# Adaptador Demo de extremo a extremo
# --------------------------------------------------------------------------
def test_demo_connect_and_balance(caplog):
    caplog.set_level(logging.DEBUG)
    server, session, db = FakeServer(), FakeHttpSession(), Database(":memory:")
    adapter = make_adapter(server, session, db)
    adapter.connect()
    try:
        bal = adapter.get_balance()
        assert (bal.balance, bal.currency, bal.loginid) == (10000.0, "USD", "DOT90004580")
        assert adapter.account.account_type == "demo"
        assert server.urls[0].startswith("wss://api.derivws.com/trading/v1/options/ws/demo?otp=")
        events = [r["event"] for r in db.audit_events()]
        assert "account_selected" in events and "account_verified" in events
    finally:
        adapter.disconnect()
    # La URL OTP / OTP nunca aparece en logs, BD ni en la representación del adaptador.
    assert OTP_SECRET_PREFIX not in caplog.text and TOKEN not in caplog.text
    for row in db.audit_events():
        assert OTP_SECRET_PREFIX not in (row["payload"] or "")
        assert TOKEN not in (row["payload"] or "")


def test_fresh_otp_on_reconnect_and_account_reverified():
    server, session = FakeServer(), FakeHttpSession()
    adapter = make_adapter(server, session)
    adapter.connect()
    try:
        server.conn.drop()
        assert wait_until(lambda: len(server.connections) == 2 and adapter.is_connected)
        assert session.otp_count == 2
        assert len(set(server.urls)) == 2                  # OTP nunca reutilizado
        assert wait_until(lambda: sum("balance" in m for m in server.received) >= 2)
        assert not adapter.locked
    finally:
        adapter.disconnect()


def test_loginid_mismatch_on_connect_locks_adapter():
    server, session = FakeServer(), FakeHttpSession()
    server.loginid = "DOT11111111"
    reasons = []
    adapter = make_adapter(server, session)
    adapter._on_locked = reasons.append
    with pytest.raises(AccountMismatchError):
        adapter.connect()
    assert adapter.locked and reasons
    assert not adapter.is_connected
    with pytest.raises(AdapterLockedError):
        adapter.connect()
    with pytest.raises(AdapterLockedError):
        adapter.get_balance()


def test_mismatch_after_reconnect_locks_adapter():
    server, session = FakeServer(), FakeHttpSession()
    adapter = make_adapter(server, session)
    adapter.connect()
    try:
        server.loginid = "DOT22222222"                     # el servidor "cambia" de cuenta
        server.conn.drop()
        assert wait_until(lambda: adapter.locked, timeout=3)
        assert "no coincide" in adapter.locked_reason
        assert wait_until(lambda: not adapter.is_connected)
    finally:
        adapter.disconnect()


def test_currency_mismatch_locks_adapter():
    server, session = FakeServer(), FakeHttpSession()
    server.currency = "EUR"
    adapter = make_adapter(server, session)
    with pytest.raises(AccountMismatchError):
        adapter.connect()
    assert adapter.locked


def test_otp_url_pointing_to_real_is_fatal_no_retry():
    server = FakeServer()
    session = FakeHttpSession(ws_path="/trading/v1/options/ws/real")
    adapter = make_adapter(server, session)
    with pytest.raises(RealModeLockedError):
        adapter.connect()
    assert server.connections == [] and session.otp_count == 1


def test_unauthorized_otp_is_fatal_no_retry():
    server, session = FakeServer(), FakeHttpSession()
    adapter = make_adapter(server, session)
    orig_post = session.post

    def post(url, **kw):
        session.next_error = FakeResponse(401, {"errors": [{"status": 401,
                                                            "code": "Unauthorized",
                                                            "message": "x"}], "meta": {}})
        return orig_post(url, **kw)

    session.post = post
    with pytest.raises(DerivAuthError):
        adapter.connect()
    assert server.connections == []
    assert sum(c["method"] == "POST" for c in session.calls) == 1


def test_demo_adapter_refuses_real_account_id():
    server = FakeServer()
    session = FakeHttpSession(accounts=[REAL_ACCOUNT, DEMO_ACCOUNT])
    adapter = make_adapter(server, session, account_id=REAL_ACCOUNT["account_id"])
    with pytest.raises(RealModeLockedError):
        adapter.connect()
    assert session.otp_count == 0 and server.connections == []


def test_account_id_from_env(monkeypatch):
    other = {**DEMO_ACCOUNT, "account_id": "DOT90009999"}
    monkeypatch.setenv("DERIV_ACCOUNT_ID", "DOT90009999")
    server, session = FakeServer(), FakeHttpSession(accounts=[DEMO_ACCOUNT, other])
    server.loginid = "DOT90009999"
    adapter = make_adapter(server, session)
    adapter.connect()
    try:
        assert adapter.account.account_id == "DOT90009999"
        assert session.calls[1]["url"].endswith("/accounts/DOT90009999/otp")
    finally:
        adapter.disconnect()


def test_demo_adapter_ticks_still_work():
    server, session = FakeServer(), FakeHttpSession()
    adapter = make_adapter(server, session)
    adapter.connect()
    try:
        got = []
        adapter.subscribe_ticks("R_100", got.append)
        assert wait_until(lambda: len(got) == 1)
    finally:
        adapter.disconnect()


def test_parse_balance_rejects_malformed():
    with pytest.raises(ValueError):
        parse_balance({"msg_type": "balance"})
    with pytest.raises(ValueError):
        parse_balance({"balance": {"balance": 1}})
