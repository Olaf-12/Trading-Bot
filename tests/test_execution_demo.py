"""Fase 3: proposal -> buy -> proposal_open_contract a través de ExecutionController."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from broker.deriv_adapter import DerivDemoAdapter
from broker.deriv_execution import parse_buy, parse_proposal
from config import Settings
from core import event_manager as ev
from core.event_manager import EventManager
from core.execution_controller import ExecutionController
from core.order_approval import ApprovalError, OrderApproval
from database.database import Database
from database.trades import TradeRepository
from risk.risk_manager import RiskLimits, RiskManager
from strategy.signals import ContractSpec, Direction, OrderRequest, Signal
from tests.fakes import FAST, FakeHttpSession, FakeServer, fake_secrets, wait_until

from broker.deriv_auth import DerivRestClient

ROOT = Path(__file__).resolve().parents[1]
LIMITS = RiskLimits(max_stake=5, max_daily_loss=10, max_open_contracts=2,
                    min_seconds_between_trades=0, max_consecutive_errors=3)


@pytest.fixture(autouse=True)
def _no_env_account(monkeypatch):
    monkeypatch.delenv("DERIV_ACCOUNT_ID", raising=False)


class Env:
    def __init__(self, limits=LIMITS):
        self.server = FakeServer()
        self.session = FakeHttpSession()
        self.db = Database(":memory:")
        self.repo = TradeRepository(self.db)
        self.events = EventManager()
        self.seen: list[tuple[str, dict]] = []
        self.events.subscribe(ev.ALL, lambda t, p: self.seen.append((t, p)))
        self.adapter = DerivDemoAdapter(
            Settings(connection=FAST), db=self.db,
            rest=DerivRestClient("https://api.derivws.com", session=self.session,
                                 secret_reader=fake_secrets),
            connection_factory=self.server.factory)
        self.adapter.connect()
        self.risk = RiskManager(limits)
        self.ctl = ExecutionController(self.adapter.settings, self.adapter.execution,
                                       self.risk, repo=self.repo, events=self.events)

    def topics(self):
        return [t for t, _ in self.seen]

    def close(self):
        self.adapter.disconnect()


@pytest.fixture
def env():
    e = Env()
    yield e
    e.close()


def order(stake=1.0, direction=Direction.UP, currency="USD"):
    sig = Signal("R_100", direction, 0.61, "xgb-test", 1_700_000_000)
    return OrderRequest(ContractSpec("R_100", "CALL" if direction is Direction.UP else "PUT",
                                     5, "t", stake, currency), sig)


# --------------------------------------------------------------------------
def test_full_flow_win(env):
    res = env.ctl.submit(order())
    assert res.status == "open" and res.contract_id
    bought = env.server.bought[0]
    assert bought["price"] == res.proposal.ask_price          # precio máximo = ask aprobado
    req = next(m for m in env.server.received if m.get("proposal") == 1)
    assert req["underlying_symbol"] == "R_100" and req["basis"] == "stake"
    assert env.risk.snapshot()["open_contracts"] == 1

    env.server.settle(res.contract_id, won=True)
    assert wait_until(lambda: env.ctl.is_settled(res.contract_id))
    t = env.repo.trade(res.contract_id)
    assert t["status"] == "won" and t["profit"] == pytest.approx(0.95)
    assert t["model_version"] == "xgb-test"
    o = env.repo.order(res.client_order_id)
    assert o["status"] == "settled" and o["signal_prob"] == 0.61
    snap = env.risk.snapshot()
    assert snap["open_contracts"] == 0 and snap["realized_pnl"] == pytest.approx(0.95)
    assert ev.ORDER_PLACED in env.topics() and ev.CONTRACT_SETTLED in env.topics()
    # Tras liquidar se da de baja el stream poc con forget.
    assert wait_until(lambda: any(m.get("forget") == env.server.poc_streams[res.contract_id]
                                  for m in env.server.received))


def test_loss_updates_daily_pnl(env):
    res = env.ctl.submit(order(stake=2))
    env.server.settle(res.contract_id, won=False, buy_price=2)
    assert wait_until(lambda: env.ctl.is_settled(res.contract_id))
    assert env.risk.snapshot()["realized_pnl"] == pytest.approx(-2)


def test_risk_rejects_before_any_proposal(env):
    res = env.ctl.submit(order(stake=50))
    assert res.status == "rejected" and "stake" in res.reasons[0]
    assert not any(m.get("proposal") == 1 for m in env.server.received)
    assert env.repo.order(res.client_order_id)["status"] == "rejected"


def test_currency_mismatch_rejected(env):
    res = env.ctl.submit(order(currency="EUR"))
    assert res.status == "rejected" and "moneda" in res.reasons[0]


def test_low_payout_rejected_without_buy(env):
    env.server.payout_ratio = 1.5
    res = env.ctl.submit(order())
    assert res.status == "rejected" and "payout" in res.reasons[0]
    assert env.server.bought == []


def test_price_slippage_rejected(env):
    env.server.ask_factor = 1.10
    res = env.ctl.submit(order())
    assert res.status == "rejected" and "ask_price" in res.reasons[0]
    assert env.server.bought == []


def test_stale_proposal_rejected(env):
    env.server.proposal_age = 60
    res = env.ctl.submit(order())
    assert res.status == "rejected" and "obsoleta" in res.reasons[0]


def test_buy_rejected_releases_exposure(env):
    env.server.buy_mode = "reject"
    res = env.ctl.submit(order())
    assert res.status == "failed"
    assert env.risk.snapshot()["open_contracts"] == 0
    assert env.risk.state.consecutive_errors == 1


def test_buy_timeout_is_unknown_and_never_retried(env):
    env.adapter.client.settings = FAST.__class__(**{**FAST.__dict__, "request_timeout_s": 0.2})
    env.server.buy_mode = "timeout"
    unknown = []
    env.ctl.on_unknown_order = unknown.append
    res = env.ctl.submit(order())
    assert res.status == "unknown" and unknown == [res.client_order_id]
    assert sum("buy" in m for m in env.server.received) == 1          # sin reintento
    assert env.risk.snapshot()["open_contracts"] == 1                 # exposición se mantiene
    assert env.repo.order(res.client_order_id)["status"] == "unknown"


def test_max_open_contracts_enforced(env):
    assert env.ctl.submit(order()).status == "open"
    assert env.ctl.submit(order()).status == "open"
    res = env.ctl.submit(order())
    assert res.status == "rejected" and "contratos abiertos" in res.reasons[0]


def test_daily_loss_and_exposure_limit():
    e = Env(RiskLimits(max_stake=5, max_daily_loss=4, max_open_contracts=5,
                       min_seconds_between_trades=0))
    try:
        r1 = e.ctl.submit(order(stake=3))
        assert r1.status == "open"
        r2 = e.ctl.submit(order(stake=3))           # 3 + 3 > 4 de pérdida potencial
        assert r2.status == "rejected" and "exposición" in " ".join(r2.reasons)
        e.server.settle(r1.contract_id, won=False, buy_price=3)
        assert wait_until(lambda: e.ctl.is_settled(r1.contract_id))
        r3 = e.ctl.submit(order(stake=1))           # 3 perdidos + 1 = 4: aún permitido
        assert r3.status == "open"
        e.server.settle(r3.contract_id, won=False, buy_price=1)
        assert wait_until(lambda: e.ctl.is_settled(r3.contract_id))
        r4 = e.ctl.submit(order(stake=1))
        assert r4.status == "rejected" and "pérdida diaria" in r4.reasons[0]
    finally:
        e.close()


def test_contract_from_other_account_triggers_emergency_stop(env):
    res = env.ctl.submit(order())
    env.server.poc_account = "DOT99999999"
    env.server.settle(res.contract_id, won=True)
    assert wait_until(lambda: env.ctl.halted)
    assert "DOT99999999" in env.ctl.halt_reason
    assert env.ctl.submit(order()).status == "rejected"
    assert ev.EMERGENCY_STOP in env.topics()


def test_emergency_stop_blocks_new_orders_but_keeps_tracking(env):
    res = env.ctl.submit(order())
    env.ctl.emergency_stop("manual")
    r2 = env.ctl.submit(order())
    assert r2.status == "rejected" and "emergencia" in r2.reasons[0]
    assert env.ctl.open_contract_ids() == [res.contract_id]
    env.server.settle(res.contract_id, won=True)
    assert wait_until(lambda: env.ctl.is_settled(res.contract_id))
    env.ctl.resume()
    assert env.ctl.submit(order()).status == "open"


def test_account_mismatch_at_submit_halts(env):
    env.server.loginid = "DOT00000001"
    res = env.ctl.submit(order())
    assert res.status == "rejected"
    assert env.ctl.halted and env.adapter.locked


def test_consecutive_errors_pause(env):
    env.server.proposal_error = "bad"
    for _ in range(3):
        assert env.ctl.submit(order()).status == "failed"
    env.server.proposal_error = None
    res = env.ctl.submit(order())
    assert res.status == "rejected" and "pausa" in res.reasons[0]


# --------------------------------------------------------------------------
# Aprobación: único punto autorizado
# --------------------------------------------------------------------------
def test_buy_without_controller_approval_is_refused(env):
    execution = env.adapter.execution
    proposal = execution.get_proposal(order().spec)
    with pytest.raises(ApprovalError):
        execution.buy(proposal, None)
    with pytest.raises(ApprovalError):
        OrderApproval(key=object(), client_order_id="x", proposal_id=proposal.id,
                      max_price=proposal.ask_price, account_id=execution.account_id)
    assert env.server.bought == []


def test_only_core_imports_execution_or_mint_key():
    """Ningún módulo de interface/, strategy/, machine_learning/ o market/ puede
    importar el módulo de ejecución ni la clave de aprobación."""
    forbidden = {"broker.deriv_execution", "core.order_approval"}
    offenders = []
    for folder in ("interface", "strategy", "machine_learning", "market", "risk"):
        for py in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                if any(n in forbidden or n.endswith("_MINT_KEY") for n in names):
                    offenders.append(str(py.relative_to(ROOT)))
    assert offenders == []


def test_parsers_reject_malformed():
    spec = order().spec
    with pytest.raises(ValueError):
        parse_proposal({"proposal": {"id": "x"}}, spec)
    with pytest.raises(ValueError):
        parse_buy({"buy": {"contract_id": 1}})
