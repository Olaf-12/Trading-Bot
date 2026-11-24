"""Fase 7: reconciliación local <-> broker y monitor de estabilidad."""

from __future__ import annotations

import pytest

from broker.broker_interface import RemoteContract
from broker.simulated_broker import SimulatedBroker
from config import Settings
from core import event_manager as ev
from core.event_manager import EventManager
from core.execution_controller import ExecutionController
from core.reconciliation import Reconciler
from core.soak import SoakMonitor
from database.database import Database
from database.trades import TradeRepository
from market.market_data import Tick
from risk.risk_manager import RiskLimits, RiskManager
from strategy.signals import ContractSpec, OrderRequest

T0 = 1_700_000_000


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


class LossyBroker(SimulatedBroker):
    """Simulador que puede 'perder' streams de seguimiento o confirmaciones de compra."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.drop_tracking = False
        self.timeout_after_buy = False
        self.foreign: list[RemoteContract] = []

    def track_contract(self, contract_id, on_update):
        if self.drop_tracking:
            return                                  # el stream nunca llega
        super().track_contract(contract_id, on_update)

    def buy(self, proposal, approval):
        receipt = super().buy(proposal, approval)
        if self.timeout_after_buy:
            raise TimeoutError("sin respuesta a buy")   # se compró, pero no lo sabemos
        return receipt

    def open_contracts(self):
        return super().open_contracts() + self.foreign


class Env:
    def __init__(self):
        self.clock = Clock()
        self.sim = LossyBroker()
        self.sim.now = self.clock.t
        self.db = Database(":memory:")
        self.events = EventManager()
        self.repo = TradeRepository(self.db, clock=self.sim.clock)
        self.risk = RiskManager(RiskLimits(min_seconds_between_trades=0, max_open_contracts=3,
                                           max_daily_loss=100), clock=self.sim.clock)
        self.ctl = ExecutionController(Settings(), self.sim, self.risk, repo=self.repo,
                                       events=self.events, clock=self.sim.clock)
        self.rec = Reconciler(self.sim, self.repo, self.ctl, clock=self.sim.clock,
                              unknown_grace_s=60)
        self.epoch = T0
        self.tick(1000)

    def tick(self, q):
        self.sim.on_tick(Tick("R_100", self.epoch, float(q), 2, self.epoch))
        self.epoch += 1

    def order(self, stake=1.0):
        return self.ctl.submit(OrderRequest(ContractSpec("R_100", "CALL", 2, "t", stake, "USD")))


@pytest.fixture
def env():
    return Env()


def test_clean_state_is_ok(env):
    env.order()
    rep = env.rec.run()
    assert rep.ok and rep.local_open == 1 and rep.remote_open == 1
    assert env.db.query("SELECT COUNT(*) n FROM reconciliation_runs")[0]["n"] == 1


def test_missed_settlement_is_recovered(env):
    env.sim.drop_tracking = True
    r = env.order()
    for q in (1001, 1002, 1003):
        env.tick(q)
    assert not env.ctl.is_settled(r.contract_id)
    assert env.risk.snapshot()["open_contracts"] == 1
    rep = env.rec.run()
    assert rep.settled_missed == [r.contract_id]
    assert env.ctl.is_settled(r.contract_id)
    assert env.risk.snapshot()["open_contracts"] == 0
    assert env.repo.trade(r.contract_id)["profit"] == pytest.approx(0.95)
    assert env.repo.order(r.client_order_id)["status"] == "settled"
    assert env.rec.run().settled_missed == []             # idempotente


def test_unknown_order_matched_to_remote_contract(env):
    env.sim.timeout_after_buy = True
    r = env.order()
    assert r.status == "unknown"
    rep = env.rec.run()
    assert rep.unknown_resolved == [r.client_order_id] and rep.ok
    o = env.repo.order(r.client_order_id)
    assert o["status"] == "open" and o["contract_id"] is not None
    env.sim.timeout_after_buy = False
    for q in (1001, 1002, 1003):
        env.tick(q)
    assert env.ctl.is_settled(o["contract_id"])            # seguimiento reanudado


def test_unknown_order_not_executed_releases_exposure(env):
    env.sim.timeout_after_buy = True
    r = env.order()
    env.sim.contracts.clear()                              # el broker NO lo ejecutó
    rep = env.rec.run()
    assert rep.unknown_not_executed == []                  # aún dentro del margen
    env.sim.now += 120
    rep = env.rec.run()
    assert rep.unknown_not_executed == [r.client_order_id]
    assert env.risk.snapshot()["open_contracts"] == 0
    assert env.repo.order(r.client_order_id)["status"] == "failed"


def test_foreign_contract_triggers_emergency_stop(env):
    halted = []
    env.events.subscribe(ev.EMERGENCY_STOP, lambda t, p: halted.append(p))
    env.sim.foreign.append(RemoteContract(777, 7770, "R_50", "PUT", 5.0, 9.5, T0))
    rep = env.rec.run()
    assert rep.unknown_remote == [777] and not rep.ok
    assert env.ctl.halted and halted


def test_profit_mismatch_corrected_to_broker_value(env):
    r = env.order()
    for q in (1001, 1002, 1003):
        env.tick(q)
    env.db.execute("UPDATE trades SET profit = 5 WHERE contract_id = ?", (r.contract_id,))
    rep = env.rec.run()
    assert rep.profit_corrected == [r.contract_id]
    assert env.repo.trade(r.contract_id)["profit"] == pytest.approx(0.95)


def test_reconciler_survives_broker_errors(env):
    def boom(*a, **k):
        raise ConnectionError("down")
    env.sim.open_contracts = boom
    rep = env.rec.run()
    assert rep.error and not rep.ok


def test_soak_monitor_report_and_criteria():
    clock = Clock(0)
    events = EventManager()
    mon = SoakMonitor(events, Database(":memory:"), connection_count=lambda: 3, clock=clock)
    events.publish(ev.TICK, {})
    clock.t = 5
    events.publish(ev.TICK, {})
    events.publish(ev.DATA_ISSUE, {"issue": "duplicate"})
    events.publish(ev.CONTRACT_SETTLED, {"profit": 0.95})
    events.publish(ev.CONTRACT_SETTLED, {"profit": -1.0})
    events.publish(ev.RECONCILIATION, {"ok": True})
    clock.t = 25 * 3600
    rep = mon.save()
    assert rep["reconnections"] == 2 and rep["ticks"] == 2 and rep["max_tick_gap_s"] == 5
    assert rep["pnl"] == pytest.approx(-0.05) and rep["data_issues"] == {"duplicate": 1}
    ok, problems = mon.passed(24)
    assert ok, problems
    events.publish(ev.RECONCILIATION, {"ok": False})
    assert not mon.passed(24)[0]
