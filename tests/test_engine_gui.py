"""Fase 6: TradingEngine integrado + GUI PySide6 (offscreen)."""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from broker.broker_interface import MarketDataBroker  # noqa: E402
from broker.simulated_broker import SimulatedBroker  # noqa: E402
from config import Settings  # noqa: E402
from core.event_manager import EventManager  # noqa: E402
from core.execution_controller import ExecutionController  # noqa: E402
from core.trading_engine import TradingEngine  # noqa: E402
from database.database import Database  # noqa: E402
from database.trades import TradeRepository  # noqa: E402
from market.market_data import SymbolInfo, Tick, TickIssue, TickValidationResult  # noqa: E402
from risk.risk_manager import RiskLimits, RiskManager  # noqa: E402
from strategy.fixed_time_strategy import FixedTimeStrategy, StrategyConfig  # noqa: E402
from strategy.signals import Direction  # noqa: E402
from strategy.strategy import Prediction  # noqa: E402
from tests.fakes import wait_until  # noqa: E402


class FakeMarket(MarketDataBroker):
    """Feed controlado: entrega ticks desde otro hilo, como el receptor WS real."""

    def __init__(self, sim: SimulatedBroker) -> None:
        self.sim = sim
        self.cb = None
        self.connected = False
        self.account = SimpleNamespace(account_id=sim.account_id, currency="USD")

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    @property
    def is_connected(self):
        return self.connected

    def get_active_symbols(self, refresh=False):
        return [SymbolInfo("R_100", "Volatility 100", "synthetic_index", "random", 0.01,
                           True, False)]

    def subscribe_ticks(self, symbol, on_tick):
        self.cb = on_tick

    def unsubscribe_ticks(self, symbol):
        self.cb = None

    def get_balance(self):
        return self.sim.ensure_account_valid()

    def push(self, quotes, start=None):
        start = start or int(time.time())

        def run():
            for i, q in enumerate(quotes):
                t = Tick("R_100", start + i, float(q), 2, time.time())
                self.sim.on_tick(t)
                if self.cb:
                    self.cb(t)
        th = threading.Thread(target=run)
        th.start()
        th.join()


class Always:
    model_version = "always-up"
    min_history = 3

    def __init__(self, p=0.8):
        self.p = p

    def predict(self, ticks):
        return Prediction(self.p, self.model_version, ticks[-1].epoch)


def make_engine(strategy=True, limits=None):
    sim = SimulatedBroker()
    sim.now = time.time()
    db = Database(":memory:")
    events = EventManager()
    risk = RiskManager(limits or RiskLimits(min_seconds_between_trades=0, max_daily_loss=100,
                                            max_open_contracts=1))
    ctl = ExecutionController(Settings(), sim, risk, repo=TradeRepository(db), events=events)
    market = FakeMarket(sim)
    strat = FixedTimeStrategy(Always(), StrategyConfig(threshold=0.6, duration=2,
                                                       cooldown_ticks=1)) if strategy else None
    eng = TradingEngine(Settings(), market, db=db, events=events, controller=ctl, strategy=strat)
    return eng, market, sim


@pytest.fixture
def engine():
    eng, market, sim = make_engine()
    eng.start()
    eng.watch_symbol("R_100")
    yield eng, market, sim
    eng.stop()


def test_engine_does_not_trade_until_auto_enabled(engine):
    eng, market, sim = engine
    market.push([100, 101, 102, 103, 104])
    assert wait_until(lambda: eng.session.get("ticks_processed") >= 5)
    assert sim.contracts == {}
    assert len(eng.predictions.recent()) > 0          # sí predice y registra


def test_engine_full_cycle_with_auto_trading(engine):
    eng, market, sim = engine
    eng.set_auto_trading(True)
    start = int(time.time())
    market.push([100, 101, 102, 103], start)
    assert wait_until(lambda: len(sim.contracts) >= 1)      # el worker ya compró
    market.push([104, 105, 106, 107], start + 4)            # ticks que liquidan
    assert wait_until(lambda: len(sim.settled) >= 1)
    assert wait_until(lambda: eng.trades.summary()["settled"] >= 1)
    acted = [p for p in eng.predictions.recent() if p["acted"]]
    assert acted and acted[0]["client_order_id"]
    assert eng.session.get("balance") is not None


def test_engine_blocks_trading_after_data_issue(engine):
    eng, market, sim = engine
    eng.set_auto_trading(True)
    eng.on_data_issue("R_100", TickValidationResult(None, False, TickIssue.PRICE_JUMP, "x"))
    market.push([100, 101, 102, 103, 104])
    assert wait_until(lambda: eng.session.get("ticks_processed") >= 5)
    assert sim.contracts == {} and not eng.session.get("market_ok")


def test_emergency_stop_disables_auto_and_requires_manual_rearm(engine):
    eng, market, sim = engine
    eng.set_auto_trading(True)
    eng.emergency_stop("prueba")
    assert not eng.session.get("auto_trading") and eng.controller.halted
    with pytest.raises(RuntimeError):
        eng.set_auto_trading(True)
    eng.resume_after_emergency()
    eng.set_auto_trading(True)


def test_manual_order_goes_through_controller_and_risk(engine):
    eng, market, sim = engine
    market.push([100, 101, 102])
    assert wait_until(lambda: eng.session.get("ticks_processed") >= 3)
    res = eng.manual_order(Direction.UP, 1.0, 2)
    assert res.status == "open"
    res2 = eng.manual_order(Direction.UP, 1.0, 2)          # max_open_contracts = 1
    assert res2.status == "rejected"
    assert eng.trades.order(res.client_order_id)["source"] == "manual"


def test_auto_trading_requires_strategy():
    eng, market, sim = make_engine(strategy=False)
    with pytest.raises(RuntimeError):
        eng.set_auto_trading(True)


def test_tick_callback_never_blocks_receive_thread(engine):
    """El callback del broker solo encola: debe volver al instante aunque la
    estrategia sea lenta."""
    eng, market, sim = engine

    class Slow(Always):
        def predict(self, ticks):
            time.sleep(0.2)
            return super().predict(ticks)
    eng.strategy.predictor = Slow()
    t0 = time.monotonic()
    market.push(list(range(100, 120)))
    assert time.monotonic() - t0 < 0.5


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_gui_builds_and_reflects_engine(qapp, engine):
    from interface.main_window import MainWindow
    eng, market, sim = engine
    w = MainWindow(eng)
    assert not w.settings.rb_real.isEnabled() and w.settings.rb_demo.isChecked()
    assert w.env_badge.text() == "DEMO"
    w.market.load_symbols()
    assert w.market.symbol_combo.count() == 1
    market.push([100, 101, 102, 103])
    assert wait_until(lambda: (qapp.processEvents(), eng.session.get("ticks_processed") >= 4)[1])
    for _ in range(5):
        qapp.processEvents()
        time.sleep(0.02)
    assert len(w.market.chart.values) >= 1
    w.refresh()
    w.history.refresh()
    w.ml.refresh()
    assert w.dashboard.btn_emergency.isEnabled()
    w.dashboard.btn_emergency.click()
    assert eng.controller.halted
    w.refresh()
    assert not w.dashboard.btn_emergency.isEnabled() and w.dashboard.btn_resume.isEnabled()
    w.close()


def test_gui_settings_apply_risk_limits(qapp, engine):
    from interface.settings_view import SettingsView
    eng, _, _ = engine
    v = SettingsView(eng)
    v.inputs["max_stake"].setValue(3.5)
    v.inputs["max_open_contracts"].setValue(4)
    v.apply()
    assert eng.controller.risk.limits.max_stake == 3.5
    assert eng.controller.risk.limits.max_open_contracts == 4
    assert eng.db.get_setting("risk_limits")["max_stake"] == 3.5


def test_gui_auto_toggle_reports_error_without_strategy(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from interface.dashboard import DashboardView
    eng, _, _ = make_engine(strategy=False)
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: shown.append(a[2]))
    d = DashboardView(eng)
    d._toggle_auto(True)
    assert shown and not d.btn_auto.isChecked()


# --------------------------------------------------------------------------
# Ensamblado real (bootstrap) contra el servidor Deriv falso
# --------------------------------------------------------------------------
def test_bootstrap_demo_end_to_end(monkeypatch):
    from broker.deriv_auth import DerivRestClient
    from core.bootstrap import build_engine
    from tests.fakes import FAST, FakeHttpSession, FakeServer, fake_secrets

    monkeypatch.delenv("DERIV_ACCOUNT_ID", raising=False)
    server, session, db = FakeServer(), FakeHttpSession(), Database(":memory:")
    db.set_setting("risk_limits", {"max_stake": 2.0, "min_seconds_between_trades": 0,
                                   "bogus_key": 1})
    rest = DerivRestClient("https://api.derivws.com", session=session,
                           secret_reader=fake_secrets)
    eng = build_engine(Settings(connection=FAST), db, demo=True, rest=rest,
                       connection_factory=server.factory)
    assert eng.controller.risk.limits.max_stake == 2.0          # límites persistidos
    eng.start()
    try:
        assert eng.session.get("account_id") == "DOT90004580"
        assert eng.session.get("balance") == 10000.0
        eng.watch_symbol("R_100")
        assert wait_until(lambda: eng.session.get("ticks_processed") >= 1)
        res = eng.manual_order(Direction.UP, 1.0, 5)
        assert res.status == "open"
        assert eng.manual_order(Direction.UP, 5.0, 5).status == "rejected"   # > max_stake
        server.portfolio = [{"contract_id": res.contract_id, "transaction_id": 1,
                             "underlying_symbol": "R_100", "contract_type": "CALL",
                             "buy_price": 1.0, "payout": 1.95, "purchase_time": 1}]
        rep = eng.reconciler.run()
        assert rep.ok, rep
        server.settle(res.contract_id, won=True)
        assert wait_until(lambda: eng.controller.is_settled(res.contract_id))
        assert eng.soak.report()["settled"] == 1
        # El adaptador bloqueado por cuenta cambiada provoca parada de emergencia.
        server.loginid = "DOT00000000"
        assert eng.manual_order(Direction.UP, 1.0, 5).status == "rejected"
        assert eng.controller.halted
    finally:
        eng.stop()
