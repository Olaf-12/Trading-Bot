"""Ensamblado de la aplicación: crea adaptador, riesgo, controlador, estrategia,
reconciliación y motor con el cableado correcto. Lo usan main.py y la GUI."""

from __future__ import annotations

from dataclasses import fields

from broker.deriv_adapter import DerivDemoAdapter, DerivPublicAdapter
from config import Settings
from core import event_manager as ev
from core.event_manager import EventManager
from core.execution_controller import ExecutionController
from core.reconciliation import Reconciler
from core.soak import SoakMonitor
from core.trading_engine import TradingEngine
from database.database import Database
from database.trades import TradeRepository
from risk.risk_manager import RiskLimits, RiskManager

RECONCILE_EVERY_S = 300.0


def load_risk_limits(db: Database) -> RiskLimits:
    saved = db.get_setting("risk_limits") or {}
    valid = {f.name for f in fields(RiskLimits)}
    try:
        return RiskLimits(**{k: v for k, v in saved.items() if k in valid})
    except TypeError:
        return RiskLimits()


def build_engine(settings: Settings, db: Database, *, demo: bool,
                 account_id: str | None = None, **adapter_kw) -> TradingEngine:
    events = EventManager()
    if not demo:
        adapter = DerivPublicAdapter(settings, db=db, **adapter_kw)
        engine = TradingEngine(settings, adapter, db=db, events=events)
        adapter._user_on_issue = engine.on_data_issue
        adapter._user_on_state = lambda s: events.publish(ev.CONNECTION, {"state": s})
        return engine

    adapter = DerivDemoAdapter(settings, account_id, db=db, **adapter_kw)
    risk = RiskManager(load_risk_limits(db))
    repo = TradeRepository(db)
    controller = ExecutionController(settings, adapter.execution, risk, repo=repo,
                                     events=events)
    engine = TradingEngine(settings, adapter, db=db, events=events, controller=controller)
    adapter._user_on_issue = engine.on_data_issue
    adapter._user_on_state = lambda s: events.publish(ev.CONNECTION, {"state": s})
    adapter._on_locked = lambda reason: controller.emergency_stop(f"adaptador: {reason}")
    engine.reconciler = Reconciler(adapter.execution, repo, controller)
    engine.add_periodic("reconcile", RECONCILE_EVERY_S, engine.reconciler.run_safe)
    engine.soak = SoakMonitor(events, db, connection_count=lambda: adapter.client.connection_count)

    active = engine.registry.active()
    if active:
        engine.autoload_model = active["version"]   # se carga en start(), con la moneda real
    return engine
