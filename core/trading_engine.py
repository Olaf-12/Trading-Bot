"""TradingEngine: coordina datos, predicción, estrategia, riesgo y ejecución.

- Los ticks llegan en el hilo receptor del WebSocket: el motor solo los ENCOLA.
  Un hilo de trabajo hace la inferencia y llama al ExecutionController (que hace
  peticiones bloqueantes); hacerlo en el hilo receptor provocaría un bloqueo.
- Un scheduler ejecuta tareas periódicas: salud de datos, saldo, reconciliación.
- La GUI solo invoca comandos del motor; nunca toca el broker ni la ejecución.
- El auto-trading arranca DESACTIVADO y se activa explícitamente.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Any, Callable

from broker.broker_interface import MarketDataBroker
from config import Settings
from core import event_manager as ev
from core.event_manager import EventManager
from core.execution_controller import ExecutionController, ExecutionResult
from core.trading_session import TradingSession
from database.database import Database
from database.model_registry import ModelRegistry
from database.predictions import PredictionRepository
from database.trades import TradeRepository
from market.market_data import Tick, TickIssue, TickValidationResult
from risk.risk_manager import RiskLimits
from strategy.signals import CONTRACT_FOR_DIRECTION, ContractSpec, Direction, OrderRequest
from strategy.strategy import Strategy

log = logging.getLogger(__name__)

# Tras estas anomalías, el motor no abre operaciones durante un tiempo.
BLOCKING_ISSUES = {TickIssue.PRICE_JUMP: 60.0, TickIssue.STALE_QUOTE: 30.0,
                   TickIssue.FUTURE_TIMESTAMP: 30.0, TickIssue.OUT_OF_ORDER: 15.0}


class _Periodic:
    def __init__(self, name: str, every_s: float, fn: Callable[[], None]) -> None:
        self.name, self.every_s, self.fn, self.next_at = name, every_s, fn, 0.0


class TradingEngine:
    def __init__(self, settings: Settings, market: MarketDataBroker, *,
                 db: Database, events: EventManager | None = None,
                 controller: ExecutionController | None = None,
                 strategy: Strategy | None = None, history: int = 600,
                 clock: Callable[[], float] = time.time) -> None:
        self.settings = settings
        self.market = market
        self.db = db
        self.events = events or (controller.events if controller else EventManager())
        self.controller = controller
        self.strategy = strategy
        self.trades = TradeRepository(db)
        self.predictions = PredictionRepository(db)
        self.registry = ModelRegistry(db)
        self._clock = clock
        mode = "demo" if controller is not None else "public"
        self.session = TradingSession(mode=mode, environment=settings.environment.value)
        self._window: deque[Tick] = deque(maxlen=history)
        self._queue: "queue.Queue[Tick | None]" = queue.Queue(maxsize=10_000)
        self._market_block_until = 0.0
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._scheduler: threading.Thread | None = None
        self._periodic: list[_Periodic] = []
        self._balance_fn: Callable[[], Any] | None = None
        self.reconciler = None
        self.soak = None
        self.autoload_model: str | None = None
        if controller is not None:
            controller.on_unknown_order = self._on_unknown_order
        self.events.subscribe(ev.EMERGENCY_STOP, self._on_emergency)
        self.events.subscribe(ev.CONTRACT_SETTLED, self._on_settled)
        self.events.subscribe(ev.ORDER_PLACED, self._on_placed)

    # ================================================================ ciclo de vida
    def start(self) -> None:
        """Conecta, arranca el hilo de trabajo y el scheduler (sin auto-trading)."""
        self._stop.clear()
        self.market.connect()
        self.session.update(connection="connected", status="running")
        acc = getattr(self.market, "account", None)
        if acc is not None:
            self.session.update(account_id=acc.account_id, currency=acc.currency)
            self._balance_fn = getattr(self.market, "get_balance", None)
        if self._balance_fn:
            try:
                self.refresh_balance()
            except Exception as exc:
                log.warning("Saldo inicial no disponible: %s", exc)
        if self.autoload_model:
            try:
                self.activate_model(self.autoload_model)
                log.info("Modelo activo cargado: %s", self.autoload_model)
            except Exception as exc:
                log.error("No se pudo cargar el modelo %s: %s", self.autoload_model, exc)
        self._worker = threading.Thread(target=self._work, name="engine-worker", daemon=True)
        self._worker.start()
        self.add_periodic("health", 2.0, self._health)
        if self._balance_fn:
            self.add_periodic("balance", 30.0, self.refresh_balance)
        self._scheduler = threading.Thread(target=self._schedule, name="engine-scheduler",
                                           daemon=True)
        self._scheduler.start()
        self._publish_state()

    def stop(self) -> None:
        self.set_auto_trading(False)
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        for t in (self._worker, self._scheduler):
            if t and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=5)
        self.market.disconnect()
        self.session.update(status="stopped", connection="disconnected")
        self._publish_state()

    def add_periodic(self, name: str, every_s: float, fn: Callable[[], None]) -> None:
        self._periodic.append(_Periodic(name, every_s, fn))

    def _schedule(self) -> None:
        while not self._stop.wait(0.5):
            now = time.monotonic()
            for p in self._periodic:
                if now >= p.next_at:
                    p.next_at = now + p.every_s
                    try:
                        p.fn()
                    except Exception:
                        log.exception("Tarea periódica '%s' falló", p.name)

    # ===================================================================== datos
    def available_symbols(self):
        return [s for s in self.market.get_active_symbols() if s.is_tradeable]

    def watch_symbol(self, symbol: str) -> None:
        current = self.session.get("symbol")
        if current == symbol:
            return
        if current:
            self.market.unsubscribe_ticks(current)
        self._window.clear()
        self.market.subscribe_ticks(symbol, self._enqueue)
        self.session.update(symbol=symbol)
        self._publish_state()

    def _enqueue(self, tick: Tick) -> None:
        """Llamado desde el hilo receptor: no bloquear."""
        try:
            self._queue.put_nowait(tick)
        except queue.Full:
            log.error("Cola de ticks llena: se descarta el tick %s", tick.epoch)

    def on_data_issue(self, symbol: str, result: TickValidationResult) -> None:
        if result.issue in BLOCKING_ISSUES:
            self._market_block_until = max(self._market_block_until,
                                           self._clock() + BLOCKING_ISSUES[result.issue])
            self.session.update(market_ok=False, market_issue=result.issue.value)
        self.events.publish(ev.DATA_ISSUE, {"symbol": symbol,
                                            "issue": result.issue.value if result.issue else "",
                                            "detail": result.detail})

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                tick = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if tick is None:
                break
            # Si hay acumulación, se incorporan todos al historial pero solo se
            # decide con el más reciente (no se opera con precios viejos).
            batch = [tick]
            while True:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._stop.set()
                    break
                batch.append(nxt)
            for t in batch:
                self._window.append(t)
                self.events.publish(ev.TICK, {"symbol": t.symbol, "epoch": t.epoch,
                                              "quote": t.quote, "pip_size": t.pip_size})
            try:
                self._process(batch[-1], len(batch))
            except Exception:
                log.exception("Error procesando tick")

    def _process(self, tick: Tick, n: int) -> None:
        self.session.incr("ticks_processed", n)
        self.session.update(last_quote=tick.quote, last_epoch=tick.epoch)
        now = self._clock()
        market_ok = now >= self._market_block_until
        if market_ok and not self.session.get("market_ok"):
            self.session.update(market_ok=True, market_issue="")
        if self.strategy is None:
            return
        decision = self.strategy.evaluate(tick.symbol, list(self._window),
                                          self.session.get("balance") or 0.0, now, market_ok)
        pred = decision.prediction
        pred_id = None
        if pred is not None:
            self.session.update(last_prediction=pred.prob_up, model_version=pred.model_version)
            direction = decision.order.signal.direction.value if decision.order else None
            pred_id = self.predictions.add(tick.symbol, pred.tick_epoch, pred.model_version,
                                           pred.prob_up, direction)
            self.events.publish(ev.PREDICTION, {"symbol": tick.symbol,
                                                "prob_up": pred.prob_up,
                                                "model_version": pred.model_version,
                                                "reason": decision.reason})
        if decision.order is None or not self.session.get("auto_trading"):
            return
        if self.controller is not None:
            # Filtro previo: evita enviar órdenes que el controlador rechazaría con seguridad
            # (la aprobación sigue siendo exclusiva del ExecutionController).
            ok, why = self.controller.risk.can_open_more()
            if not ok:
                log.debug("Señal omitida: %s", why)
                return
        self.session.incr("signals")
        self.events.publish(ev.SIGNAL, {"symbol": tick.symbol, "reason": decision.reason})
        result = self.submit(decision.order)
        if pred_id is not None and result is not None:
            self.predictions.mark_acted(pred_id, result.client_order_id)

    # ================================================================== órdenes
    def submit(self, order: OrderRequest) -> ExecutionResult | None:
        if self.controller is None:
            log.warning("Modo solo datos: no hay ExecutionController")
            return None
        return self.controller.submit(order)

    def manual_order(self, direction: Direction, stake: float, duration: int,
                     duration_unit: str = "t") -> ExecutionResult | None:
        """Orden manual desde la GUI: pasa por el MISMO controlador y riesgo."""
        symbol = self.session.get("symbol")
        currency = self.session.get("currency")
        if not symbol or not currency:
            raise RuntimeError("Selecciona un símbolo y conecta una cuenta Demo")
        spec = ContractSpec(symbol, CONTRACT_FOR_DIRECTION[direction], duration,
                            duration_unit, stake, currency)
        return self.submit(OrderRequest(spec, source="manual"))

    def set_auto_trading(self, enabled: bool) -> None:
        if enabled and (self.controller is None or self.strategy is None):
            raise RuntimeError("Auto-trading requiere cuenta Demo y un modelo activo")
        if enabled and self.controller.halted:
            raise RuntimeError("Parada de emergencia activa: rearma manualmente primero")
        self.session.update(auto_trading=enabled)
        self._publish_state()

    def emergency_stop(self, reason: str = "manual (GUI)") -> None:
        self.session.update(auto_trading=False)
        if self.controller:
            self.controller.emergency_stop(reason)
        self.session.update(status="halted", halt_reason=reason)
        self._publish_state()

    def resume_after_emergency(self) -> None:
        if self.controller:
            self.controller.resume()
        self.session.update(status="running", halt_reason="")
        self._publish_state()

    def update_risk_limits(self, limits: RiskLimits) -> None:
        if self.controller:
            self.controller.risk.update_limits(limits)
        self.db.set_setting("risk_limits", limits.__dict__)
        self.events.publish(ev.RISK, self.risk_snapshot())

    def risk_snapshot(self) -> dict[str, Any]:
        return self.controller.risk.snapshot() if self.controller else {}

    def set_strategy(self, strategy: Strategy | None) -> None:
        if self.session.get("auto_trading"):
            self.set_auto_trading(False)
        self.strategy = strategy

    def activate_model(self, version: str,
                       strategy_factory: Callable[[Any], Strategy] | None = None) -> None:
        """Activa un modelo APROBADO (hash verificado) y crea la estrategia."""
        from machine_learning.predict import ModelPredictor
        from strategy.fixed_time_strategy import FixedTimeStrategy, StrategyConfig

        meta = self.registry.get(version)
        if meta is None:
            raise KeyError(f"Modelo {version} no registrado")
        if not (meta.get("metrics") or {}).get("approved"):
            raise PermissionError("Modelo no aprobado por la validación walk-forward")
        predictor = ModelPredictor.from_file(meta["path"], meta["sha256"])
        if strategy_factory is not None:
            strategy = strategy_factory(predictor)
        else:
            params = meta.get("params") or {}
            strategy = FixedTimeStrategy(predictor, StrategyConfig(
                threshold=params.get("threshold", 0.56),
                expected_payout_ratio=params.get("payout_ratio", 1.95),
                duration=int(params.get("horizon", 5)), duration_unit="t",
                currency=self.session.get("currency") or "USD"))
        self.registry.activate(version)
        self.set_strategy(strategy)
        self.session.update(model_version=version)
        self._publish_state()

    # ============================================================ periódicas
    def _health(self) -> None:
        check = getattr(self.market, "check_health", None)
        if check:
            check()
        self.session.update(connection="connected" if self.market.is_connected
                            else "reconnecting")
        if self.controller:
            self.session.update(orders_open=len(self.controller.open_contract_ids()))
            self.events.publish(ev.RISK, self.risk_snapshot())
        self._publish_state()

    def refresh_balance(self) -> None:
        if not self._balance_fn:
            return
        bal = self._balance_fn()
        self.session.update(balance=bal.balance, currency=bal.currency)
        self.events.publish(ev.BALANCE, {"balance": bal.balance, "currency": bal.currency})

    # ================================================================ eventos
    def _on_emergency(self, _t: str, p: dict) -> None:
        self.session.update(auto_trading=False, status="halted", halt_reason=p.get("reason", ""))
        self._publish_state()

    def _on_settled(self, _t: str, p: dict) -> None:
        if self.strategy:
            self.strategy.on_settled(p.get("profit", 0.0))
        threading.Thread(target=self._safe_refresh_balance, daemon=True).start()

    def _on_placed(self, _t: str, p: dict) -> None:
        if "balance_after" in p:
            self.session.update(balance=p["balance_after"])

    def _safe_refresh_balance(self) -> None:
        try:
            self.refresh_balance()
        except Exception as exc:
            log.debug("refresh_balance: %s", exc)

    def _on_unknown_order(self, order_id: str) -> None:
        if self.reconciler is not None:
            threading.Thread(target=self.reconciler.run_safe, daemon=True).start()

    def _publish_state(self) -> None:
        self.events.publish(ev.ENGINE_STATE, self.session.snapshot())
