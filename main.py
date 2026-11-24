"""TradingBotML — punto de entrada.

    python main.py                         # GUI, solo datos públicos
    python main.py gui --demo              # GUI con cuenta Demo (DERIV_TOKEN/DERIV_APP_ID)
    python main.py symbols
    python main.py ticks --symbol R_100 --seconds 60 [--demo]
    python main.py balance                 # cuenta Demo + saldo
    python main.py download --symbol R_100 --count 5000 [--pages 10]
    python main.py train --symbol R_100 --horizon 5 [--activate]
    python main.py backtest --symbol R_100 [--model VERSION] [--payout 1.95]
    python main.py soak --symbol R_100 --hours 24     # Demo prolongado (Fase 7)

No existe ninguna opción para el entorno Real: está bloqueado (Fase 8).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from broker.deriv_auth import DerivAuthError
from config import SecretNotFoundError, load_settings, setup_logging
from database.database import Database

log = logging.getLogger("tradingbot")


# ----------------------------------------------------------------------------
def cmd_gui(args, settings, db) -> int:
    from PySide6.QtWidgets import QApplication, QMessageBox

    from core.bootstrap import build_engine
    from interface.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    engine = build_engine(settings, db, demo=args.demo, account_id=args.account_id)
    try:
        engine.start()
    except Exception as exc:
        QMessageBox.critical(None, "TradingBotML", f"No se pudo conectar: {exc}")
        return 1
    win = MainWindow(engine)
    win.show()
    try:
        return app.exec()
    finally:
        engine.stop()


def cmd_symbols(args, settings, db) -> int:
    from broker.deriv_adapter import DerivPublicAdapter

    adapter = DerivPublicAdapter(settings, db=db)
    adapter.connect()
    try:
        for s in sorted(adapter.get_active_symbols(), key=lambda s: (s.market,
                                                                     s.underlying_symbol)):
            flag = "abierto" if s.is_tradeable else "cerrado/suspendido"
            print(f"{s.underlying_symbol:<14} {s.market:<16} {flag:<20} "
                  f"{s.underlying_symbol_name}")
    finally:
        adapter.disconnect()
    return 0


def cmd_ticks(args, settings, db) -> int:
    from broker.deriv_adapter import DerivDemoAdapter, DerivPublicAdapter

    adapter = (DerivDemoAdapter(settings, args.account_id, db=db) if args.demo
               else DerivPublicAdapter(settings, db=db))
    adapter.connect()
    try:
        adapter.subscribe_ticks(args.symbol, lambda t: print(f"{t.symbol} {t.epoch} {t.quote}"))
        end = time.time() + args.seconds
        while time.time() < end:
            time.sleep(1)
            adapter.check_health()
    finally:
        adapter.disconnect()
    return 0


def cmd_balance(args, settings, db) -> int:
    from broker.deriv_adapter import DerivDemoAdapter

    adapter = DerivDemoAdapter(settings, args.account_id, db=db)
    adapter.connect()
    try:
        acc, bal = adapter.account, adapter.get_balance()
        print(f"Cuenta Demo {acc.account_id} ({acc.account_type}, {acc.status}) "
              f"saldo: {bal.balance:.2f} {bal.currency}")
    finally:
        adapter.disconnect()
    return 0


def cmd_download(args, settings, db) -> int:
    from broker.deriv_adapter import DerivPublicAdapter

    adapter = DerivPublicAdapter(settings, db=db)
    adapter.connect()
    total = 0
    try:
        end: str | int = "latest"
        for page in range(args.pages):
            ticks = adapter.market.get_ticks_history(args.symbol, args.count, end)
            if not ticks:
                break
            total += sum(db.insert_tick(t.symbol, t.epoch, t.quote, t.pip_size, t.received_at)
                         for t in ticks)
            end = ticks[0].epoch - 1
            log.info("Página %d: %d ticks (hasta %s)", page + 1, len(ticks), ticks[0].epoch)
            time.sleep(0.5)                               # respetar límites de la API
    finally:
        adapter.disconnect()
    print(f"{total} ticks nuevos guardados para {args.symbol}")
    return 0


def cmd_train(args, settings, db) -> int:
    from database.model_registry import ModelRegistry
    from machine_learning.dataset import build_dataset, load_ticks
    from machine_learning.train import TrainConfig, train

    frame = load_ticks(db, args.symbol)
    if len(frame) < 2000:
        print(f"Solo hay {len(frame)} ticks de {args.symbol}; descarga más con 'download'.")
        return 1
    ds = build_dataset(frame, horizon=args.horizon, latency=args.latency, symbol=args.symbol)
    cfg = TrainConfig(payout_ratio=args.payout, threshold=args.threshold)
    res = train(ds, cfg, registry=ModelRegistry(db), activate_if_approved=args.activate)
    print(json.dumps(res.table(), indent=2, default=str))
    best = res.results.get(res.best_kind)
    print(f"\nMejor: {res.best_kind} · versión {res.version} · aprobado={res.approved}")
    if best and best.reasons:
        print("Motivos de no aprobación:", "; ".join(best.reasons))
    return 0


def cmd_backtest(args, settings, db) -> int:
    from core.backtester import Backtester
    from database.model_registry import ModelRegistry
    from machine_learning.predict import ModelPredictor
    from market.market_data import Tick
    from strategy.fixed_time_strategy import FixedTimeStrategy, StrategyConfig

    registry = ModelRegistry(db)
    predictor = ModelPredictor.from_registry(registry, args.model)
    frame = db.query("SELECT epoch, quote, pip_size FROM ticks WHERE symbol = ? "
                     "ORDER BY epoch", (args.symbol,))
    ticks = [Tick(args.symbol, r["epoch"], r["quote"], r["pip_size"], r["epoch"]) for r in frame]
    meta = registry.get(predictor.model_version)
    if meta and meta.get("train_to") and not args.include_train:
        ticks = [t for t in ticks if t.epoch > meta["train_to"]]   # solo fuera de muestra
    if len(ticks) < 200:
        print("No hay suficientes ticks fuera de muestra para el backtest.")
        return 1
    strat = FixedTimeStrategy(predictor, StrategyConfig(
        threshold=args.threshold, expected_payout_ratio=args.payout,
        duration=predictor.horizon, duration_unit="t"))
    bt = Backtester(strat, payout_ratio=args.payout, latency_ticks=args.latency,
                    db=Database(":memory:"))
    rep = bt.run(ticks, args.symbol)
    db.log_event("backtest", "run", {"model": predictor.model_version,
                                     "report": rep.to_dict()})
    print(json.dumps(rep.to_dict(), indent=2))
    return 0


def cmd_soak(args, settings, db) -> int:
    from core.bootstrap import build_engine

    engine = build_engine(settings, db, demo=True, account_id=args.account_id)
    engine.start()
    engine.watch_symbol(args.symbol)
    if engine.strategy is not None and args.auto:
        engine.set_auto_trading(True)
    log.info("Sesión Demo prolongada: %s h, auto-trading=%s", args.hours,
             engine.session.get("auto_trading"))
    end = time.time() + args.hours * 3600
    next_report = time.time() + args.report_every * 60
    try:
        while time.time() < end:
            time.sleep(5)
            if time.time() >= next_report:
                next_report += args.report_every * 60
                print(json.dumps(engine.soak.save()))
    except KeyboardInterrupt:
        pass
    finally:
        engine.reconciler.run_safe()
        final = engine.soak.save()
        engine.stop()
    ok, problems = engine.soak.passed(min_hours=args.hours)
    print(json.dumps({"final": final, "passed": ok, "problems": problems}, indent=2))
    return 0 if ok else 2


# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TradingBotML (Deriv, Demo)")
    sub = p.add_subparsers(dest="cmd")

    def demo_opts(sp, demo_flag=True):
        if demo_flag:
            sp.add_argument("--demo", action="store_true", help="cuenta Demo autenticada")
        sp.add_argument("--account-id", help="cuenta Demo (si hay varias)")

    demo_opts(sub.add_parser("gui", help="interfaz gráfica"))
    sub.add_parser("symbols", help="listar active_symbols")
    sp = sub.add_parser("ticks", help="mostrar ticks en consola")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--seconds", type=float, default=30)
    demo_opts(sp)
    demo_opts(sub.add_parser("balance", help="saldo Demo"), demo_flag=False)
    sp = sub.add_parser("download", help="descargar histórico de ticks a SQLite")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--count", type=int, default=5000)
    sp.add_argument("--pages", type=int, default=1)
    sp = sub.add_parser("train", help="entrenar con validación walk-forward")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--horizon", type=int, default=5, help="duración del contrato en ticks")
    sp.add_argument("--latency", type=int, default=0)
    sp.add_argument("--payout", type=float, default=1.95)
    sp.add_argument("--threshold", type=float, default=0.56)
    sp.add_argument("--activate", action="store_true", help="activar si queda aprobado")
    sp = sub.add_parser("backtest", help="backtest fuera de muestra del modelo")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--model", help="versión (por defecto, la activa)")
    sp.add_argument("--payout", type=float, default=1.95)
    sp.add_argument("--threshold", type=float, default=0.56)
    sp.add_argument("--latency", type=int, default=1)
    sp.add_argument("--include-train", action="store_true")
    sp = sub.add_parser("soak", help="sesión Demo prolongada headless (Fase 7)")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--hours", type=float, default=24)
    sp.add_argument("--report-every", type=float, default=60, help="minutos")
    sp.add_argument("--auto", action="store_true", help="auto-trading si hay modelo activo")
    demo_opts(sp, demo_flag=False)
    return p


COMMANDS = {"gui": cmd_gui, "symbols": cmd_symbols, "ticks": cmd_ticks,
            "balance": cmd_balance, "download": cmd_download, "train": cmd_train,
            "backtest": cmd_backtest, "soak": cmd_soak}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd is None:
        args = build_parser().parse_args(["gui"])
    settings = load_settings()
    setup_logging(settings.log_level)
    log.info("Entorno: %s (Real bloqueado)", settings.environment.value)
    db = Database(settings.db_path)
    try:
        return COMMANDS[args.cmd](args, settings, db)
    except KeyboardInterrupt:
        return 130
    except (SecretNotFoundError, DerivAuthError, PermissionError, ConnectionError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
