"""Fase 4: velas, indicadores causales, estrategia, simulador y backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from broker.simulated_broker import SimulatedBroker
from core.backtester import Backtester, max_drawdown
from core.execution_controller import ExecutionController
from config import Settings
from market import indicators as ind
from market.candles import CandleBuilder, build_candles
from market.market_data import Tick
from risk.position_sizing import SizingConfig
from risk.risk_manager import RiskLimits, RiskManager
from strategy.fixed_time_strategy import FixedTimeStrategy, StrategyConfig
from strategy.signals import ContractSpec, Direction, OrderRequest
from strategy.strategy import Prediction

T0 = 1_700_000_000


def ticks_from(prices, symbol="R_100", step=1):
    return [Tick(symbol, T0 + i * step, float(p), 2, T0 + i * step) for i, p in enumerate(prices)]


class FixedPredictor:
    model_version = "fixed"
    min_history = 3

    def __init__(self, prob_up):
        self.prob_up = prob_up

    def predict(self, ticks):
        return Prediction(self.prob_up, self.model_version, ticks[-1].epoch)


class OraclePredictor:
    """Hace trampa a propósito: mira el futuro. Sirve para validar la contabilidad."""
    model_version = "oracle"
    min_history = 1

    def __init__(self, all_ticks, horizon):
        self.future = {t.epoch: i for i, t in enumerate(all_ticks)}
        self.all = all_ticks
        self.h = horizon

    def predict(self, ticks):
        i = self.future[ticks[-1].epoch]
        if i + 1 + self.h >= len(self.all):
            return None
        entry, exit_ = self.all[i + 1].quote, self.all[i + 1 + self.h].quote
        return Prediction(0.99 if exit_ > entry else 0.01, "oracle", ticks[-1].epoch)


# --------------------------------------------------------------------------
# Velas e indicadores
# --------------------------------------------------------------------------
def test_candles_only_emit_closed():
    b = CandleBuilder(60)
    ts = [Tick("X", T0 - T0 % 60 + s, p, 2, 0) for s, p in [(0, 1), (30, 3), (59, 2), (60, 5)]]
    out = [b.add(t) for t in ts]
    assert out[:3] == [None, None, None]
    c = out[3]
    assert (c.open, c.high, c.low, c.close, c.ticks) == (1, 3, 1, 2, 3)
    assert len(build_candles(ts, 60)) == 1


@pytest.mark.parametrize("fn", [
    lambda s: ind.sma(s, 5), lambda s: ind.ema(s, 5), lambda s: ind.rsi(s, 5),
    lambda s: ind.rsi_finite(s, 5),
    lambda s: ind.bollinger_z(s, 5), lambda s: ind.rolling_volatility(s, 5),
    lambda s: ind.log_returns(s, 3), lambda s: ind.zscore(s, 5),
])
def test_indicators_are_causal(fn):
    rng = np.random.default_rng(0)
    s = pd.Series(100 + rng.normal(0, 1, 200).cumsum())
    full = fn(s)
    for cut in (50, 120, 199):
        partial = fn(s.iloc[: cut + 1])
        a, b = full.iloc[cut], partial.iloc[-1]
        assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b)


# --------------------------------------------------------------------------
# Estrategia
# --------------------------------------------------------------------------
def test_threshold_must_beat_breakeven():
    with pytest.raises(ValueError):
        StrategyConfig(threshold=0.51, expected_payout_ratio=1.95)
    StrategyConfig(threshold=0.53, expected_payout_ratio=1.95)


def test_strategy_signals_and_cooldown():
    ticks = ticks_from([100, 101, 102, 103])
    s = FixedTimeStrategy(FixedPredictor(0.7), StrategyConfig(cooldown_ticks=3))
    d = s.evaluate("R_100", ticks, 1000, ticks[-1].epoch)
    assert d.order and d.order.spec.contract_type == "CALL"
    assert d.order.signal.direction is Direction.UP and d.order.signal.probability == 0.7
    assert s.evaluate("R_100", ticks, 1000, ticks[-1].epoch).reason == "cooldown"


def test_strategy_down_and_neutral():
    ticks = ticks_from([100, 101, 102])
    d = FixedTimeStrategy(FixedPredictor(0.2)).evaluate("R_100", ticks, 1000, ticks[-1].epoch)
    assert d.order.spec.contract_type == "PUT" and d.order.signal.probability == pytest.approx(0.8)
    d = FixedTimeStrategy(FixedPredictor(0.52)).evaluate("R_100", ticks, 1000, ticks[-1].epoch)
    assert d.order is None and "umbral" in d.reason


def test_strategy_guards():
    ticks = ticks_from([100, 101, 102])
    s = FixedTimeStrategy(FixedPredictor(0.9))
    assert s.evaluate("R_100", ticks, 1000, ticks[-1].epoch + 60).reason == "último tick obsoleto"
    assert s.evaluate("R_100", ticks[:2], 1000, ticks[-1].epoch).reason == "historial insuficiente"
    assert s.evaluate("R_100", ticks, 1000, ticks[-1].epoch, market_ok=False).order is None


def test_strategy_rejects_lookahead_predictor():
    class Cheater(FixedPredictor):
        def predict(self, ticks):
            return Prediction(0.9, "cheat", ticks[-1].epoch + 1)
    ticks = ticks_from([1, 2, 3])
    with pytest.raises(RuntimeError):
        FixedTimeStrategy(Cheater(0.9)).evaluate("R_100", ticks, 1000, ticks[-1].epoch)


# --------------------------------------------------------------------------
# Simulador: reglas de liquidación
# --------------------------------------------------------------------------
def run_single(prices, contract_type, duration, unit, step=1, latency=0):
    sim = SimulatedBroker(payout_ratio=1.9, latency_ticks=latency)
    ctl = ExecutionController(Settings(), sim, RiskManager(RiskLimits(min_balance=0),
                                                           clock=sim.clock), clock=sim.clock)
    ticks = ticks_from(prices, step=step)
    sim.on_tick(ticks[0])
    res = ctl.submit(OrderRequest(ContractSpec("R_100", contract_type, duration, unit, 1, "USD")))
    assert res.status == "open", res.reasons
    for t in ticks[1:]:
        sim.on_tick(t)
    return sim.settled[0] if sim.settled else None, sim


@pytest.mark.parametrize("prices, ctype, won", [
    ([100, 101, 102, 103], "CALL", True),     # entrada 101 (tick siguiente), salida 103
    ([100, 101, 100, 99], "CALL", False),
    ([100, 101, 100, 99], "PUT", True),
    ([100, 101, 105, 101], "CALL", False),    # empate entrada=salida -> pérdida
    ([100, 101, 105, 101], "PUT", False),
])
def test_tick_contract_settlement(prices, ctype, won):
    u, sim = run_single(prices, ctype, 2, "t")
    assert u.entry_spot == 101 and u.status == ("won" if won else "lost")
    assert u.profit == pytest.approx(0.9 if won else -1.0)
    assert sim.balance == pytest.approx(10_000 + (0.9 if won else -1.0))


def test_entry_is_next_tick_not_purchase_tick():
    u, _ = run_single([100, 200, 300, 400], "CALL", 1, "t")
    assert u.entry_spot == 200 and u.exit_spot == 300


def test_latency_delays_entry():
    u, _ = run_single([100, 200, 300, 400, 500], "CALL", 1, "t", latency=1)
    assert u.entry_spot == 300 and u.exit_spot == 400


def test_time_contract_uses_last_tick_at_or_before_expiry():
    # ticks cada 2s; compra en T0, vence T0+5 -> último tick <= T0+5 es T0+4
    u, _ = run_single([100, 101, 103, 99, 50], "CALL", 5, "s", step=2)
    assert u.exit_spot_time == T0 + 4 and u.exit_spot == 103 and u.status == "won"


# --------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------
def random_walk(n, seed=1):
    rng = np.random.default_rng(seed)
    return ticks_from(1000 + rng.choice([-1, 1], n).cumsum() * 0.5)


def test_backtest_oracle_wins_every_trade():
    ticks = random_walk(400)
    strat = FixedTimeStrategy(OraclePredictor(ticks, 5),
                              StrategyConfig(threshold=0.6, duration=5, cooldown_ticks=1))
    bt = Backtester(strat, limits=RiskLimits(min_seconds_between_trades=0, max_daily_loss=1e9,
                                             max_trades_per_hour=10_000, max_open_contracts=1),
                    payout_ratio=1.95)
    r = bt.run(ticks)
    assert r.trades > 20
    # Solo el oráculo de empates pierde: entradas con exit == entry dan prob 0.01 (PUT) y pierden.
    assert r.wins + r.losses == r.trades
    assert r.losses == r.ties
    assert r.pnl == pytest.approx(r.wins * 0.95 - r.losses * 1.0, abs=0.05)
    assert bt.repo.summary()["settled"] == r.trades


def test_backtest_constant_bias_loses_on_random_walk():
    ticks = random_walk(3000, seed=7)
    strat = FixedTimeStrategy(FixedPredictor(0.9), StrategyConfig(threshold=0.6, cooldown_ticks=1))
    bt = Backtester(strat, limits=RiskLimits(min_seconds_between_trades=0, max_daily_loss=1e9,
                                             max_trades_per_hour=10_000))
    r = bt.run(ticks)
    assert r.trades > 100
    assert r.ev_per_trade < 0          # sin ventaja, el payout < 2 hace perder
    assert r.breakeven_win_rate == pytest.approx(1 / 1.95)
    assert r.max_drawdown == pytest.approx(max_drawdown(r.equity_curve))
    assert bt.save("bias") == r.trades


def test_backtest_respects_daily_loss_limit():
    ticks = random_walk(3000, seed=3)
    strat = FixedTimeStrategy(FixedPredictor(0.9), StrategyConfig(
        threshold=0.6, cooldown_ticks=1, sizing=SizingConfig(fixed_stake=1)))
    bt = Backtester(strat, limits=RiskLimits(min_seconds_between_trades=0, max_daily_loss=10,
                                             max_trades_per_hour=10_000,
                                             max_consecutive_losses=10**6))
    r = bt.run(ticks)
    assert r.pnl >= -10 - 1e-9
    assert any("exposición" in k or "pérdida diaria" in k for k in r.reject_reasons)
