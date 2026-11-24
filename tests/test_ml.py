"""Fase 5: dataset, prevención de leakage, walk-forward, evaluación neta y registro."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtester import Backtester
from database.database import Database
from database.model_registry import ModelRegistry
from machine_learning.dataset import build_dataset, make_labels, synthetic_ticks, ticks_to_frame
from machine_learning.evaluate import evaluate_probs
from machine_learning.features import FEATURE_COLUMNS, build_features
from machine_learning.predict import ModelIntegrityError, ModelPredictor
from machine_learning.train import TrainConfig, train, walk_forward_splits
from risk.risk_manager import RiskLimits
from strategy.fixed_time_strategy import FixedTimeStrategy, StrategyConfig

FAST_CFG = TrainConfig(n_splits=4, min_trades=30, models=("baseline", "logreg", "xgboost"))


def test_features_do_not_use_future():
    frame = ticks_to_frame(synthetic_ticks(400, seed=1))
    full = build_features(frame)
    altered = frame.copy()
    altered.loc[300:, "quote"] *= 1.5             # cambiar el futuro
    part = build_features(altered)
    pd.testing.assert_frame_equal(full.iloc[:300], part.iloc[:300])
    assert list(full.columns) == FEATURE_COLUMNS


def test_labels_use_next_tick_as_entry_and_ties():
    q = pd.Series([10.0, 11, 12, 12, 11])
    y, tie = make_labels(q, horizon=1)
    # fila 0: entrada q1=11, salida q2=12 -> sube ; fila 1: 12 vs 12 -> empate
    assert y.tolist()[:3] == [1, 0, 0] and tie.tolist()[:3] == [0, 1, 0]
    assert np.isnan(y.iloc[-1]) and np.isnan(y.iloc[-2])


def test_dataset_drops_incomplete_rows():
    ds = build_dataset(ticks_to_frame(synthetic_ticks(500, seed=2)), horizon=5)
    assert not ds.X.isna().any().any() and len(ds) == len(ds.y) == len(ds.ties)
    assert (np.diff(ds.epochs) > 0).all()


def test_walk_forward_splits_are_purged_and_ordered():
    splits = list(walk_forward_splits(1000, 4, 0.4, gap=7))
    assert len(splits) == 4
    prev_end = -1
    for tr, te in splits:
        assert tr.max() + 7 <= te.min()
        assert te.min() > prev_end
        prev_end = te.max()
    assert splits[-1][1].max() == 999


def test_evaluate_counts_ties_as_losses_and_net_payout():
    prob = np.array([0.9, 0.9, 0.1, 0.1, 0.5])
    y = np.array([1, 0, 0, 0, 1])
    ties = np.array([False, True, False, True, False])
    m = evaluate_probs(prob, y, ties, payout_ratio=1.9, threshold=0.6)
    # Trades: up gana, up empata(pierde), down gana, down empata(pierde) ; 0.5 no opera
    assert m["trades"] == 4 and m["win_rate"] == 0.5
    assert m["pnl_units"] == pytest.approx(2 * 0.9 - 2)
    assert m["breakeven_win_rate"] == pytest.approx(1 / 1.9)


def test_random_walk_model_is_not_approved(tmp_path):
    ds = build_dataset(ticks_to_frame(synthetic_ticks(6000, seed=11)), horizon=5,
                       symbol="SYN")
    db = Database(":memory:")
    reg = ModelRegistry(db)
    res = train(ds, FAST_CFG, registry=reg, models_dir=tmp_path, activate_if_approved=True)
    assert not res.approved
    assert reg.active() is None                  # nunca se activa sin aprobación
    assert reg.get(res.version)["metrics"]["reasons"]


def test_learnable_signal_is_found_and_approved(tmp_path):
    # Reversión a la media fuerte: el siguiente movimiento es predecible.
    ticks = synthetic_ticks(8000, seed=5, mean_reversion=0.6)
    ds = build_dataset(ticks_to_frame(ticks), horizon=1, symbol="SYN")
    db = Database(":memory:")
    reg = ModelRegistry(db)
    res = train(ds, TrainConfig(n_splits=4, min_trades=30, threshold=0.56),
                registry=reg, models_dir=tmp_path, activate_if_approved=True)
    best = res.results[res.best_kind].summary
    assert res.approved, res.results[res.best_kind].reasons
    assert best["ev_per_trade"] > 0 and best["win_rate"] > 1 / 1.95
    assert best["logloss"] < res.results["baseline"].summary["logloss"]
    assert reg.active()["version"] == res.version
    assert reg.verify(res.version)

    # Inferencia con integridad verificada + backtest fuera de muestra.
    pred = ModelPredictor.from_registry(reg)
    oos = synthetic_ticks(2500, seed=99, mean_reversion=0.6)
    p = pred.predict(oos[:200])
    assert p is not None and 0 <= p.prob_up <= 1 and p.tick_epoch == oos[199].epoch
    strat = FixedTimeStrategy(pred, StrategyConfig(threshold=0.56, duration=1, cooldown_ticks=1))
    bt = Backtester(strat, limits=RiskLimits(min_seconds_between_trades=0, max_daily_loss=1e9,
                                             max_trades_per_hour=10_000))
    r = bt.run(oos)
    assert r.trades > 50 and r.pnl > 0


def test_tampered_model_refused(tmp_path):
    ds = build_dataset(ticks_to_frame(synthetic_ticks(3000, seed=3)), horizon=3, symbol="SYN")
    reg = ModelRegistry(Database(":memory:"))
    res = train(ds, TrainConfig(n_splits=3, models=("baseline", "logreg")), registry=reg,
                models_dir=tmp_path)
    with open(res.path, "ab") as fh:
        fh.write(b"tampered")
    assert not reg.verify(res.version)
    with pytest.raises(ModelIntegrityError):
        ModelPredictor.from_registry(reg, res.version)


def test_features_identical_with_short_window():
    """Ventana de inferencia (corta) == serie completa de entrenamiento."""
    from machine_learning.features import MIN_HISTORY
    frame = ticks_to_frame(synthetic_ticks(1500, seed=8))
    full = build_features(frame)
    for end in (MIN_HISTORY + 1, 700, 1499):
        short = build_features(frame.iloc[end - MIN_HISTORY:end + 1].reset_index(drop=True))
        np.testing.assert_allclose(short.iloc[-1].to_numpy(), full.iloc[end].to_numpy(),
                                   rtol=1e-9)


def test_prediction_matches_training_features(tmp_path):
    """La inferencia online produce la misma probabilidad que el pipeline batch."""
    ticks = synthetic_ticks(3000, seed=4, mean_reversion=0.3)
    ds = build_dataset(ticks_to_frame(ticks), horizon=1, symbol="SYN")
    reg = ModelRegistry(Database(":memory:"))
    res = train(ds, TrainConfig(n_splits=3, models=("baseline", "logreg")), registry=reg,
                models_dir=tmp_path)
    pred = ModelPredictor.from_registry(reg, res.version)
    i = 2500
    online = pred.predict(ticks[: i + 1]).prob_up
    feats = build_features(ticks_to_frame(ticks[: i + 1])).iloc[[-1]]
    batch = pred.model.predict_proba(feats)[0, 1]
    assert online == pytest.approx(batch, rel=1e-6)
