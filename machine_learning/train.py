"""Entrenamiento con validación walk-forward (Fase 5).

- Splits temporales con ventana creciente y PURGA entre train y test igual al
  horizonte de la etiqueta (evita que etiquetas de train vean precios de test).
- Modelos: baseline (prior constante), regresión logística y XGBoost.
- Selección por EV neto de payout en walk-forward, no por accuracy.
- Un modelo solo se marca ``approved`` si: EV > 0, trades suficientes, mayoría de
  folds positivos y logloss mejor que el baseline. Si no, se registra pero no se activa.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from config import MODELS_DIR
from database.model_registry import ModelRegistry
from machine_learning.dataset import Dataset
from machine_learning.evaluate import aggregate, evaluate_probs
from machine_learning.preprocessing import scaled

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    n_splits: int = 5
    min_train_frac: float = 0.4
    payout_ratio: float = 1.95
    threshold: float = 0.56
    min_trades: int = 50
    min_positive_fold_frac: float = 0.6
    random_state: int = 42
    models: tuple[str, ...] = ("baseline", "logreg", "xgboost")


def walk_forward_splits(n: int, n_splits: int, min_train_frac: float,
                        gap: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Ventana creciente: train = [0, test_start - gap), test = bloque siguiente."""
    first_test = int(n * min_train_frac)
    test_size = (n - first_test) // n_splits
    if test_size <= 0 or first_test - gap <= 0:
        raise ValueError("Dataset demasiado pequeño para walk-forward")
    for k in range(n_splits):
        start = first_test + k * test_size
        end = n if k == n_splits - 1 else start + test_size
        yield np.arange(0, start - gap), np.arange(start, end)


def _xgb(random_state: int):
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                         objective="binary:logistic", eval_metric="logloss",
                         n_jobs=2, random_state=random_state, verbosity=0)


MODEL_FACTORIES: dict[str, Callable[[int], Any]] = {
    "baseline": lambda rs: DummyClassifier(strategy="prior"),
    "logreg": lambda rs: scaled(LogisticRegression(C=0.5, max_iter=1000)),
    "xgboost": _xgb,
}


@dataclass
class ModelResult:
    kind: str
    folds: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    approved: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass
class TrainingResult:
    results: dict[str, ModelResult]
    best_kind: str | None
    version: str | None = None
    path: Path | None = None
    approved: bool = False

    def table(self) -> list[dict[str, Any]]:
        return [{"model": k, "approved": r.approved, **{x: r.summary.get(x) for x in (
            "trades", "win_rate", "ev_per_trade", "pnl_units", "positive_folds", "auc",
            "logloss")}} for k, r in self.results.items()]


def _predict_up(model, X) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        return np.zeros(len(X))
    return proba[:, classes.index(1)]


def cross_validate(ds: Dataset, cfg: TrainConfig) -> dict[str, ModelResult]:
    gap = ds.horizon + ds.latency + 1
    results = {k: ModelResult(k) for k in cfg.models}
    for train_idx, test_idx in walk_forward_splits(len(ds), cfg.n_splits, cfg.min_train_frac,
                                                   gap):
        assert train_idx.max() + gap <= test_idx.min()          # sin solapamiento
        Xtr, ytr = ds.X.iloc[train_idx], ds.y[train_idx]
        Xte, yte, tte = ds.X.iloc[test_idx], ds.y[test_idx], ds.ties[test_idx]
        for kind in cfg.models:
            model = MODEL_FACTORIES[kind](cfg.random_state)
            model.fit(Xtr, ytr)
            m = evaluate_probs(_predict_up(model, Xte), yte, tte,
                               payout_ratio=cfg.payout_ratio, threshold=cfg.threshold)
            m["train_from"], m["test_from"] = int(ds.epochs[train_idx[0]]), int(ds.epochs[test_idx[0]])
            results[kind].folds.append(m)
    for r in results.values():
        r.summary = aggregate(r.folds)
    _approve(results, cfg)
    return results


def _approve(results: dict[str, ModelResult], cfg: TrainConfig) -> None:
    base = results.get("baseline")
    base_ll = base.summary.get("logloss", math.inf) if base else math.inf
    for kind, r in results.items():
        s = r.summary
        if kind == "baseline":
            r.reasons.append("el baseline es solo referencia")
            continue
        if s["trades"] < cfg.min_trades:
            r.reasons.append(f"pocos trades ({s['trades']} < {cfg.min_trades})")
        if s["ev_per_trade"] <= 0:
            r.reasons.append(f"EV neto por trade {s['ev_per_trade']:.4f} <= 0")
        if s["positive_folds"] < math.ceil(s["folds"] * cfg.min_positive_fold_frac):
            r.reasons.append(f"solo {s['positive_folds']}/{s['folds']} folds positivos")
        if not s["logloss"] < base_ll:
            r.reasons.append("logloss no mejora al baseline")
        r.approved = not r.reasons


def train(ds: Dataset, cfg: TrainConfig | None = None, *,
          registry: ModelRegistry | None = None, models_dir: Path = MODELS_DIR,
          activate_if_approved: bool = False) -> TrainingResult:
    cfg = cfg or TrainConfig()
    results = cross_validate(ds, cfg)
    candidates = [r for k, r in results.items() if k != "baseline"]
    best = max(candidates, key=lambda r: (r.approved, r.summary.get("ev_per_trade", -1)),
               default=None)
    out = TrainingResult(results, best.kind if best else None)
    if best is None:
        return out

    # Modelo final: se reentrena con todo el histórico disponible.
    model = MODEL_FACTORIES[best.kind](cfg.random_state)
    model.fit(ds.X, ds.y)
    stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    digest = hashlib.sha256(json.dumps(best.summary, sort_keys=True, default=str)
                            .encode()).hexdigest()[:8]
    version = f"{best.kind}-{ds.symbol or 'any'}-h{ds.horizon}-{stamp}-{digest}"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{version}.joblib"
    bundle = {"model": model, "features": ds.features, "version": version,
              "kind": best.kind, "horizon": ds.horizon, "latency": ds.latency,
              "symbol": ds.symbol, "threshold": cfg.threshold,
              "payout_ratio": cfg.payout_ratio, "approved": best.approved,
              "trained_at": time.time()}
    joblib.dump(bundle, path)
    out.version, out.path, out.approved = version, path, best.approved
    if registry is not None:
        registry.register(version, best.kind, path, ds.features,
                          params={"threshold": cfg.threshold, "horizon": ds.horizon,
                                  "latency": ds.latency, "payout_ratio": cfg.payout_ratio},
                          metrics={"walk_forward": best.summary, "approved": best.approved,
                                   "reasons": best.reasons,
                                   "baseline": results["baseline"].summary
                                   if "baseline" in results else None},
                          symbol=ds.symbol, train_from=int(ds.epochs[0]),
                          train_to=int(ds.epochs[-1]))
        if activate_if_approved and best.approved:
            registry.activate(version)
    log.info("Modelo %s (%s) aprobado=%s motivos=%s", version, best.kind, best.approved,
             best.reasons)
    return out
