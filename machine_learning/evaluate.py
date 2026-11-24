"""Evaluación NETA de pagos (PDF §10): no basta el % de aciertos.

Por unidad de stake: ganar = payout_ratio - 1; perder o empatar = -1.
Breakeven = 1 / payout_ratio.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def trading_outcomes(prob_up: np.ndarray, y: np.ndarray, ties: np.ndarray,
                     threshold: float, allow_down: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (operado?, ganado?) por fila según el umbral de la estrategia."""
    up = prob_up >= threshold
    down = (1 - prob_up >= threshold) & allow_down & ~up
    traded = up | down
    won = (up & (y == 1)) | (down & (y == 0) & ~ties)
    return traded, won


def evaluate_probs(prob_up: np.ndarray, y: np.ndarray, ties: np.ndarray, *,
                   payout_ratio: float, threshold: float,
                   allow_down: bool = True) -> dict[str, Any]:
    prob_up = np.clip(np.asarray(prob_up, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    ties = np.asarray(ties, dtype=bool)
    traded, won = trading_outcomes(prob_up, y, ties, threshold, allow_down)
    n = int(traded.sum())
    wins = int((won & traded).sum())
    pnl = wins * (payout_ratio - 1) - (n - wins)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(((prob_up >= 0.5) == (y == 1)).mean()) if len(y) else math.nan,
        "auc": float(roc_auc_score(y, prob_up)) if len(set(y.tolist())) == 2 else math.nan,
        "logloss": float(log_loss(y, prob_up, labels=[0, 1])) if len(y) else math.nan,
        "brier": float(brier_score_loss(y, prob_up)) if len(y) else math.nan,
        "tie_rate": float(ties.mean()) if len(ties) else 0.0,
        "threshold": threshold,
        "payout_ratio": payout_ratio,
        "breakeven_win_rate": 1 / payout_ratio,
        "trades": n,
        "win_rate": wins / n if n else 0.0,
        "pnl_units": float(pnl),
        "ev_per_trade": float(pnl / n) if n else 0.0,
    }
    return out


def aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega folds: PnL y trades se suman; métricas de clasificación se promedian."""
    if not folds:
        return {}
    trades = sum(f["trades"] for f in folds)
    pnl = sum(f["pnl_units"] for f in folds)
    wins = sum(round(f["win_rate"] * f["trades"]) for f in folds)

    def mean(k: str) -> float:
        vals = [f[k] for f in folds if not math.isnan(f[k])]
        return float(np.mean(vals)) if vals else math.nan

    return {"folds": len(folds), "trades": trades, "pnl_units": pnl,
            "ev_per_trade": pnl / trades if trades else 0.0,
            "win_rate": wins / trades if trades else 0.0,
            "positive_folds": sum(f["pnl_units"] > 0 for f in folds),
            "auc": mean("auc"), "logloss": mean("logloss"), "brier": mean("brier"),
            "accuracy": mean("accuracy"),
            "breakeven_win_rate": folds[0]["breakeven_win_rate"]}
