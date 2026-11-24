"""Construcción del dataset supervisado para contratos Rise/Fall en ticks.

Para la fila i (decisión tomada con el tick i):
  entrada = tick i+1+latencia   (Deriv usa el tick siguiente a la compra)
  salida  = tick i+1+latencia+horizonte
  y = 1 si salida > entrada; tie = salida == entrada (pierde tanto CALL como PUT)
Las etiquetas usan el futuro (es su definición); las features nunca.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from database.database import Database
from machine_learning.features import FEATURE_COLUMNS, build_features
from machine_learning.preprocessing import clean_xy
from market.market_data import Tick


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    ties: np.ndarray
    epochs: np.ndarray
    horizon: int
    latency: int
    symbol: str | None = None

    def __len__(self) -> int:
        return len(self.y)

    @property
    def features(self) -> list[str]:
        return list(self.X.columns)


def ticks_to_frame(ticks: Iterable[Tick]) -> pd.DataFrame:
    rows = [(t.epoch, t.quote) for t in ticks]
    return pd.DataFrame(rows, columns=["epoch", "quote"])


def load_ticks(db: Database, symbol: str, limit: int | None = None) -> pd.DataFrame:
    sql = "SELECT epoch, quote FROM ticks WHERE symbol = ? ORDER BY epoch"
    rows = db.query(sql, (symbol,))
    frame = pd.DataFrame([tuple(r) for r in rows], columns=["epoch", "quote"])
    return frame.tail(limit).reset_index(drop=True) if limit else frame


def make_labels(quote: pd.Series, horizon: int, latency: int = 0) -> tuple[pd.Series, pd.Series]:
    entry = quote.shift(-(1 + latency))
    exit_ = quote.shift(-(1 + latency + horizon))
    y = (exit_ > entry).astype(float).where(exit_.notna())
    tie = (exit_ == entry).astype(float).where(exit_.notna())
    return y, tie


def build_dataset(frame: pd.DataFrame, horizon: int, latency: int = 0,
                  symbol: str | None = None) -> Dataset:
    frame = frame.sort_values("epoch").drop_duplicates("epoch").reset_index(drop=True)
    X = build_features(frame)
    y, tie = make_labels(frame["quote"], horizon, latency)
    X, y, tie, epochs = clean_xy(X, y, tie, frame["epoch"])
    return Dataset(X[FEATURE_COLUMNS], y, tie, epochs, horizon, latency, symbol)


# ----------------------------------------------------------------------------
# Datos sintéticos (pruebas y demostración; NUNCA para decidir sobre dinero real)
# ----------------------------------------------------------------------------
def synthetic_ticks(n: int, *, seed: int = 0, mean_reversion: float = 0.0,
                    symbol: str = "SYN", start: int = 1_700_000_000,
                    step: int = 1, price: float = 1000.0) -> list[Tick]:
    """Paseo aleatorio con reversión opcional a la media de los incrementos
    (mean_reversion > 0 introduce una señal aprendible)."""
    rng = np.random.default_rng(seed)
    out: list[Tick] = []
    prev = 0.0
    for i in range(n):
        shock = rng.normal(0, 1)
        inc = -mean_reversion * prev + shock
        price = max(1.0, price + inc * 0.1)
        prev = inc
        out.append(Tick(symbol, start + i * step, round(price, 2), 2, start + i * step))
    return out


def frame_from_sequence(ticks: Sequence[Tick]) -> pd.DataFrame:
    return ticks_to_frame(ticks)
