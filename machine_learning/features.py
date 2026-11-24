"""Features por tick, todas CAUSALES (fila i usa solo ticks <= i) y de VENTANA
FINITA (sin EMA/RSI recursivos): con >= MIN_HISTORY ticks el valor es idéntico
en entrenamiento e inferencia.

La misma función se usa en entrenamiento y en inferencia, de modo que el modelo
ve exactamente las mismas transformaciones en Demo que en el backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market import indicators as ind

FEATURE_COLUMNS: list[str] = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_10", "vol_30", "vol_ratio",
    "rsi_14", "boll_z_20", "sma_diff_5_20", "zret_30",
    "up_ratio_10", "tick_dt", "dist_max_20", "dist_min_20",
]

# Ticks de historial necesarios para que todas las features estén definidas.
MIN_HISTORY = 45


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` con columnas ``epoch`` y ``quote`` ordenadas por epoch."""
    p = frame["quote"].astype(float)
    r1 = ind.log_returns(p, 1)
    f = pd.DataFrame(index=frame.index)
    for lag in (1, 3, 5, 10, 20):
        f[f"ret_{lag}"] = ind.log_returns(p, lag)
    f["vol_10"] = ind.rolling_volatility(p, 10)
    f["vol_30"] = ind.rolling_volatility(p, 30)
    f["vol_ratio"] = f["vol_10"] / f["vol_30"].replace(0, np.nan)
    f["rsi_14"] = ind.rsi_finite(p, 14) / 100.0
    f["boll_z_20"] = ind.bollinger_z(p, 20)
    f["sma_diff_5_20"] = (ind.sma(p, 5) - ind.sma(p, 20)) / p
    f["zret_30"] = ind.zscore(r1, 30)
    f["up_ratio_10"] = (p.diff() > 0).astype(float).rolling(10, min_periods=10).mean()
    f["tick_dt"] = frame["epoch"].astype(float).diff()
    f["dist_max_20"] = np.log(p / p.rolling(20, min_periods=20).max())
    f["dist_min_20"] = np.log(p / p.rolling(20, min_periods=20).min())
    return f.replace([np.inf, -np.inf], np.nan)[FEATURE_COLUMNS]
