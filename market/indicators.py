"""Indicadores técnicos CAUSALES: el valor en la posición i solo usa datos <= i.

``ema`` y ``rsi`` son recursivos (dependen de todo el historial): úsalos para
visualización. Las features de ML usan solo indicadores de ventana finita para
evitar desfase entre entrenamiento e inferencia.

Implementados con pandas para que el mismo código sirva en entrenamiento
(series largas) e inferencia (ventana reciente) y produzca valores idénticos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(price: pd.Series, lag: int = 1) -> pd.Series:
    return np.log(price / price.shift(lag))


def sma(price: pd.Series, window: int) -> pd.Series:
    return price.rolling(window, min_periods=window).mean()


def ema(price: pd.Series, span: int) -> pd.Series:
    # adjust=False => recursivo y causal; min_periods evita valores con poco historial.
    return price.ewm(span=span, adjust=False, min_periods=span).mean()


def rolling_volatility(price: pd.Series, window: int) -> pd.Series:
    return log_returns(price).rolling(window, min_periods=window).std()


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def rsi_finite(price: pd.Series, window: int = 14) -> pd.Series:
    """RSI de Cutler (medias simples): depende solo de los últimos ``window`` ticks,
    por lo que da el mismo valor en entrenamiento (serie larga) e inferencia (ventana)."""
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    total = gain + loss
    return (100 * gain / total.replace(0, np.nan)).where(total != 0, 50.0).where(gain.notna())


def bollinger_z(price: pd.Series, window: int = 20) -> pd.Series:
    mean = price.rolling(window, min_periods=window).mean()
    std = price.rolling(window, min_periods=window).std()
    return (price - mean) / std.replace(0, np.nan)


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0, np.nan)
