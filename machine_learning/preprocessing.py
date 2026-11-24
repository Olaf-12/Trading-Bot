"""Limpieza y pipelines de preprocesado. El escalado se ajusta SOLO con el tramo
de entrenamiento de cada fold (está dentro del Pipeline de scikit-learn)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def clean_xy(X: pd.DataFrame, y: pd.Series, tie: pd.Series, epochs: pd.Series):
    mask = X.notna().all(axis=1) & y.notna()
    return (X[mask].reset_index(drop=True), y[mask].to_numpy(dtype=int),
            tie[mask].to_numpy(dtype=bool), epochs[mask].to_numpy(dtype=np.int64))


def scaled(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])
