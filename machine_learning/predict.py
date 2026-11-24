"""Inferencia en tiempo real con el modelo registrado.

Seguridad: ``joblib``/pickle puede ejecutar código al cargar. Solo se cargan
artefactos cuyo SHA-256 coincide con el del registro de modelos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from database.model_registry import ModelRegistry, file_sha256
from machine_learning.dataset import ticks_to_frame
from machine_learning.features import FEATURE_COLUMNS, MIN_HISTORY, build_features
from market.market_data import Tick
from strategy.strategy import Prediction


class ModelIntegrityError(RuntimeError):
    pass


class ModelPredictor:
    def __init__(self, bundle: dict[str, Any]) -> None:
        if list(bundle["features"]) != FEATURE_COLUMNS:
            raise ModelIntegrityError("Las features del modelo no coinciden con el código "
                                      "actual: reentrena el modelo")
        self.bundle = bundle
        self.model = bundle["model"]
        self.model_version: str = bundle["version"]
        self.horizon: int = bundle["horizon"]
        self.threshold: float = bundle.get("threshold", 0.56)
        self.approved: bool = bool(bundle.get("approved"))
        self.min_history = MIN_HISTORY + 5
        classes = list(getattr(self.model, "classes_", [0, 1]))
        self._up_index = classes.index(1) if 1 in classes else None

    @classmethod
    def from_file(cls, path: str | Path, expected_sha256: str) -> "ModelPredictor":
        if file_sha256(path) != expected_sha256:
            raise ModelIntegrityError(f"Hash del modelo no coincide: {path}")
        return cls(joblib.load(path))

    @classmethod
    def from_registry(cls, registry: ModelRegistry, version: str | None = None
                      ) -> "ModelPredictor":
        meta = registry.get(version) if version else registry.active()
        if meta is None:
            raise ModelIntegrityError("No hay modelo activo/registrado")
        return cls.from_file(meta["path"], meta["sha256"])

    def predict(self, ticks: Sequence[Tick]) -> Prediction | None:
        if len(ticks) < self.min_history:
            return None
        window = list(ticks)[-(self.min_history * 2):]
        feats = build_features(ticks_to_frame(window))
        row = feats.iloc[[-1]]
        if row.isna().any(axis=1).iloc[0] or not np.isfinite(row.to_numpy()).all():
            return None
        if self._up_index is None:
            return Prediction(0.0, self.model_version, window[-1].epoch)
        prob = float(self.model.predict_proba(row)[0, self._up_index])
        return Prediction(prob, self.model_version, window[-1].epoch)
