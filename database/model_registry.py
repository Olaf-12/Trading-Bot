"""Registro de modelos ML: versión, hash del artefacto, features, métricas y rango
temporal de entrenamiento. Cada trade guarda la ``model_version`` que lo originó."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from database.database import Database


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    def __init__(self, db: Database) -> None:
        self.db = db

    def register(self, version: str, kind: str, path: str | Path, features: list[str],
                 params: dict[str, Any] | None, metrics: dict[str, Any] | None,
                 symbol: str | None, train_from: int | None, train_to: int | None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO models(version, created_at, kind, symbol, path, sha256, "
            "features, params, metrics, train_from, train_to, active) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?, COALESCE((SELECT active FROM models WHERE "
            "version = ?), 0))",
            (version, time.time(), kind, symbol, str(path), file_sha256(path),
             json.dumps(features), json.dumps(params or {}, default=str),
             json.dumps(metrics or {}, default=str), train_from, train_to, version))

    def activate(self, version: str) -> None:
        if not self.get(version):
            raise KeyError(f"Modelo {version} no registrado")
        self.db.execute("UPDATE models SET active = (version = ?)", (version,))

    def get(self, version: str) -> dict[str, Any] | None:
        rows = self.db.query("SELECT * FROM models WHERE version = ?", (version,))
        return self._row(rows[0]) if rows else None

    def active(self) -> dict[str, Any] | None:
        rows = self.db.query("SELECT * FROM models WHERE active = 1")
        return self._row(rows[0]) if rows else None

    def list(self) -> list[dict[str, Any]]:
        return [self._row(r) for r in self.db.query(
            "SELECT * FROM models ORDER BY created_at DESC")]

    def verify(self, version: str) -> bool:
        """El artefacto en disco coincide con el hash registrado (integridad)."""
        m = self.get(version)
        return bool(m) and Path(m["path"]).exists() and file_sha256(m["path"]) == m["sha256"]

    @staticmethod
    def _row(r) -> dict[str, Any]:
        d = dict(r)
        for k in ("features", "params", "metrics"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        d["active"] = bool(d["active"])
        return d
