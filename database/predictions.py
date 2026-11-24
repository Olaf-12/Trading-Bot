"""Repositorio de predicciones del modelo (auditoría ML)."""

from __future__ import annotations

import time

from database.database import Database


class PredictionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, symbol: str, tick_epoch: int, model_version: str, prob_up: float,
            direction: str | None) -> int:
        cur = self.db.execute(
            "INSERT INTO predictions(ts, symbol, tick_epoch, model_version, prob_up, direction) "
            "VALUES (?,?,?,?,?,?)",
            (time.time(), symbol, int(tick_epoch), model_version, float(prob_up), direction))
        return int(cur.lastrowid)

    def mark_acted(self, prediction_id: int, client_order_id: str) -> None:
        self.db.execute("UPDATE predictions SET acted = 1, client_order_id = ? WHERE id = ?",
                        (client_order_id, prediction_id))

    def recent(self, symbol: str | None = None, limit: int = 200):
        if symbol:
            return self.db.query("SELECT * FROM predictions WHERE symbol = ? "
                                 "ORDER BY id DESC LIMIT ?", (symbol, limit))
        return self.db.query("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
