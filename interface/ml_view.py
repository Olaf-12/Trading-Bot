"""ML: modelos registrados (métricas walk-forward) y predicciones recientes."""

from __future__ import annotations

from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from core import event_manager as ev


def fill_table(table: QTableWidget, headers: list[str], rows: list[list]) -> None:
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            if isinstance(val, float):
                val = f"{val:.4f}"
            table.setItem(r, c, QTableWidgetItem("" if val is None else str(val)))
    table.resizeColumnsToContents()


class MLView(QWidget):
    def __init__(self, engine) -> None:
        super().__init__()
        self.engine = engine
        root = QVBoxLayout(self)
        self.lbl_pred = QLabel("Predicción: —")
        self.lbl_pred.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(self.lbl_pred)

        root.addWidget(QLabel("Modelos (métricas walk-forward, netas de payout)"))
        self.models = QTableWidget()
        self.models.setEditTriggers(QTableWidget.NoEditTriggers)
        self.models.setSelectionBehavior(QTableWidget.SelectRows)
        root.addWidget(self.models, 1)
        row = QHBoxLayout()
        self.btn_activate = QPushButton("Activar modelo seleccionado")
        self.btn_activate.clicked.connect(self._activate)
        row.addWidget(self.btn_activate)
        row.addStretch(1)
        root.addLayout(row)

        root.addWidget(QLabel("Predicciones recientes"))
        self.preds = QTableWidget()
        self.preds.setEditTriggers(QTableWidget.NoEditTriggers)
        root.addWidget(self.preds, 1)
        self._models: list[dict] = []

    def _activate(self) -> None:
        idx = self.models.currentRow()
        if idx < 0 or idx >= len(self._models):
            return
        m = self._models[idx]
        if not (m.get("metrics") or {}).get("approved"):
            QMessageBox.warning(self, "Modelo no aprobado",
                                "Este modelo no superó los criterios walk-forward "
                                "(EV neto > 0, folds positivos, mejor que baseline). "
                                "No se puede activar.")
            return
        try:
            self.engine.activate_model(m["version"])
        except Exception as exc:
            QMessageBox.warning(self, "Activar modelo", str(exc))
        self.refresh()

    def on_event(self, topic: str, p: dict) -> None:
        if topic == ev.PREDICTION:
            self.lbl_pred.setText(f"Predicción {p['symbol']}: P(sube) = {p['prob_up']:.3f} "
                                  f"· {p['model_version']} · {p.get('reason', '')}")

    def refresh(self) -> None:
        self._models = self.engine.registry.list()
        rows = []
        for m in self._models:
            wf = (m.get("metrics") or {}).get("walk_forward") or {}
            rows.append([m["version"], m["kind"], "sí" if m["active"] else "",
                         "sí" if (m.get("metrics") or {}).get("approved") else "no",
                         wf.get("trades"), wf.get("win_rate"), wf.get("ev_per_trade"),
                         wf.get("breakeven_win_rate"), wf.get("auc")])
        fill_table(self.models, ["Versión", "Tipo", "Activo", "Aprobado", "Trades",
                                 "Win rate", "EV/trade", "Breakeven", "AUC"], rows)
        preds = self.engine.predictions.recent(limit=100)
        fill_table(self.preds, ["Epoch", "Símbolo", "P(sube)", "Dirección", "Operado",
                                "Modelo"],
                   [[p["tick_epoch"], p["symbol"], p["prob_up"], p["direction"],
                     "sí" if p["acted"] else "", p["model_version"]] for p in preds])
