"""Historial: contratos liquidados, órdenes (incluidas las rechazadas) y resumen."""

from __future__ import annotations

import json

from PySide6.QtWidgets import QLabel, QTableWidget, QVBoxLayout, QWidget

from core import event_manager as ev
from interface.ml_view import fill_table


class HistoryView(QWidget):
    def __init__(self, engine) -> None:
        super().__init__()
        self.engine = engine
        root = QVBoxLayout(self)
        self.summary = QLabel("—")
        self.summary.setStyleSheet("font-size: 15px; font-weight: 600;")
        root.addWidget(self.summary)
        root.addWidget(QLabel("Contratos"))
        self.trades = QTableWidget()
        self.trades.setEditTriggers(QTableWidget.NoEditTriggers)
        root.addWidget(self.trades, 2)
        root.addWidget(QLabel("Órdenes"))
        self.orders = QTableWidget()
        self.orders.setEditTriggers(QTableWidget.NoEditTriggers)
        root.addWidget(self.orders, 1)

    def on_event(self, topic: str, _p: dict) -> None:
        if topic in (ev.CONTRACT_SETTLED, ev.RECONCILIATION):
            self.refresh()

    def refresh(self) -> None:
        s = self.engine.trades.summary()
        self.summary.setText(f"Liquidados {s['settled']} · ganados {s['wins']} · perdidos "
                             f"{s['losses']} · win rate {s['win_rate']:.1%} · P&L "
                             f"{s['pnl']:+.2f} sobre {s['staked']:.2f}")
        fill_table(self.trades, ["Contrato", "Símbolo", "Tipo", "Estado", "Precio", "Payout",
                                 "Profit", "Entrada", "Salida", "Modelo", "P(señal)"],
                   [[t["contract_id"], t["symbol"], t["contract_type"], t["status"],
                     t["buy_price"], t["payout"], t["profit"], t["entry_spot"], t["exit_spot"],
                     t["model_version"], t["signal_prob"]]
                    for t in self.engine.trades.recent_trades(300)])
        fill_table(self.orders, ["Orden", "Origen", "Símbolo", "Tipo", "Stake", "Estado",
                                 "Motivos / error"],
                   [[o["client_order_id"][:8], o["source"], o["symbol"], o["contract_type"],
                     o["stake"], o["status"], _reasons(o)]
                    for o in self.engine.trades.recent_orders(300)])


def _reasons(order) -> str:
    raw = order["reject_reasons"]
    if raw:
        try:
            return "; ".join(json.loads(raw))
        except (TypeError, ValueError):
            return str(raw)
    return order["error"] or ""
