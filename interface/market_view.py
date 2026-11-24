"""Mercado: selección de símbolo, gráfico de ticks y orden manual (vía motor)."""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QMessageBox, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from core import event_manager as ev
from strategy.signals import Direction


def format_quote(quote: float, pip_size) -> str:
    """Deriv envía ``pip_size`` como número de decimales a mostrar."""
    decimals = int(pip_size) if isinstance(pip_size, (int, float)) and 0 <= pip_size <= 8 else 4
    return f"{quote:.{decimals}f}"


class Sparkline(QWidget):
    def __init__(self, maxlen: int = 300) -> None:
        super().__init__()
        self.values: deque[float] = deque(maxlen=maxlen)
        self.setMinimumHeight(220)

    def add(self, value: float) -> None:
        self.values.append(value)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - API Qt
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), self.palette().base())
        if len(self.values) < 2:
            p.drawText(self.rect(), Qt.AlignCenter, "Esperando ticks…")
            return
        lo, hi = min(self.values), max(self.values)
        span = (hi - lo) or 1.0
        w, h, pad = self.width(), self.height(), 10
        n = len(self.values)
        pts = QPolygonF([QPointF(pad + i * (w - 2 * pad) / (n - 1),
                                 h - pad - (v - lo) / span * (h - 2 * pad))
                         for i, v in enumerate(self.values)])
        p.setPen(QPen(QColor("#2b6cb0"), 1.6))
        p.drawPolyline(pts)
        p.setPen(self.palette().text().color())
        p.drawText(pad, pad + 10, f"{hi:.4f}")
        p.drawText(pad, h - pad, f"{lo:.4f}")


class MarketView(QWidget):
    def __init__(self, engine) -> None:
        super().__init__()
        self.engine = engine
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.symbol_combo = QComboBox()
        self.symbol_combo.setMinimumWidth(320)
        self.btn_load = QPushButton("Cargar símbolos")
        self.btn_load.clicked.connect(self.load_symbols)
        self.btn_watch = QPushButton("Suscribir ticks")
        self.btn_watch.clicked.connect(self._watch)
        self.lbl_quote = QLabel("—")
        self.lbl_quote.setStyleSheet("font-size: 20px; font-weight: 600;")
        for w in (QLabel("Símbolo:"), self.symbol_combo, self.btn_load, self.btn_watch):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.lbl_quote)
        root.addLayout(top)

        self.chart = Sparkline()
        root.addWidget(self.chart, 1)

        manual = QGroupBox("Orden manual (pasa por ExecutionController y riesgo)")
        form = QFormLayout(manual)
        self.direction = QComboBox()
        self.direction.addItem("Sube (CALL)", Direction.UP)
        self.direction.addItem("Baja (PUT)", Direction.DOWN)
        self.stake = QDoubleSpinBox()
        self.stake.setRange(0.35, 1000)
        self.stake.setValue(1.0)
        self.duration = QSpinBox()
        self.duration.setRange(1, 10)
        self.duration.setValue(5)
        self.duration.setSuffix(" ticks")
        self.btn_order = QPushButton("Enviar orden Demo")
        self.btn_order.clicked.connect(self._manual_order)
        form.addRow("Dirección", self.direction)
        form.addRow("Importe", self.stake)
        form.addRow("Duración", self.duration)
        form.addRow(self.btn_order)
        root.addWidget(manual)

        root.addWidget(QLabel("Calidad de datos"))
        self.issues = QListWidget()
        self.issues.setMaximumHeight(120)
        root.addWidget(self.issues)

    def load_symbols(self) -> None:
        try:
            symbols = self.engine.available_symbols()
        except Exception as exc:
            QMessageBox.warning(self, "Símbolos", f"No se pudieron cargar: {exc}")
            return
        self.symbol_combo.clear()
        for s in sorted(symbols, key=lambda s: (s.market, s.underlying_symbol)):
            self.symbol_combo.addItem(f"{s.underlying_symbol} — {s.underlying_symbol_name}",
                                      s.underlying_symbol)

    def _watch(self) -> None:
        symbol = self.symbol_combo.currentData()
        if symbol:
            self.chart.values.clear()
            self.engine.watch_symbol(symbol)

    def _manual_order(self) -> None:
        try:
            res = self.engine.manual_order(self.direction.currentData(), self.stake.value(),
                                           self.duration.value(), "t")
        except Exception as exc:
            QMessageBox.warning(self, "Orden manual", str(exc))
            return
        if res is None:
            QMessageBox.information(self, "Orden manual", "Modo solo datos: sin cuenta Demo.")
        elif res.status != "open":
            QMessageBox.warning(self, "Orden manual",
                                f"{res.status}: {'; '.join(res.reasons)}")

    def on_event(self, topic: str, p: dict) -> None:
        if topic == ev.TICK:
            self.chart.add(p["quote"])
            self.lbl_quote.setText(f"{p['symbol']}  {format_quote(p['quote'], p.get('pip_size'))}")
        elif topic == ev.DATA_ISSUE:
            self.issues.insertItem(0, f"{p.get('symbol')}: {p.get('issue')} {p.get('detail')}")
            while self.issues.count() > 200:
                self.issues.takeItem(self.issues.count() - 1)
        elif topic == ev.ENGINE_STATE:
            self.btn_order.setEnabled(p.get("mode") == "demo" and p.get("status") != "halted")
