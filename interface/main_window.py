"""Ventana principal (PySide6). La GUI NO contiene lógica de trading: solo envía
comandos al TradingEngine y muestra estado/eventos."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QTabWidget

from core import event_manager as ev
from interface.dashboard import DashboardView
from interface.history_view import HistoryView
from interface.market_view import MarketView
from interface.ml_view import MLView
from interface.settings_view import SettingsView

STYLE = """
QMainWindow, QWidget { font-size: 13px; }
QLabel#envBadge { background: #1f7a3a; color: white; padding: 3px 10px; border-radius: 4px;
                  font-weight: bold; }
QPushButton#emergency { background: #b3261e; color: white; font-weight: bold; padding: 10px; }
QPushButton#emergency:disabled { background: #888; }
QLabel.metric { font-size: 18px; font-weight: 600; }
QGroupBox { font-weight: 600; margin-top: 8px; }
"""


class EventBridge(QObject):
    """Reemite eventos del bus (hilos de trabajo) como señal Qt (hilo de GUI)."""

    event = Signal(str, dict)

    def __init__(self, events: ev.EventManager) -> None:
        super().__init__()
        events.subscribe(ev.ALL, self._forward)

    def _forward(self, topic: str, payload: dict[str, Any]) -> None:
        self.event.emit(topic, dict(payload))


class MainWindow(QMainWindow):
    def __init__(self, engine) -> None:
        super().__init__()
        self.engine = engine
        self.setWindowTitle("TradingBotML — Deriv (Demo)")
        self.resize(1200, 780)
        self.setStyleSheet(STYLE)

        self.bridge = EventBridge(engine.events)
        self.tabs = QTabWidget()
        self.dashboard = DashboardView(engine)
        self.market = MarketView(engine)
        self.ml = MLView(engine)
        self.history = HistoryView(engine)
        self.settings = SettingsView(engine)
        for widget, name in ((self.dashboard, "Dashboard"), (self.market, "Mercado"),
                             (self.ml, "ML"), (self.history, "Historial"),
                             (self.settings, "Configuración")):
            self.tabs.addTab(widget, name)
        self.setCentralWidget(self.tabs)

        self.env_badge = QLabel("DEMO")
        self.env_badge.setObjectName("envBadge")
        self.conn_label = QLabel("desconectado")
        self.statusBar().addPermanentWidget(self.conn_label)
        self.statusBar().addPermanentWidget(self.env_badge)

        self.bridge.event.connect(self._on_event, Qt.QueuedConnection)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh_timer.start(2000)
        self.refresh()

    def _on_event(self, topic: str, payload: dict) -> None:
        for view in (self.dashboard, self.market, self.ml, self.history):
            view.on_event(topic, payload)
        if topic == ev.ENGINE_STATE:
            self.conn_label.setText(payload.get("connection", ""))
            self.env_badge.setText(payload.get("environment", "demo").upper())
        elif topic == ev.EMERGENCY_STOP:
            self.statusBar().showMessage(f"PARADA DE EMERGENCIA: {payload.get('reason')}")

    def refresh(self) -> None:
        self.dashboard.refresh()
        if self.tabs.currentWidget() in (self.history, self.ml):
            self.tabs.currentWidget().refresh()

    def closeEvent(self, event) -> None:  # noqa: N802 - API Qt
        open_ids = self.engine.controller.open_contract_ids() if self.engine.controller else []
        if open_ids:
            answer = QMessageBox.question(
                self, "Contratos abiertos",
                f"Hay {len(open_ids)} contrato(s) abiertos. Se liquidarán en Deriv aunque "
                "cierres la aplicación. ¿Salir?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        self.refresh_timer.stop()
        event.accept()
