"""Dashboard: estado de cuenta, riesgo, controles y registro de eventos."""

from __future__ import annotations

import time

from PySide6.QtWidgets import (QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

from core import event_manager as ev


def _metric(text: str = "—") -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("class", "metric")
    lbl.setStyleSheet("font-size: 18px; font-weight: 600;")
    return lbl


class DashboardView(QWidget):
    def __init__(self, engine) -> None:
        super().__init__()
        self.engine = engine
        root = QVBoxLayout(self)

        box = QGroupBox("Cuenta y sesión")
        grid = QGridLayout(box)
        self.lbl_account = _metric()
        self.lbl_balance = _metric()
        self.lbl_pnl = _metric()
        self.lbl_trades = _metric()
        self.lbl_open = _metric()
        self.lbl_status = _metric()
        for col, (title, lbl) in enumerate((("Cuenta", self.lbl_account),
                                            ("Saldo", self.lbl_balance),
                                            ("P&L hoy", self.lbl_pnl),
                                            ("Trades hoy", self.lbl_trades),
                                            ("Abiertos", self.lbl_open),
                                            ("Estado", self.lbl_status))):
            grid.addWidget(QLabel(title), 0, col)
            grid.addWidget(lbl, 1, col)
        root.addWidget(box)

        self.lbl_risk = QLabel("")
        self.lbl_risk.setWordWrap(True)
        root.addWidget(self.lbl_risk)

        controls = QHBoxLayout()
        self.btn_auto = QPushButton("Activar auto-trading")
        self.btn_auto.setCheckable(True)
        self.btn_auto.toggled.connect(self._toggle_auto)
        self.btn_resume = QPushButton("Rearmar tras parada")
        self.btn_resume.clicked.connect(self._resume)
        self.btn_emergency = QPushButton("PARADA DE EMERGENCIA")
        self.btn_emergency.setObjectName("emergency")
        self.btn_emergency.clicked.connect(self._emergency)
        controls.addWidget(self.btn_auto)
        controls.addWidget(self.btn_resume)
        controls.addStretch(1)
        controls.addWidget(self.btn_emergency)
        root.addLayout(controls)

        root.addWidget(QLabel("Eventos"))
        self.log = QListWidget()
        root.addWidget(self.log, 1)

    # ---------------------------------------------------------------- acciones
    def _toggle_auto(self, checked: bool) -> None:
        try:
            self.engine.set_auto_trading(checked)
        except RuntimeError as exc:
            self.btn_auto.blockSignals(True)
            self.btn_auto.setChecked(False)
            self.btn_auto.blockSignals(False)
            QMessageBox.warning(self, "Auto-trading", str(exc))
        self.btn_auto.setText("Desactivar auto-trading" if self.btn_auto.isChecked()
                              else "Activar auto-trading")

    def _emergency(self) -> None:
        self.engine.emergency_stop("manual (GUI)")
        self.btn_auto.setChecked(False)

    def _resume(self) -> None:
        if QMessageBox.question(self, "Rearmar",
                                "¿Revisaste la causa de la parada y quieres rearmar?"
                                ) == QMessageBox.Yes:
            self.engine.resume_after_emergency()

    # ----------------------------------------------------------------- vista
    def add_log(self, text: str) -> None:
        self.log.insertItem(0, f"{time.strftime('%H:%M:%S')}  {text}")
        while self.log.count() > 500:
            self.log.takeItem(self.log.count() - 1)

    def on_event(self, topic: str, p: dict) -> None:
        if topic == ev.ORDER_PLACED:
            self.add_log(f"Compra {p.get('contract_type')} {p.get('symbol')} "
                         f"#{p.get('contract_id')} precio {p.get('buy_price')}")
        elif topic == ev.ORDER_REJECTED:
            self.add_log(f"Orden {p.get('status')}: {'; '.join(p.get('reasons', []))}")
        elif topic == ev.CONTRACT_SETTLED:
            self.add_log(f"Liquidado #{p.get('contract_id')} {p.get('status')} "
                         f"{p.get('profit'):+.2f}")
        elif topic == ev.EMERGENCY_STOP:
            self.add_log(f"PARADA DE EMERGENCIA: {p.get('reason')}")
        elif topic == ev.DATA_ISSUE:
            self.add_log(f"Dato rechazado {p.get('symbol')}: {p.get('issue')}")
        elif topic == ev.RECONCILIATION:
            self.add_log(f"Reconciliación: {'OK' if p.get('ok') else 'DISCREPANCIAS'}")

    def refresh(self) -> None:
        s = self.engine.session.snapshot()
        r = self.engine.risk_snapshot()
        self.lbl_account.setText(s["account_id"] or ("solo datos" if s["mode"] == "public"
                                                     else "—"))
        self.lbl_balance.setText("—" if s["balance"] is None
                                 else f"{s['balance']:.2f} {s['currency'] or ''}")
        self.lbl_pnl.setText(f"{r.get('realized_pnl', 0):+.2f}" if r else "—")
        self.lbl_trades.setText(str(r.get("trades_today", 0)) if r else "—")
        self.lbl_open.setText(str(r.get("open_contracts", 0)) if r else "—")
        status = s["status"] + (" · auto" if s["auto_trading"] else "")
        self.lbl_status.setText(status)
        # Sincroniza el botón con el estado real (p. ej. tras parada de emergencia).
        if self.btn_auto.isChecked() != s["auto_trading"]:
            self.btn_auto.blockSignals(True)
            self.btn_auto.setChecked(s["auto_trading"])
            self.btn_auto.blockSignals(False)
        self.btn_auto.setText("Desactivar auto-trading" if s["auto_trading"]
                              else "Activar auto-trading")
        has_ctl = self.engine.controller is not None
        halted = bool(r.get("halted")) if r else False
        self.btn_auto.setEnabled(has_ctl and self.engine.strategy is not None and not halted)
        self.btn_resume.setEnabled(halted)
        self.btn_emergency.setEnabled(has_ctl and not halted)
        if r:
            pause = f" · pausa {r['paused_for_s']:.0f}s ({r['pause_reason']})" \
                if r.get("paused_for_s") else ""
            halt = f" · DETENIDO: {r['halt_reason']}" if halted else ""
            market = "" if s["market_ok"] else f" · datos no aptos ({s['market_issue']})"
            self.lbl_risk.setText(f"Errores seguidos {r['consecutive_errors']} · pérdidas "
                                  f"seguidas {r['consecutive_losses']}{pause}{halt}{market}")
