"""Configuración: entorno (Real bloqueado) y límites de riesgo."""

from __future__ import annotations

from dataclasses import fields

from PySide6.QtWidgets import (QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QPushButton,
                               QRadioButton, QSpinBox, QVBoxLayout, QWidget)

from risk.risk_manager import RiskLimits

LABELS = {
    "max_stake": "Importe máximo por operación",
    "max_daily_loss": "Pérdida diaria máxima",
    "max_open_contracts": "Contratos simultáneos máx.",
    "max_trades_per_hour": "Operaciones por hora máx.",
    "min_seconds_between_trades": "Segundos mínimos entre operaciones",
    "max_consecutive_errors": "Errores seguidos antes de pausa",
    "error_pause_s": "Pausa tras errores (s)",
    "max_consecutive_losses": "Pérdidas seguidas antes de pausa",
    "loss_streak_pause_s": "Pausa tras racha de pérdidas (s)",
    "min_payout_ratio": "Payout/precio mínimo",
    "max_price_slippage": "Desviación máx. de precio (fracción)",
    "min_balance": "Saldo mínimo",
}


class SettingsView(QWidget):
    def __init__(self, engine) -> None:
        super().__init__()
        self.engine = engine
        root = QVBoxLayout(self)

        env = QGroupBox("Entorno")
        env_l = QVBoxLayout(env)
        self.rb_demo = QRadioButton("Demo")
        self.rb_demo.setChecked(True)
        self.rb_real = QRadioButton("Real — bloqueado (Fase 8: revisión manual, legal y de "
                                    "seguridad)")
        self.rb_real.setEnabled(False)
        self.rb_real.setToolTip("El modo Real no puede activarse desde la aplicación.")
        acc = engine.session.get("account_id")
        env_l.addWidget(self.rb_demo)
        env_l.addWidget(self.rb_real)
        env_l.addWidget(QLabel(f"Cuenta: {acc or 'sin autenticar (solo datos públicos)'}"))
        root.addWidget(env)

        risk = QGroupBox("Límites de riesgo")
        form = QFormLayout(risk)
        current = engine.controller.risk.limits if engine.controller else RiskLimits()
        self.inputs: dict[str, QSpinBox | QDoubleSpinBox] = {}
        for f in fields(RiskLimits):
            value = getattr(current, f.name)
            if isinstance(value, int):
                w = QSpinBox()
                w.setRange(0, 1_000_000)
            else:
                w = QDoubleSpinBox()
                w.setDecimals(3)
                w.setRange(0, 1_000_000)
            w.setValue(value)
            self.inputs[f.name] = w
            form.addRow(LABELS.get(f.name, f.name), w)
        self.btn_apply = QPushButton("Aplicar límites")
        self.btn_apply.clicked.connect(self.apply)
        self.btn_apply.setEnabled(engine.controller is not None)
        form.addRow(self.btn_apply)
        root.addWidget(risk)
        root.addStretch(1)

    def limits(self) -> RiskLimits:
        return RiskLimits(**{k: (int(w.value()) if isinstance(w, QSpinBox) else float(w.value()))
                             for k, w in self.inputs.items()})

    def apply(self) -> None:
        self.engine.update_risk_limits(self.limits())
