"""Cálculo del importe (stake). Nunca martingala: el importe no crece tras pérdidas."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SizingConfig:
    mode: str = "fixed"              # fixed | fraction
    fixed_stake: float = 1.0
    balance_fraction: float = 0.01   # 1 % del saldo en modo "fraction"
    min_stake: float = 0.35          # mínimo habitual en Deriv; validar con la propuesta
    max_stake: float = 10.0


def stake_for(balance: float, cfg: SizingConfig) -> float:
    if cfg.mode == "fixed":
        stake = cfg.fixed_stake
    elif cfg.mode == "fraction":
        stake = balance * cfg.balance_fraction
    else:
        raise ValueError(f"Modo de sizing desconocido: {cfg.mode}")
    stake = min(stake, cfg.max_stake)
    stake = math.floor(stake * 100) / 100          # 2 decimales hacia abajo
    return stake if stake >= cfg.min_stake else 0.0
