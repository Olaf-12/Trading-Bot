"""Tipos de señal y orden compartidos por estrategia, riesgo y ejecución."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"


# Rise/Fall de duración fija. CALL gana si exit > entry (estrictamente); PUT si exit < entry.
# Empate = pérdida en ambos (regla documentada en el longcode: "strictly higher/lower").
CONTRACT_FOR_DIRECTION = {Direction.UP: "CALL", Direction.DOWN: "PUT"}
SUPPORTED_CONTRACT_TYPES = frozenset(CONTRACT_FOR_DIRECTION.values())
DURATION_UNITS = frozenset({"t", "s", "m", "h", "d"})


@dataclass(frozen=True)
class Signal:
    symbol: str
    direction: Direction
    probability: float               # probabilidad estimada de la dirección elegida
    model_version: str
    tick_epoch: int                  # epoch del último tick usado (sin datos futuros)
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    contract_type: str
    duration: int
    duration_unit: str
    stake: float
    currency: str
    basis: str = "stake"

    def __post_init__(self) -> None:
        if self.contract_type not in SUPPORTED_CONTRACT_TYPES:
            raise ValueError(f"Tipo de contrato no soportado: {self.contract_type}")
        if self.duration_unit not in DURATION_UNITS or self.duration <= 0:
            raise ValueError("Duración inválida")
        if not (self.stake > 0):
            raise ValueError("Stake debe ser > 0")
        if self.basis != "stake":
            raise ValueError("Solo se admite basis='stake'")

    def duration_seconds(self) -> int | None:
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(self.duration_unit)
        return None if mult is None else self.duration * mult

    def to_proposal_request(self) -> dict:
        return {"proposal": 1, "amount": round(self.stake, 2), "basis": self.basis,
                "contract_type": self.contract_type, "currency": self.currency,
                "duration": int(self.duration), "duration_unit": self.duration_unit,
                "underlying_symbol": self.symbol}


@dataclass(frozen=True)
class OrderRequest:
    spec: ContractSpec
    signal: Signal | None = None
    client_order_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source: str = "strategy"          # strategy | manual (GUI -> engine -> controller)
