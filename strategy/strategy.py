"""Interfaces de estrategia y predictor (desacopladas del broker y de la ejecución).

Una estrategia NUNCA ejecuta: devuelve un ``OrderRequest`` que el TradingEngine
envía al ExecutionController.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Sequence

from market.market_data import Tick
from strategy.signals import OrderRequest


@dataclass(frozen=True)
class Prediction:
    prob_up: float
    model_version: str
    tick_epoch: int          # último tick usado: garantiza que no se usaron datos futuros


class Predictor(Protocol):
    model_version: str
    min_history: int

    def predict(self, ticks: Sequence[Tick]) -> Prediction | None: ...


@dataclass(frozen=True)
class Decision:
    prediction: Prediction | None
    order: OrderRequest | None
    reason: str = ""


class Strategy(ABC):
    @abstractmethod
    def evaluate(self, symbol: str, ticks: Sequence[Tick], balance: float, now: float,
                 market_ok: bool = True) -> Decision: ...

    def on_settled(self, profit: float) -> None:  # noqa: B027 - gancho opcional
        pass
