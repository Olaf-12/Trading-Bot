"""Estrategia de duración fija (Rise/Fall: CALL/PUT).

Convierte la probabilidad del modelo en señal solo si supera un umbral que,
además, debe estar por encima del punto de equilibrio del payout:
    win_rate_breakeven = 1 / payout_ratio   (p. ej. 1/1.95 = 51.3 %)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from market.market_data import Tick
from risk.position_sizing import SizingConfig, stake_for
from strategy.signals import (CONTRACT_FOR_DIRECTION, ContractSpec, Direction, OrderRequest,
                              Signal)
from strategy.strategy import Decision, Predictor, Strategy


@dataclass(frozen=True)
class StrategyConfig:
    threshold: float = 0.56              # prob. mínima de la dirección elegida
    expected_payout_ratio: float = 1.95  # para validar el umbral frente al breakeven
    min_edge: float = 0.01               # margen mínimo sobre el breakeven
    duration: int = 5
    duration_unit: str = "t"
    currency: str = "USD"
    sizing: SizingConfig = field(default_factory=SizingConfig)
    cooldown_ticks: int = 5              # ticks mínimos entre señales
    max_tick_age_s: float = 5.0
    allow_down: bool = True

    def __post_init__(self) -> None:
        if not 0.5 < self.threshold < 1:
            raise ValueError("threshold debe estar en (0.5, 1)")
        breakeven = 1 / self.expected_payout_ratio
        if self.threshold < breakeven + self.min_edge:
            raise ValueError(f"threshold {self.threshold} no supera el breakeven "
                             f"{breakeven:.3f} + margen {self.min_edge}")


class FixedTimeStrategy(Strategy):
    def __init__(self, predictor: Predictor, config: StrategyConfig | None = None) -> None:
        self.predictor = predictor
        self.config = config or StrategyConfig()
        self._ticks_since_signal = 10**9

    def evaluate(self, symbol: str, ticks: Sequence[Tick], balance: float, now: float,
                 market_ok: bool = True) -> Decision:
        cfg = self.config
        self._ticks_since_signal += 1
        if not ticks:
            return Decision(None, None, "sin ticks")
        last = ticks[-1]
        if now - last.epoch > cfg.max_tick_age_s:
            return Decision(None, None, "último tick obsoleto")
        if len(ticks) < self.predictor.min_history:
            return Decision(None, None, "historial insuficiente")
        pred = self.predictor.predict(ticks)
        if pred is None:
            return Decision(None, None, "sin predicción")
        if pred.tick_epoch > last.epoch:
            raise RuntimeError("El predictor usó datos posteriores al último tick")
        if not market_ok:
            return Decision(pred, None, "condiciones de mercado no aptas")
        if self._ticks_since_signal < cfg.cooldown_ticks:
            return Decision(pred, None, "cooldown")

        if pred.prob_up >= cfg.threshold:
            direction, prob = Direction.UP, pred.prob_up
        elif cfg.allow_down and (1 - pred.prob_up) >= cfg.threshold:
            direction, prob = Direction.DOWN, 1 - pred.prob_up
        else:
            return Decision(pred, None, "probabilidad bajo el umbral")

        stake = stake_for(balance, cfg.sizing)
        if stake <= 0:
            return Decision(pred, None, "stake por debajo del mínimo")
        self._ticks_since_signal = 0
        signal = Signal(symbol, direction, prob, pred.model_version, last.epoch, created_at=now)
        spec = ContractSpec(symbol, CONTRACT_FOR_DIRECTION[direction], cfg.duration,
                            cfg.duration_unit, stake, cfg.currency)
        return Decision(pred, OrderRequest(spec, signal), f"{direction.value} p={prob:.3f}")
