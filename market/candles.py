"""Construcción de velas OHLC a partir de ticks (solo datos pasados)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from market.market_data import Tick


@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    ticks: int

    def update(self, price: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.ticks += 1


class CandleBuilder:
    """Agrega ticks en velas de ``granularity`` segundos. Una vela solo se emite
    como cerrada cuando llega un tick de la siguiente (evita usar velas incompletas)."""

    def __init__(self, granularity_s: int = 60, max_candles: int = 2000) -> None:
        if granularity_s <= 0:
            raise ValueError("granularity_s debe ser > 0")
        self.granularity = granularity_s
        self.max_candles = max_candles
        self.closed: list[Candle] = []
        self.current: Candle | None = None

    def add(self, tick: Tick) -> Candle | None:
        bucket = tick.epoch - tick.epoch % self.granularity
        closed = None
        if self.current is None or bucket > self.current.open_time:
            if self.current is not None:
                closed = self.current
                self.closed.append(closed)
                if len(self.closed) > self.max_candles:
                    del self.closed[: len(self.closed) - self.max_candles]
            self.current = Candle(bucket, tick.quote, tick.quote, tick.quote, tick.quote, 1)
        elif bucket == self.current.open_time:
            self.current.update(tick.quote)
        # bucket < current: tick atrasado, ya filtrado por el validador; se ignora.
        return closed


def build_candles(ticks: Iterable[Tick], granularity_s: int = 60) -> list[Candle]:
    b = CandleBuilder(granularity_s, max_candles=10**9)
    for t in ticks:
        b.add(t)
    return list(b.closed)


def closes(candles: Sequence[Candle]) -> list[float]:
    return [c.close for c in candles]
