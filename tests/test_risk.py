"""Gestión de riesgo y position sizing."""

from __future__ import annotations

import pytest

from risk.position_sizing import SizingConfig, stake_for
from risk.risk_manager import RiskLimits, RiskManager
from strategy.signals import ContractSpec, OrderRequest


class Clock:
    def __init__(self, t=1_700_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def req(stake=1.0):
    return OrderRequest(ContractSpec("R_100", "CALL", 5, "t", stake, "USD"))


def rm(clock=None, **kw):
    return RiskManager(RiskLimits(**kw), clock=clock or Clock())


def test_approves_normal_order():
    assert rm().check(req(), 1000).approved


def test_max_stake_and_min_balance():
    r = rm(max_stake=2, min_balance=10)
    assert not r.check(req(3), 1000).approved
    assert not r.check(req(1), 10.5).approved


def test_frequency_limits():
    c = Clock()
    r = rm(c, min_seconds_between_trades=10, max_trades_per_hour=2, max_open_contracts=10,
           max_daily_loss=100)
    r.record_open("a", 1)
    assert "frecuencia" in r.check(req(), 1000).reasons[0]
    c.t += 11
    r.record_open("b", 1)
    c.t += 11
    assert any("por hora" in x for x in r.check(req(), 1000).reasons)
    c.t += 3600
    assert r.check(req(), 1000).approved


def test_daily_reset_at_utc_midnight():
    c = Clock(1_700_006_399.0)            # 23:59:59 UTC
    r = rm(c, max_daily_loss=5, min_seconds_between_trades=0)
    r.record_open("a", 5)
    r.record_settlement("a", -5)
    assert not r.check(req(), 1000).approved
    c.t += 2                              # nuevo día UTC
    assert r.check(req(), 1000).approved


def test_loss_streak_pause():
    c = Clock()
    r = rm(c, max_consecutive_losses=2, loss_streak_pause_s=60, max_daily_loss=100,
           min_seconds_between_trades=0)
    for oid in ("a", "b"):
        r.record_open(oid, 1)
        r.record_settlement(oid, -1)
    assert "pausa" in r.check(req(), 1000).reasons[0]
    c.t += 61
    assert r.check(req(), 1000).approved


def test_win_resets_loss_streak():
    r = rm(max_consecutive_losses=2, max_daily_loss=100)
    r.record_settlement("a", -1)
    r.record_settlement("b", 0.9)
    r.record_settlement("c", -1)
    assert r.state.consecutive_losses == 1


def test_halt_requires_manual_resume():
    r = rm()
    r.halt("test")
    assert not r.check(req(), 1000).approved
    r.resume()
    assert r.check(req(), 1000).approved


def test_check_proposal():
    r = rm(min_payout_ratio=1.8, max_price_slippage=0.0, max_stake=5)
    assert r.check_proposal(1, 1, 1.9).approved
    assert not r.check_proposal(1, 1, 1.7).approved
    assert not r.check_proposal(1, 1.01, 2).approved


@pytest.mark.parametrize("cfg, balance, expected", [
    (SizingConfig(mode="fixed", fixed_stake=2), 1000, 2.0),
    (SizingConfig(mode="fixed", fixed_stake=20, max_stake=5), 1000, 5.0),
    (SizingConfig(mode="fraction", balance_fraction=0.01, max_stake=50), 1234, 12.34),
    (SizingConfig(mode="fraction", balance_fraction=0.01, min_stake=0.35), 20, 0.0),
])
def test_stake_for(cfg, balance, expected):
    assert stake_for(balance, cfg) == pytest.approx(expected)


def test_can_open_more_precheck():
    c = Clock()
    r = rm(c, max_open_contracts=1, max_daily_loss=100)
    assert r.can_open_more() == (True, "")
    r.record_open("a", 1)
    assert r.can_open_more()[0] is False
    r.record_settlement("a", 1)
    assert r.can_open_more()[0] is True
    r.halt("x")
    assert r.can_open_more() == (False, "parada de emergencia")
