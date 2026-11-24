"""Repositorio de órdenes y trades (contratos)."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from broker.broker_interface import BuyReceipt, ContractUpdate, Proposal
from database.database import Database
from strategy.signals import OrderRequest

OPEN_STATUSES = ("buying", "open", "unknown")


class TradeRepository:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time) -> None:
        self.db = db
        self._clock = clock

    def create_order(self, order: OrderRequest, account_id: str | None, status: str,
                     reasons: tuple[str, ...] = ()) -> None:
        s, sig, now = order.spec, order.signal, self._clock()
        self.db.execute(
            "INSERT INTO orders(client_order_id, created_at, updated_at, account_id, source, "
            "symbol, contract_type, duration, duration_unit, stake, currency, status, "
            "reject_reasons, signal_direction, signal_prob, model_version, signal_epoch) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (order.client_order_id, now, now, account_id, order.source, s.symbol,
             s.contract_type, s.duration, s.duration_unit, s.stake, s.currency, status,
             json.dumps(list(reasons), ensure_ascii=False) if reasons else None,
             sig.direction.value if sig else None, sig.probability if sig else None,
             sig.model_version if sig else None, sig.tick_epoch if sig else None))

    def _update(self, order_id: str, **fields: Any) -> None:
        fields["updated_at"] = self._clock()
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE orders SET {cols} WHERE client_order_id = ?",
                        (*fields.values(), order_id))

    def set_status(self, order_id: str, status: str, error: str | None = None,
                   reasons: tuple[str, ...] = ()) -> None:
        extra: dict[str, Any] = {"status": status}
        if error is not None:
            extra["error"] = error
        if reasons:
            extra["reject_reasons"] = json.dumps(list(reasons), ensure_ascii=False)
        self._update(order_id, **extra)

    def record_proposal(self, order_id: str, p: Proposal) -> None:
        self._update(order_id, status="proposed", proposal_ask=p.ask_price,
                     proposal_payout=p.payout, proposal_spot=p.spot)

    def record_buy(self, order_id: str, r: BuyReceipt) -> None:
        self._update(order_id, status="open", contract_id=r.contract_id,
                     transaction_id=r.transaction_id, buy_price=r.buy_price,
                     payout=r.payout, purchase_time=r.purchase_time)

    def attach_contract(self, order_id: str, contract_id: int, transaction_id: int | None,
                        buy_price: float | None, payout: float | None,
                        purchase_time: int | None) -> None:
        self._update(order_id, status="open", contract_id=contract_id,
                     transaction_id=transaction_id, buy_price=buy_price, payout=payout,
                     purchase_time=purchase_time)

    def upsert_trade(self, order_id: str | None, u: ContractUpdate,
                     model_version: str | None = None) -> None:
        settled = self._clock() if u.is_settled else None
        self.db.execute(
            "INSERT INTO trades(contract_id, client_order_id, account_id, symbol, "
            "contract_type, status, buy_price, payout, sell_price, profit, entry_spot, "
            "entry_time, exit_spot, exit_time, date_expiry, settled_at, model_version, "
            "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(contract_id) DO UPDATE SET status=excluded.status, "
            "sell_price=excluded.sell_price, profit=excluded.profit, "
            "entry_spot=COALESCE(excluded.entry_spot, trades.entry_spot), "
            "entry_time=COALESCE(excluded.entry_time, trades.entry_time), "
            "exit_spot=excluded.exit_spot, exit_time=excluded.exit_time, "
            "settled_at=COALESCE(trades.settled_at, excluded.settled_at), "
            "updated_at=excluded.updated_at",
            (u.contract_id, order_id, u.account_id, u.underlying_symbol, u.contract_type,
             u.status, u.buy_price, u.payout, u.sell_price, u.profit, u.entry_spot,
             u.entry_spot_time, u.exit_spot, u.exit_spot_time, u.date_expiry, settled,
             model_version, self._clock()))
        if u.is_settled and order_id:
            self._update(order_id, status="settled")

    # ------------------------------------------------------------- consultas
    def order(self, order_id: str):
        rows = self.db.query("SELECT * FROM orders WHERE client_order_id = ?", (order_id,))
        return rows[0] if rows else None

    def order_by_contract(self, contract_id: int):
        rows = self.db.query("SELECT * FROM orders WHERE contract_id = ?", (contract_id,))
        return rows[0] if rows else None

    def orders_with_status(self, *statuses: str):
        marks = ",".join("?" * len(statuses))
        return self.db.query(f"SELECT * FROM orders WHERE status IN ({marks}) "
                             "ORDER BY created_at", statuses)

    def open_orders(self):
        return self.orders_with_status(*OPEN_STATUSES)

    def trade(self, contract_id: int):
        rows = self.db.query("SELECT * FROM trades WHERE contract_id = ?", (contract_id,))
        return rows[0] if rows else None

    def recent_trades(self, limit: int = 200):
        return self.db.query("SELECT t.*, o.signal_prob, o.signal_direction FROM trades t "
                             "LEFT JOIN orders o ON o.client_order_id = t.client_order_id "
                             "ORDER BY COALESCE(t.settled_at, t.updated_at) DESC LIMIT ?",
                             (limit,))

    def recent_orders(self, limit: int = 200):
        return self.db.query("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,))

    def summary(self) -> dict[str, Any]:
        row = self.db.query(
            "SELECT COUNT(*) n, SUM(profit > 0) wins, SUM(profit <= 0) losses, "
            "COALESCE(SUM(profit), 0) pnl, COALESCE(SUM(buy_price), 0) staked "
            "FROM trades WHERE settled_at IS NOT NULL")[0]
        n = row["n"] or 0
        return {"settled": n, "wins": row["wins"] or 0, "losses": row["losses"] or 0,
                "win_rate": (row["wins"] or 0) / n if n else 0.0,
                "pnl": round(row["pnl"], 2), "staked": round(row["staked"], 2)}
