"""Dobles de prueba compartidos: servidor WebSocket y sesión HTTP falsos."""

from __future__ import annotations

import json
import queue
import time
from typing import Any, Callable

from broker.deriv_ws_client import RecvTimeout
from config import ConnectionSettings

FAST = ConnectionSettings(connect_timeout_s=2, request_timeout_s=2, recv_poll_s=0.02,
                          ping_interval_s=60, reconnect_initial_delay_s=0.01,
                          reconnect_max_delay_s=0.05)

ACTIVE_SYMBOLS = {
    "msg_type": "active_symbols",
    "active_symbols": [
        {"underlying_symbol": "R_100", "underlying_symbol_name": "Volatility 100 Index",
         "underlying_symbol_type": "stockindex", "market": "synthetic_index",
         "submarket": "random_index", "subgroup": "synthetics", "pip_size": 0.01,
         "exchange_is_open": 1, "is_trading_suspended": 0, "trade_count": 0},
        {"underlying_symbol": "frxEURUSD", "underlying_symbol_name": "EUR/USD",
         "underlying_symbol_type": "forex", "market": "forex", "submarket": "major_pairs",
         "subgroup": "none", "pip_size": 0.00001, "exchange_is_open": 0,
         "is_trading_suspended": 0, "trade_count": 0},
    ],
}


def wait_until(pred: Callable[[], bool], timeout: float = 2.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


# --------------------------------------------------------------------------
# Servidor falso
# --------------------------------------------------------------------------
class FakeConnection:
    def __init__(self, server: "FakeServer") -> None:
        self.server = server
        self.inbox: "queue.Queue[str | None]" = queue.Queue()
        self.closed = False
        self.timeout = 1.0

    def settimeout(self, t):
        self.timeout = t

    def send(self, payload: str) -> None:
        if self.closed:
            raise ConnectionError("closed")
        msg = json.loads(payload)
        self.server.received.append(msg)
        for reply in self.server.handle(msg):
            self.push(reply)

    def push(self, message: dict[str, Any]) -> None:
        self.inbox.put(json.dumps(message))

    def recv(self) -> str:
        try:
            item = self.inbox.get(timeout=self.timeout)
        except queue.Empty:
            raise RecvTimeout() from None
        if item is None:
            raise ConnectionError("server closed")
        return item

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.inbox.put(None)

    def drop(self) -> None:  # caída simulada del servidor
        self.close()


class FakeServer:
    def __init__(self) -> None:
        self.connections: list[FakeConnection] = []
        self.received: list[dict[str, Any]] = []
        self.urls: list[str] = []
        self.stream_seq = 0
        self.stream_ids: dict[str, list[str]] = {}
        self.auto_reply = True
        self.fail_next_connects = 0
        self.loginid = "DOT90004580"
        self.currency = "USD"
        # Trading (Fase 3)
        self.proposal_seq = 0
        self.contract_seq = 0
        self.payout_ratio = 1.95
        self.ask_factor = 1.0
        self.proposal_age = 0
        self.proposal_error: str | None = None
        self.buy_mode = "ok"            # ok | reject | timeout
        self.bought: list[dict[str, Any]] = []
        self.poc_streams: dict[int, str] = {}
        self.poc_account: str | None = None
        self.portfolio: list[dict[str, Any]] = []
        self.profit_table: list[dict[str, Any]] = []

    def poc_message(self, cid: int, status: str = "open", profit: float = 0.0,
                    buy_price: float = 1.0, req_id: int | None = None) -> dict[str, Any]:
        settled = status in ("won", "lost", "sold")
        m = {"msg_type": "proposal_open_contract",
             "subscription": {"id": self.poc_streams.get(cid, "none-" + "z" * 30)},
             "proposal_open_contract": {
                 "contract_id": cid, "status": status, "is_sold": int(settled),
                 "is_expired": int(settled), "buy_price": f"{buy_price:.2f}",
                 "payout": f"{buy_price * self.payout_ratio:.2f}", "profit": f"{profit:.2f}",
                 "sell_price": f"{buy_price + profit:.2f}" if settled else None,
                 "entry_spot": "1000.00", "entry_spot_time": int(time.time()),
                 "exit_spot": "1001.00" if settled else None,
                 "exit_spot_time": int(time.time()) if settled else None,
                 "underlying_symbol": "R_100", "contract_type": "CALL", "currency": "USD",
                 "account_id": self.poc_account or self.loginid}}
        if req_id is not None:
            m["req_id"] = req_id
        return m

    def settle(self, cid: int, won: bool, buy_price: float = 1.0) -> None:
        profit = round(buy_price * (self.payout_ratio - 1), 2) if won else -buy_price
        self.conn.push(self.poc_message(cid, "won" if won else "lost", profit, buy_price))

    def factory(self, url: str, timeout: float) -> FakeConnection:
        self.urls.append(url)
        if self.fail_next_connects:
            self.fail_next_connects -= 1
            raise ConnectionRefusedError("down")
        conn = FakeConnection(self)
        self.connections.append(conn)
        return conn

    @property
    def conn(self) -> FakeConnection:
        return self.connections[-1]

    def handle(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.auto_reply:
            return []
        rid = msg.get("req_id")
        if "active_symbols" in msg:
            return [{**ACTIVE_SYMBOLS, "req_id": rid, "echo_req": msg}]
        if "ticks_history" in msg:
            n = int(msg.get("count", 10))
            now = int(time.time())
            times = list(range(now - n, now))
            prices = [1000 + (i % 7) * 0.1 for i in range(n)]
            if n > 2:                                  # duplicado para probar filtrado
                times[2] = times[1]
            return [{"msg_type": "history", "req_id": rid, "echo_req": msg, "pip_size": 2,
                     "history": {"prices": prices, "times": times}}]
        if "ticks" in msg:
            self.stream_seq += 1
            sid = f"stream-{self.stream_seq:04d}-" + "x" * 24
            self.stream_ids.setdefault(msg["ticks"], []).append(sid)
            return [{"msg_type": "tick", "req_id": rid, "echo_req": msg,
                     "subscription": {"id": sid},
                     "tick": {"symbol": msg["ticks"], "epoch": int(time.time()) + self.stream_seq - 1,
                              "quote": 1000.0, "pip_size": 2, "id": sid}}]
        if "forget" in msg:
            return [{"msg_type": "forget", "forget": 1, "req_id": rid}]
        if "balance" in msg:
            return [{"msg_type": "balance", "req_id": rid, "echo_req": msg,
                     "balance": {"balance": 10000.0, "currency": self.currency,
                                 "loginid": self.loginid}}]
        if "proposal" in msg and "proposal_open_contract" not in msg:
            if self.proposal_error:
                return [{"msg_type": "proposal", "req_id": rid, "echo_req": msg,
                         "error": {"code": "ContractBuyValidationError",
                                   "message": self.proposal_error}}]
            self.proposal_seq += 1
            now = int(time.time())
            amount = float(msg["amount"])
            return [{"msg_type": "proposal", "req_id": rid, "echo_req": msg, "proposal": {
                "id": f"prop-{self.proposal_seq}", "ask_price": round(amount * self.ask_factor, 2),
                "payout": round(amount * self.payout_ratio, 2), "spot": 1000.0,
                "spot_time": now - self.proposal_age, "date_start": now,
                "date_expiry": now + int(msg["duration"]), "longcode": "Win payout if ..."}}]
        if "buy" in msg:
            if self.buy_mode == "reject":
                return [{"msg_type": "buy", "req_id": rid, "echo_req": msg,
                         "error": {"code": "InvalidContractProposal", "message": "expired"}}]
            if self.buy_mode == "timeout":
                return []
            self.contract_seq += 1
            cid = 250000 + self.contract_seq
            self.bought.append({"contract_id": cid, "proposal_id": msg["buy"],
                                "price": msg["price"]})
            return [{"msg_type": "buy", "req_id": rid, "echo_req": msg, "buy": {
                "contract_id": cid, "transaction_id": 900000 + self.contract_seq,
                "buy_price": msg["price"], "payout": round(msg["price"] * self.payout_ratio, 2),
                "balance_after": 10000.0 - msg["price"], "purchase_time": int(time.time()),
                "start_time": int(time.time()), "longcode": "Win payout if ...",
                "shortcode": f"CALL_R_100_{cid}"}}]
        if "proposal_open_contract" in msg:
            cid = msg["contract_id"]
            self.stream_seq += 1
            sid = f"poc-{self.stream_seq:04d}-" + "y" * 28
            self.poc_streams[cid] = sid
            return [self.poc_message(cid, status="open", req_id=rid)]
        if "portfolio" in msg:
            return [{"msg_type": "portfolio", "req_id": rid, "echo_req": msg,
                     "portfolio": {"contracts": self.portfolio}}]
        if "profit_table" in msg:
            return [{"msg_type": "profit_table", "req_id": rid, "echo_req": msg,
                     "profit_table": {"count": len(self.profit_table),
                                      "transactions": self.profit_table}}]
        if "ping" in msg:
            return [{"msg_type": "ping", "ping": "pong", "req_id": rid}]
        return [{"msg_type": "error", "req_id": rid,
                 "error": {"code": "UnrecognisedRequest", "message": "?"}}]

    def last_stream_id(self, symbol: str) -> str:
        return self.stream_ids[symbol][-1]


def tick_msg(symbol, epoch, quote, sid, req_id=None):
    m = {"msg_type": "tick", "subscription": {"id": sid},
         "tick": {"symbol": symbol, "epoch": epoch, "quote": quote, "pip_size": 2}}
    if req_id is not None:
        m["req_id"] = req_id
    return m


# --------------------------------------------------------------------------
# REST falso (requests.Session)
# --------------------------------------------------------------------------
DEMO_ACCOUNT = {"account_id": "DOT90004580", "balance": 10000, "currency": "USD",
                "group": "row", "status": "active", "account_type": "demo"}
REAL_ACCOUNT = {"account_id": "ROT90000001", "balance": 50, "currency": "USD",
                "group": "row", "status": "active", "account_type": "real"}
META = {"endpoint": "/x", "method": "GET", "timing": 1}


class FakeResponse:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeHttpSession:
    """Emula GET /accounts y POST /accounts/{id}/otp."""

    def __init__(self, accounts: list[dict[str, Any]] | None = None,
                 ws_host: str = "api.derivws.com", ws_path: str = "/trading/v1/options/ws/demo"):
        self.accounts = [DEMO_ACCOUNT] if accounts is None else accounts
        self.ws_host = ws_host
        self.ws_path = ws_path
        self.calls: list[dict[str, Any]] = []
        self.otp_count = 0
        self.next_error: FakeResponse | None = None

    def _record(self, method: str, url: str, kw: dict[str, Any]) -> FakeResponse | None:
        self.calls.append({"method": method, "url": url, **kw})
        if self.next_error is not None:
            err, self.next_error = self.next_error, None
            return err
        return None

    def get(self, url: str, **kw: Any) -> FakeResponse:
        err = self._record("GET", url, kw)
        if err:
            return err
        assert url.endswith("/trading/v1/options/accounts")
        return FakeResponse(200, {"data": self.accounts, "meta": META})

    def post(self, url: str, **kw: Any) -> FakeResponse:
        err = self._record("POST", url, kw)
        if err:
            return err
        assert url.endswith("/otp")
        self.otp_count += 1
        return FakeResponse(200, {
            "data": {"url": f"wss://{self.ws_host}{self.ws_path}?otp=SECRETOTP{self.otp_count:04d}"},
            "meta": META})


def fake_secrets(name: str) -> str:
    return {"DERIV_TOKEN": "pat_FAKE_TOKEN_123", "DERIV_APP_ID": "12345"}[name]
