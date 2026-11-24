"""Persistencia SQLite de TradingBotML.

Migraciones versionadas (``schema_version``):
  1. ticks validados, calidad de datos, auditoría y ajustes no secretos.
  2. órdenes, trades (contratos), predicciones, registro de modelos y reconciliación.

Garantías de seguridad:
- ``set_setting`` rechaza claves que parezcan secretos (token, otp, password...).
- ``set_setting`` rechaza persistir el entorno Real: Real nunca se restaura desde BD.
- ``log_event`` redacta el texto antes de guardarlo.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from config import redact

SCHEMA_VERSION = 2

_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS ticks (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol      TEXT    NOT NULL,
        epoch       INTEGER NOT NULL,
        quote       REAL    NOT NULL,
        pip_size    REAL,
        received_at REAL    NOT NULL,
        UNIQUE(symbol, epoch)
    );
    CREATE INDEX IF NOT EXISTS ix_ticks_symbol_epoch ON ticks(symbol, epoch);

    CREATE TABLE IF NOT EXISTS data_quality_events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         REAL NOT NULL,
        symbol     TEXT,
        issue      TEXT NOT NULL,
        detail     TEXT
    );

    CREATE TABLE IF NOT EXISTS audit_events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         REAL NOT NULL,
        level      TEXT NOT NULL,
        source     TEXT NOT NULL,
        event      TEXT NOT NULL,
        payload    TEXT
    );

    CREATE TABLE IF NOT EXISTS settings (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_at REAL NOT NULL
    );
    """,
    2: """
    CREATE TABLE IF NOT EXISTS orders (
        client_order_id  TEXT PRIMARY KEY,
        created_at       REAL NOT NULL,
        updated_at       REAL NOT NULL,
        account_id       TEXT,
        source           TEXT NOT NULL,
        symbol           TEXT NOT NULL,
        contract_type    TEXT NOT NULL,
        duration         INTEGER NOT NULL,
        duration_unit    TEXT NOT NULL,
        stake            REAL NOT NULL,
        currency         TEXT NOT NULL,
        status           TEXT NOT NULL,      -- rejected|proposed|buying|open|unknown|settled|failed
        reject_reasons   TEXT,
        proposal_ask     REAL,
        proposal_payout  REAL,
        proposal_spot    REAL,
        contract_id      INTEGER UNIQUE,
        transaction_id   INTEGER,
        buy_price        REAL,
        payout           REAL,
        purchase_time    INTEGER,
        signal_direction TEXT,
        signal_prob      REAL,
        model_version    TEXT,
        signal_epoch     INTEGER,
        error            TEXT
    );
    CREATE INDEX IF NOT EXISTS ix_orders_status ON orders(status);

    CREATE TABLE IF NOT EXISTS trades (
        contract_id      INTEGER PRIMARY KEY,
        client_order_id  TEXT REFERENCES orders(client_order_id),
        account_id       TEXT,
        symbol           TEXT,
        contract_type    TEXT,
        status           TEXT,
        buy_price        REAL,
        payout           REAL,
        sell_price       REAL,
        profit           REAL,
        entry_spot       REAL,
        entry_time       INTEGER,
        exit_spot        REAL,
        exit_time        INTEGER,
        date_expiry      INTEGER,
        settled_at       REAL,
        model_version    TEXT,
        updated_at       REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        ts            REAL NOT NULL,
        symbol        TEXT NOT NULL,
        tick_epoch    INTEGER NOT NULL,
        model_version TEXT NOT NULL,
        prob_up       REAL NOT NULL,
        direction     TEXT,
        acted         INTEGER NOT NULL DEFAULT 0,
        client_order_id TEXT
    );
    CREATE INDEX IF NOT EXISTS ix_predictions_symbol ON predictions(symbol, tick_epoch);

    CREATE TABLE IF NOT EXISTS models (
        version       TEXT PRIMARY KEY,
        created_at    REAL NOT NULL,
        kind          TEXT NOT NULL,
        symbol        TEXT,
        path          TEXT NOT NULL,
        sha256        TEXT NOT NULL,
        features      TEXT NOT NULL,
        params        TEXT,
        metrics       TEXT,
        train_from    INTEGER,
        train_to      INTEGER,
        active        INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS reconciliation_runs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        ts            REAL NOT NULL,
        ok            INTEGER NOT NULL,
        report        TEXT NOT NULL
    );
    """,
}

_FORBIDDEN_SETTING_FRAGMENTS = ("token", "secret", "password", "otp", "api_key",
                                "apikey", "authorization", "credential")
_FORBIDDEN_SETTING_KEYS = ("environment", "account_environment", "env")


class ForbiddenSettingError(ValueError):
    """Se intentó persistir un secreto o el entorno de cuenta."""


class Database:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + lock: el hilo receptor WS escribe ticks.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
        self.migrate()

    # ------------------------------------------------------------------ schema
    def migrate(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = self._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = row[0] or 0
            for version in sorted(v for v in _MIGRATIONS if v > current):
                self._conn.executescript(_MIGRATIONS[version])
                self._conn.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))

    def schema_version(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]

    # ------------------------------------------------------------------- ticks
    def insert_tick(self, symbol: str, epoch: int, quote: float,
                    pip_size: float | None, received_at: float | None = None) -> bool:
        """Inserta un tick validado. Devuelve False si ya existía (duplicado)."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO ticks(symbol, epoch, quote, pip_size, received_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (symbol, int(epoch), float(quote), pip_size, received_at or time.time()),
            )
            return cur.rowcount == 1

    def recent_ticks(self, symbol: str, limit: int = 100) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT symbol, epoch, quote, pip_size FROM ticks WHERE symbol = ? "
                "ORDER BY epoch DESC LIMIT ?", (symbol, limit)).fetchall()
        return list(reversed(rows))

    # ------------------------------------------------------------ data quality
    def log_data_issue(self, symbol: str | None, issue: str, detail: str = "") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO data_quality_events(ts, symbol, issue, detail) VALUES (?,?,?,?)",
                (time.time(), symbol, issue, redact(detail)))

    def data_issues(self, symbol: str | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM data_quality_events"
        args: Iterable[Any] = ()
        if symbol:
            q += " WHERE symbol = ?"
            args = (symbol,)
        with self._lock:
            return self._conn.execute(q + " ORDER BY id", tuple(args)).fetchall()

    # ------------------------------------------------------------------- audit
    def log_event(self, source: str, event: str, payload: Any = None,
                  level: str = "INFO") -> None:
        text = None if payload is None else redact(
            payload if isinstance(payload, str) else json.dumps(payload, default=str))
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO audit_events(ts, level, source, event, payload) VALUES (?,?,?,?,?)",
                (time.time(), level, source, event, text))

    def audit_events(self, source: str | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM audit_events", ()
        if source:
            q, args = q + " WHERE source = ?", (source,)
        with self._lock:
            return self._conn.execute(q + " ORDER BY id", args).fetchall()

    # ---------------------------------------------------------------- settings
    def set_setting(self, key: str, value: Any) -> None:
        lowered = key.lower()
        if any(f in lowered for f in _FORBIDDEN_SETTING_FRAGMENTS):
            raise ForbiddenSettingError(f"No se permiten secretos en SQLite: '{key}'")
        if lowered in _FORBIDDEN_SETTING_KEYS:
            raise ForbiddenSettingError(
                "El entorno de cuenta no se persiste: siempre se arranca en Demo.")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO settings(key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (key, json.dumps(value), time.time()))

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM settings WHERE key = ?",
                                     (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    # ------------------------------------------------------- acceso genérico
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
