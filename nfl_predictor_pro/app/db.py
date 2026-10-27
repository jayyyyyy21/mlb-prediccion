"""Persistencia en SQLite: tablas de datos, registro de actualizaciones y pronósticos."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_updates (
    source TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,
    rows INTEGER,
    status TEXT,
    message TEXT
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prediction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    season INTEGER,
    week INTEGER,
    kickoff_utc TEXT,
    created_at TEXT NOT NULL,
    home_team TEXT,
    away_team TEXT,
    pred_home REAL,
    pred_away REAL,
    p_home REAL,
    p_away REAL,
    p_tie REAL,
    status TEXT,
    confidence TEXT,
    inputs_hash TEXT,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS ix_predlog_game ON prediction_log(game_id);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    con = connect()
    try:
        yield con
        con.commit()
    finally:
        con.close()


def table_exists(name: str) -> bool:
    with session() as con:
        row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


def write_df(df: pd.DataFrame, name: str, replace_where: tuple[str, list[Any]] | None = None) -> None:
    """Guarda un DataFrame. Con replace_where=(col, valores) sólo sustituye esas filas."""
    with session() as con:
        if replace_where and table_exists(name):
            col, values = replace_where
            con.executemany(f"DELETE FROM {name} WHERE {col} = ?", [(v,) for v in values])
            df.to_sql(name, con, if_exists="append", index=False)
        else:
            df.to_sql(name, con, if_exists="replace", index=False)


def read_df(sql: str, params: tuple | list = ()) -> pd.DataFrame:
    with session() as con:
        return pd.read_sql_query(sql, con, params=params)


def read_table(name: str) -> pd.DataFrame:
    if not table_exists(name):
        return pd.DataFrame()
    return read_df(f"SELECT * FROM {name}")


def mark_source(source: str, rows: int, status: str = "ok", message: str = "") -> None:
    with session() as con:
        con.execute(
            "INSERT OR REPLACE INTO source_updates(source, updated_at, rows, status, message) VALUES (?,?,?,?,?)",
            (source, utcnow(), rows, status, message),
        )


def source_updates() -> list[dict]:
    with session() as con:
        return [dict(r) for r in con.execute("SELECT * FROM source_updates ORDER BY source")]


def kv_set(key: str, value: Any) -> None:
    with session() as con:
        con.execute(
            "INSERT OR REPLACE INTO kv(key, value, updated_at) VALUES (?,?,?)",
            (key, json.dumps(value, default=float), utcnow()),
        )


def kv_get(key: str, default: Any = None) -> Any:
    with session() as con:
        row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default
