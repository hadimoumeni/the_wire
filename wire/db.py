"""SQLite persistence for generated theses (p4p-style hand-rolled db layer)."""
from __future__ import annotations

import os
import sqlite3

from .schemas import Thesis

_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "data", "wire.db")
DB_PATH = os.environ.get("WIRE_DB", _DEFAULT)


def get_conn(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str | None = None) -> None:
    with get_conn(path) as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS theses (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker         TEXT,
                quarter        TEXT,
                created_at     TEXT,
                stance         TEXT,
                conviction     INTEGER,
                grounding_rate REAL,
                mode           TEXT,
                model_name     TEXT,
                json           TEXT
            )
            """
        )


def save_thesis(t: Thesis, path: str | None = None) -> int:
    init_db(path)
    with get_conn(path) as c:
        cur = c.execute(
            """INSERT INTO theses
               (ticker, quarter, created_at, stance, conviction, grounding_rate,
                mode, model_name, json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (t.ticker, t.quarter, t.created_at, t.stance, t.conviction,
             t.verification.grounding_rate, t.mode, t.model_name, t.model_dump_json()),
        )
        return int(cur.lastrowid)


def list_theses(path: str | None = None) -> list[dict]:
    init_db(path)
    with get_conn(path) as c:
        rows = c.execute(
            """SELECT id, ticker, quarter, created_at, stance, conviction,
                      grounding_rate, mode, model_name
               FROM theses ORDER BY id DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_thesis(thesis_id: int, path: str | None = None) -> Thesis | None:
    init_db(path)
    with get_conn(path) as c:
        row = c.execute("SELECT json FROM theses WHERE id = ?", (thesis_id,)).fetchone()
    if not row:
        return None
    return Thesis.model_validate_json(row["json"])
