"""SQLite persistence. One file, no external services, no accounts.

Identity is a signed cookie holding a random player id. That is the whole
of "who you are": no email, no password, no OAuth. A **room code** is the
only thing that grants access to a room, so any number of unrelated groups
can play at once, each in their own room, by sharing a code.

Housekeeping matters for a public deployment: idle rooms and their audio
are purged on a schedule so a small disk does not fill with abandoned
games.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional

DATA_DIR = Path(os.environ.get("QC_DATA_DIR", "./data")).resolve()
DB_PATH = DATA_DIR / "quoteclub.sqlite3"
CLIP_DIR = DATA_DIR / "clips"

# Rooms nobody has touched for this long are deleted, with their clips.
ROOM_TTL_DAYS = float(os.environ.get("QC_ROOM_TTL_DAYS", 45))

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL DEFAULT 'player',
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rooms (
  id TEXT PRIMARY KEY,
  code TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL DEFAULT 'Quote Club',
  seats TEXT NOT NULL DEFAULT '[]',
  names TEXT NOT NULL DEFAULT '{}',
  turn TEXT,
  model TEXT,
  locked INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  last_active REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
  id TEXT PRIMARY KEY,
  room_id TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS clips (
  id TEXT PRIMARY KEY,
  room_id TEXT NOT NULL,
  owner TEXT NOT NULL,
  seconds REAL NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ratings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  room_id TEXT NOT NULL,
  features TEXT NOT NULL,
  suggested INTEGER NOT NULL,
  chosen INTEGER NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS rounds_by_room ON rounds(room_id, created_at);
CREATE INDEX IF NOT EXISTS clips_by_room ON clips(room_id);
CREATE INDEX IF NOT EXISTS rooms_by_active ON rooms(last_active);
"""

# Letters only, minus I and O. Dropping digits entirely removes every
# 0/O and 1/I/l confusion at once, which matters when a code is read out
# loud or typed from a screenshot. 24^6 = 191 million codes.
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    return conn


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def _code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(6))


class Store:
    def __init__(self, conn: Optional[sqlite3.Connection] = None):
        self.conn = conn or connect()

    # -- players -----------------------------------------------------
    def create_player(self, name: str = "player") -> dict:
        pid = new_id("p")
        self.conn.execute(
            "INSERT INTO players (id, name, created_at) VALUES (?,?,?)",
            (pid, name[:40] or "player", time.time()),
        )
        self.conn.commit()
        return {"id": pid, "name": name}

    def get_player(self, player_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM players WHERE id=?", (player_id,)
        ).fetchone()
        return dict(row) if row else None

    def rename_player(self, player_id: str, name: str) -> None:
        self.conn.execute(
            "UPDATE players SET name=? WHERE id=?", (name[:40] or "player", player_id)
        )
        self.conn.commit()

    # -- rooms -------------------------------------------------------
    def create_room(self, name: str = "Quote Club") -> dict:
        rid = new_id("r")
        now = time.time()
        for _ in range(40):
            code = _code()
            try:
                self.conn.execute(
                    "INSERT INTO rooms (id, code, name, seats, names, turn, model,"
                    " locked, created_at, last_active) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (rid, code, name[:60] or "Quote Club", "[]", "{}", None, None,
                     0, now, now),
                )
                self.conn.commit()
                return self.get_room(rid)  # type: ignore[return-value]
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError("could not allocate a free room code")

    def get_room(self, room_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM rooms WHERE id=?", (room_id,)).fetchone()
        return _room(row)

    def room_by_code(self, code: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM rooms WHERE code=?", (normalise_code(code),)
        ).fetchone()
        return _room(row)

    def save_room(self, room: dict, touch: bool = True) -> None:
        self.conn.execute(
            "UPDATE rooms SET seats=?, names=?, turn=?, model=?, locked=?,"
            " last_active=? WHERE id=?",
            (
                json.dumps(room.get("seats", [])),
                json.dumps(room.get("names", {})),
                room.get("turn"),
                room.get("model"),
                1 if room.get("locked") else 0,
                time.time() if touch else room.get("last_active", time.time()),
                room["id"],
            ),
        )
        self.conn.commit()

    def rooms_for_player(self, player_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM rooms WHERE seats LIKE ? ORDER BY last_active DESC LIMIT 20",
            (f'%"{player_id}"%',),
        ).fetchall()
        out = []
        for row in rows:
            room = _room(row)
            if room and player_id in room["seats"]:
                out.append(room)
        return out

    # -- rounds ------------------------------------------------------
    def add_round(self, rnd: dict) -> dict:
        rid = new_id("rd")
        rnd["id"] = rid
        self.conn.execute(
            "INSERT INTO rounds (id, room_id, payload, created_at) VALUES (?,?,?,?)",
            (rid, rnd["room_id"], json.dumps(rnd), rnd.get("created_at", time.time())),
        )
        self.conn.commit()
        return rnd

    def save_round(self, rnd: dict) -> None:
        self.conn.execute(
            "UPDATE rounds SET payload=? WHERE id=?", (json.dumps(rnd), rnd["id"])
        )
        self.conn.commit()

    def get_round(self, round_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT payload FROM rounds WHERE id=?", (round_id,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def rounds(self, room_id: str, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT payload FROM rounds WHERE room_id=? ORDER BY created_at DESC"
            " LIMIT ?",
            (room_id, limit),
        ).fetchall()
        return [json.loads(r["payload"]) for r in reversed(rows)]

    # -- clips -------------------------------------------------------
    def put_clip(self, room_id: str, owner: str, wav: bytes, seconds: float) -> str:
        cid = new_id("c")
        (CLIP_DIR / f"{cid}.wav").write_bytes(wav)
        self.conn.execute(
            "INSERT INTO clips (id, room_id, owner, seconds, created_at)"
            " VALUES (?,?,?,?,?)",
            (cid, room_id, owner, seconds, time.time()),
        )
        self.conn.commit()
        return cid

    def clip_meta(self, clip_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
        return dict(row) if row else None

    def clip_path(self, clip_id: str) -> Path:
        return CLIP_DIR / f"{clip_id}.wav"

    # -- ratings -----------------------------------------------------
    def record_rating(
        self, room_id: str, features: dict, suggested: int, chosen: int
    ) -> None:
        self.conn.execute(
            "INSERT INTO ratings (room_id, features, suggested, chosen, created_at)"
            " VALUES (?,?,?,?,?)",
            (room_id, json.dumps(features), suggested, chosen, time.time()),
        )
        self.conn.commit()

    # -- housekeeping ------------------------------------------------
    def purge_idle_rooms(self, ttl_days: float = ROOM_TTL_DAYS) -> int:
        """Delete rooms nobody has touched in a while, and their audio."""
        cutoff = time.time() - ttl_days * 86400
        rows = self.conn.execute(
            "SELECT id FROM rooms WHERE last_active < ?", (cutoff,)
        ).fetchall()
        removed = 0
        for row in rows:
            rid = row["id"]
            for clip in self.conn.execute(
                "SELECT id FROM clips WHERE room_id=?", (rid,)
            ).fetchall():
                self.clip_path(clip["id"]).unlink(missing_ok=True)
            self.conn.execute("DELETE FROM clips WHERE room_id=?", (rid,))
            self.conn.execute("DELETE FROM rounds WHERE room_id=?", (rid,))
            self.conn.execute("DELETE FROM ratings WHERE room_id=?", (rid,))
            self.conn.execute("DELETE FROM rooms WHERE id=?", (rid,))
            removed += 1
        if removed:
            self.conn.commit()
        return removed

    def stats(self) -> dict:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "rooms": q("SELECT COUNT(*) FROM rooms"),
            "rounds": q("SELECT COUNT(*) FROM rounds"),
            "clips": q("SELECT COUNT(*) FROM clips"),
        }


def normalise_code(code: str) -> str:
    """Accept sloppy typing: lowercase, spaces, dashes, stray punctuation.

    Codes contain no digits, so digits are dropped rather than guessed at.
    """
    return "".join(ch for ch in (code or "").upper() if "A" <= ch <= "Z")


def pretty_code(code: str) -> str:
    """ABCDEF -> ABC-DEF, for display only."""
    return f"{code[:3]}-{code[3:]}" if len(code) == 6 else code


def _room(row: Optional[sqlite3.Row]) -> Optional[dict]:
    if row is None:
        return None
    return {
        "id": row["id"],
        "code": row["code"],
        "name": row["name"],
        "seats": json.loads(row["seats"]),
        "names": json.loads(row["names"]),
        "turn": row["turn"],
        "model": row["model"],
        "locked": bool(row["locked"]),
        "created_at": row["created_at"],
        "last_active": row["last_active"],
    }
