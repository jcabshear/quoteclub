"""Game rules: rooms, turn rotation, hints, guessing, disputes, scoring.

Pure functions over plain dicts, so every rule is testable without a
database, a network, or an audio file.

Shape of a game
---------------
A **room** is a code and a list of seats. Anyone holding the code can take
a seat; there are no accounts. Seats rotate: whoever sends a clip is the
one player who cannot guess it, and the next seat sends next.

A **round** is one clip plus a per-player record. Every other player in the
room guesses independently and banks points independently - one person
solving it does not end the round for anyone else. The sender banks
nothing; making a hard clip is not how you score.

Privacy rule that drives the view functions: a round rendered for someone
who has not solved it must contain audio and points and nothing else. No
answer, no transcript, no search phrase, no source URL, no filename, no
title. `round_view()` is the only place an answer may appear, so there is
exactly one function to audit.
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterable, Optional

from .scoring import normalise_title, reward_after_hints

MIN_PLAYERS = 2
MAX_PLAYERS = 12

OPEN = "open"
CLOSED = "closed"

# Per-player state within a round.
WAITING = "guessing"
SOLVED = "solved"
GAVE_UP = "revealed"


class RuleError(Exception):
    """A move the rules do not allow."""


# --------------------------------------------------------------------------
# Rooms and seats
# --------------------------------------------------------------------------

def seat(room: dict, player_id: str, name: str = "") -> dict:
    """Take a seat in the room. Idempotent for a player already seated."""
    seats: list[str] = list(room.get("seats", []))
    if player_id in seats:
        if name:
            room.setdefault("names", {})[player_id] = name
        return room
    if len(seats) >= MAX_PLAYERS:
        raise RuleError(f"this room is full ({MAX_PLAYERS} players)")
    if room.get("locked"):
        raise RuleError("this room is closed to new players")
    seats.append(player_id)
    room["seats"] = seats
    room.setdefault("names", {})[player_id] = name or "player"
    if not room.get("turn"):
        room["turn"] = player_id
    return room


def leave(room: dict, player_id: str) -> dict:
    """Give up a seat. The turn moves on if it was theirs."""
    seats = [p for p in room.get("seats", []) if p != player_id]
    was_turn = room.get("turn") == player_id
    room["seats"] = seats
    if was_turn:
        room["turn"] = seats[0] if seats else None
    return room


def is_member(room: dict, player_id: str) -> bool:
    return player_id in (room.get("seats") or [])


def others(room: dict, player_id: str) -> list[str]:
    return [p for p in room.get("seats") or [] if p != player_id]


def next_seat(room: dict, after: str) -> Optional[str]:
    seats = room.get("seats") or []
    if not seats:
        return None
    if after not in seats:
        return seats[0]
    return seats[(seats.index(after) + 1) % len(seats)]


def require_turn(room: dict, player_id: str) -> None:
    if not is_member(room, player_id):
        raise RuleError("you are not in this room")
    if len(room.get("seats") or []) < MIN_PLAYERS:
        raise RuleError("wait for at least one more player to join")
    if room.get("turn") != player_id:
        who = room.get("names", {}).get(room.get("turn"), "someone else")
        raise RuleError(f"it is {who}'s turn to send")


def open_round(rounds: Iterable[dict]) -> Optional[dict]:
    for r in rounds:
        if r.get("state") == OPEN:
            return r
    return None


# --------------------------------------------------------------------------
# Creating a round
# --------------------------------------------------------------------------

def create_round(
    room: dict,
    sender: str,
    clip_id: str,
    answer: str,
    points: int,
    *,
    rounds: Iterable[dict] = (),
    now: Optional[float] = None,
) -> dict:
    require_turn(room, sender)
    if open_round(rounds) is not None:
        raise RuleError("there is already a clip in play")
    answer = (answer or "").strip()
    if not answer:
        raise RuleError("fill in the answer - only you can see it")
    if not clip_id:
        raise RuleError("a round needs audio")
    guessers = others(room, sender)
    if not guessers:
        raise RuleError("nobody else is in the room yet")
    return {
        "id": None,  # assigned by the store
        "room_id": room.get("id"),
        "sender": sender,
        "clip_id": clip_id,
        "answer": answer,
        "base_points": int(points),
        "state": OPEN,
        "hints": [],  # [{clip_id, requested_by, requested_at, delivered_at}]
        "players": {
            p: {"status": WAITING, "guesses": [], "awarded": 0,
                "disputed": False, "resolved_at": None}
            for p in guessers
        },
        "created_at": now or time.time(),
        "closed_at": None,
    }


def _entry(rnd: dict, player_id: str) -> dict:
    entry = (rnd.get("players") or {}).get(player_id)
    if entry is None:
        if rnd.get("sender") == player_id:
            raise RuleError("you sent this clip")
        raise RuleError("you are not part of this round")
    return entry


def hints_delivered(rnd: dict) -> int:
    return sum(1 for h in rnd.get("hints", []) if h.get("delivered_at"))


def reward_now(rnd: dict) -> int:
    """What a player still guessing would bank by solving it right now."""
    return reward_after_hints(int(rnd.get("base_points", 0)), hints_delivered(rnd))


# --------------------------------------------------------------------------
# Hints
# --------------------------------------------------------------------------

def request_hint(rnd: dict, player_id: str, now: Optional[float] = None) -> dict:
    entry = _entry(rnd, player_id)
    if rnd.get("state") != OPEN:
        raise RuleError("this round is finished")
    if entry["status"] != WAITING:
        raise RuleError("you are done with this one")
    if any(not h.get("delivered_at") for h in rnd.get("hints", [])):
        raise RuleError("a hint has already been asked for")
    rnd.setdefault("hints", []).append(
        {
            "clip_id": None,
            "requested_by": player_id,
            "requested_at": now or time.time(),
            "delivered_at": None,
        }
    )
    return rnd


def deliver_hint(
    rnd: dict, player_id: str, clip_id: str, now: Optional[float] = None
) -> dict:
    """The sender supplies more audio. THIS is what costs points.

    It costs them for everyone still guessing. Anyone who already solved
    the round keeps what they banked.
    """
    if rnd.get("sender") != player_id:
        raise RuleError("only the sender can supply a hint")
    if rnd.get("state") != OPEN:
        raise RuleError("this round is finished")
    if not clip_id:
        raise RuleError("a hint is another audio excerpt, not text")
    pending = [h for h in rnd.get("hints", []) if not h.get("delivered_at")]
    if not pending:
        raise RuleError("nobody has asked for a hint")
    pending[0]["clip_id"] = clip_id
    pending[0]["delivered_at"] = now or time.time()
    return rnd


def pending_hint(rnd: dict) -> Optional[dict]:
    for h in rnd.get("hints", []):
        if not h.get("delivered_at"):
            return h
    return None


# --------------------------------------------------------------------------
# Guessing
# --------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(the|a|an)\b")
_PUNCT = re.compile(r"[^a-z0-9]+")
_YEAR = re.compile(r"\(?\b(19|20)\d{2}\b\)?")
_EPISODE = re.compile(r"\bs\d{1,2}\s*e\d{1,2}\b")


def canonical(text: str) -> str:
    t = (text or "").lower()
    t = _YEAR.sub(" ", t)
    t = _EPISODE.sub(" ", t)
    t = _ARTICLES.sub(" ", t)
    t = _PUNCT.sub(" ", t)
    return " ".join(t.split())


def titles_match(guess: str, answer: str) -> bool:
    g, a = canonical(guess), canonical(answer)
    if not g or not a:
        return False
    if g == a:
        return True
    if g in a or a in g:
        return min(len(g), len(a)) >= 4
    gt, at = set(g.split()), set(a.split())
    if not gt or not at:
        return False
    return len(gt & at) / len(gt | at) >= 0.8


def submit_guess(
    rnd: dict, player_id: str, text: str, now: Optional[float] = None
) -> dict:
    entry = _entry(rnd, player_id)
    if rnd.get("state") != OPEN:
        raise RuleError("this round is finished")
    if entry["status"] != WAITING:
        raise RuleError("you already finished this one")
    text = (text or "").strip()
    if not text:
        raise RuleError("empty guess")
    ok = titles_match(text, rnd.get("answer", ""))
    entry["guesses"].append({"text": text, "correct": ok, "at": now or time.time()})
    if ok:
        entry["status"] = SOLVED
        entry["awarded"] = reward_now(rnd)
        entry["resolved_at"] = now or time.time()
        entry["disputed"] = False
    return rnd


def reveal(rnd: dict, player_id: str, now: Optional[float] = None) -> dict:
    entry = _entry(rnd, player_id)
    if rnd.get("state") != OPEN:
        raise RuleError("this round is finished")
    if entry["status"] != WAITING:
        raise RuleError("you already finished this one")
    entry["status"] = GAVE_UP
    entry["awarded"] = 0
    entry["resolved_at"] = now or time.time()
    entry["disputed"] = False
    return rnd


def dispute(rnd: dict, player_id: str) -> dict:
    entry = _entry(rnd, player_id)
    if rnd.get("state") != OPEN:
        raise RuleError("this round is finished")
    if entry["status"] != WAITING:
        raise RuleError("you already finished this one")
    if not entry["guesses"]:
        raise RuleError("make a guess first")
    entry["disputed"] = True
    return rnd


def resolve_dispute(
    rnd: dict, sender_id: str, player_id: str, accept: bool,
    now: Optional[float] = None,
) -> dict:
    if rnd.get("sender") != sender_id:
        raise RuleError("only the sender reviews a disputed guess")
    entry = _entry(rnd, player_id)
    if not entry.get("disputed"):
        raise RuleError("that guess is not disputed")
    entry["disputed"] = False
    if accept:
        entry["status"] = SOLVED
        entry["awarded"] = reward_now(rnd)
        entry["resolved_at"] = now or time.time()
        if entry["guesses"]:
            entry["guesses"][-1]["correct"] = True
    return rnd


# --------------------------------------------------------------------------
# Closing a round and moving the turn
# --------------------------------------------------------------------------

def everyone_done(rnd: dict) -> bool:
    return all(
        e["status"] != WAITING for e in (rnd.get("players") or {}).values()
    )


def close_round(
    rnd: dict, by: Optional[str] = None, now: Optional[float] = None
) -> dict:
    """Close automatically when everyone is done, or let the sender close.

    Anyone still guessing when the sender closes it simply scores nothing;
    the answer becomes visible to them.
    """
    if rnd.get("state") != OPEN:
        return rnd
    if not everyone_done(rnd):
        if by is None:
            return rnd
        if by != rnd.get("sender"):
            raise RuleError("only the sender can end the round early")
    rnd["state"] = CLOSED
    rnd["closed_at"] = now or time.time()
    return rnd


def advance_turn(room: dict, rnd: dict) -> dict:
    if rnd.get("state") != CLOSED:
        return room
    room["turn"] = next_seat(room, rnd.get("sender", ""))
    return room


# --------------------------------------------------------------------------
# Views - the only place answers are allowed out
# --------------------------------------------------------------------------

def may_see_answer(rnd: dict, viewer: str) -> bool:
    if viewer == rnd.get("sender"):
        return True
    if rnd.get("state") == CLOSED:
        return viewer in (rnd.get("players") or {})
    entry = (rnd.get("players") or {}).get(viewer)
    return bool(entry and entry["status"] in (SOLVED, GAVE_UP))


def round_view(rnd: dict, viewer: str, names: dict | None = None) -> dict[str, Any]:
    names = names or {}
    is_sender = viewer == rnd.get("sender")
    entry = (rnd.get("players") or {}).get(viewer)
    pending = pending_hint(rnd)

    view: dict[str, Any] = {
        "id": rnd.get("id"),
        "state": rnd.get("state"),
        "role": "sender" if is_sender else ("guesser" if entry else "spectator"),
        "sender_name": names.get(rnd.get("sender"), "someone"),
        "clip_id": rnd.get("clip_id"),
        "hint_clip_ids": [
            h["clip_id"] for h in rnd.get("hints", []) if h.get("delivered_at")
        ],
        "hints_delivered": hints_delivered(rnd),
        "hint_pending": bool(pending),
        "hint_pending_from": names.get(pending["requested_by"]) if pending else None,
        "base_points": rnd.get("base_points"),
        "reward_now": reward_now(rnd),
        "created_at": rnd.get("created_at"),
        "closed_at": rnd.get("closed_at"),
        # Per-player progress: status and score only. Guess text is private
        # to its author and the sender - another player's wrong guess is a
        # hint you did not earn.
        "table": [
            {
                "name": names.get(pid, "player"),
                "status": e["status"],
                "awarded": e["awarded"],
                "you": pid == viewer,
                "disputed": bool(e.get("disputed")),
            }
            for pid, e in (rnd.get("players") or {}).items()
        ],
    }

    if entry is not None:
        view["your_status"] = entry["status"]
        view["your_awarded"] = entry["awarded"]
        view["your_guesses"] = list(entry["guesses"])
        view["you_disputed"] = bool(entry.get("disputed"))
    if is_sender:
        view["disputes"] = [
            {"player_id": pid, "name": names.get(pid, "player"),
             "guess": e["guesses"][-1]["text"] if e["guesses"] else ""}
            for pid, e in (rnd.get("players") or {}).items()
            if e.get("disputed")
        ]
    if may_see_answer(rnd, viewer):
        view["answer"] = rnd.get("answer")
    # Never present for anyone: search phrase, transcript, source url,
    # source page title, original filename.
    return view


def scoreboard(seats: list[str], rounds: Iterable[dict]) -> dict[str, int]:
    """Only guessers score. Creating a brutal clip earns nothing."""
    totals = {p: 0 for p in seats}
    for r in rounds:
        for pid, e in (r.get("players") or {}).items():
            if pid in totals:
                totals[pid] += int(e.get("awarded", 0))
    return totals


def play_history(rounds: Iterable[dict]) -> dict[str, int]:
    """How often each source has come up - feeds familiarity in scoring."""
    hist: dict[str, int] = {}
    for r in rounds:
        key = normalise_title(r.get("answer") or "")
        if key:
            hist[key] = hist.get(key, 0) + 1
    return hist
