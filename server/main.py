"""Quote Club - one service: API, client, audio work, storage.

Access model: **a room code and nothing else.** There are no accounts, no
email, no OAuth. Your browser holds a signed random player id so the app
knows which seat is yours; a room code is what lets you take a seat. Any
number of unrelated groups can play at once, each in its own room, and a
person can be in several rooms at the same time.

Search is a *streaming job*, not a blocking request: every candidate
reports its outcome as it happens - admitted, rejected and why, fetched,
verified, or failed and why - so "no playable audio" is never the whole
story.

Built on Starlette rather than FastAPI: smaller dependency surface, and
every route here validates its own input anyway.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import secrets
import shutil
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import audioutil as au
from . import extract as X
from . import game as G
from . import scoring as S
from .discovery import Candidate, FailureMemory, build_extractor_gate, plan
from .providers import websearch, yarn
from .store import Store, new_id, normalise_code, pretty_code

APP_SECRET = os.environ.get("QC_SECRET") or secrets.token_hex(32)
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
PREVIEW_TTL = 60 * 45
MAX_ATTEMPTS = int(os.environ.get("QC_MAX_ATTEMPTS", 8))
TARGET_RESULTS = int(os.environ.get("QC_TARGET_RESULTS", 4))
SECURE_COOKIES = os.environ.get("QC_INSECURE_COOKIES") != "1"
MAX_ROOMS_PER_PLAYER = int(os.environ.get("QC_MAX_ROOMS_PER_PLAYER", 20))
# One search at a time per player: this is the expensive endpoint and the
# only one worth rate-limiting on a small host.
ACTIVE_SEARCHES: dict[str, str] = {}

store = Store()
failure_memory = FailureMemory()
_gate: Optional[Callable[[str], Optional[str]]] = None


def gate() -> Callable[[str], Optional[str]]:
    global _gate
    if _gate is None:
        try:
            _gate = build_extractor_gate()
        except Exception:
            # Without yt-dlp nothing is extractable. /api/health says so
            # rather than the app quietly reporting an empty internet.
            _gate = lambda url: None  # noqa: E731
    return _gate


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# --------------------------------------------------------------------------
# Identity: a signed random id in a cookie. That is all.
# --------------------------------------------------------------------------

def sign(value: str) -> str:
    mac = hmac.new(APP_SECRET.encode(), value.encode(), sha256).hexdigest()[:32]
    return f"{value}.{mac}"


def unsign(token: Optional[str]) -> Optional[str]:
    if not token or "." not in token:
        return None
    value, mac = token.rsplit(".", 1)
    expected = hmac.new(APP_SECRET.encode(), value.encode(), sha256).hexdigest()[:32]
    return value if hmac.compare_digest(mac, expected) else None


def set_player_cookie(resp, player_id: str):
    resp.set_cookie(
        "qc_player",
        sign(player_id),
        httponly=True,
        samesite="lax",
        secure=SECURE_COOKIES,
        max_age=60 * 60 * 24 * 365 * 2,
        path="/",
    )
    return resp


def current_player(request: Request) -> Optional[dict]:
    pid = unsign(request.cookies.get("qc_player"))
    return store.get_player(pid) if pid else None


def require_player(request: Request) -> dict:
    p = current_player(request)
    if not p:
        raise ApiError(401, "no player session - pick a name to start")
    return p


def require_room(request: Request, player: dict) -> dict:
    code = normalise_code(request.path_params.get("code", ""))
    room = store.room_by_code(code)
    if not room:
        raise ApiError(404, "no room with that code")
    if not G.is_member(room, player["id"]):
        raise ApiError(403, "join this room first")
    return room


async def body_json(request: Request) -> dict:
    try:
        data = await request.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------
# Session and rooms
# --------------------------------------------------------------------------

async def session(request: Request):
    """Create or rename the browser's player. No credentials involved."""
    body = await body_json(request)
    name = (body.get("name") or "").strip()[:40]
    player = current_player(request)
    if player is None:
        player = store.create_player(name or "player")
        return set_player_cookie(
            JSONResponse({"player": {"id": player["id"], "name": name or "player"}}),
            player["id"],
        )
    if name and name != player["name"]:
        store.rename_player(player["id"], name)
        player["name"] = name
        # Keep the display name current in every room they sit in.
        for room in store.rooms_for_player(player["id"]):
            room["names"][player["id"]] = name
            store.save_room(room, touch=False)
    return JSONResponse({"player": {"id": player["id"], "name": player["name"]}})


async def me(request: Request):
    player = current_player(request)
    if not player:
        return JSONResponse({"player": None, "rooms": []})
    rooms = []
    for room in store.rooms_for_player(player["id"]):
        rounds = store.rounds(room["id"], limit=1)
        rooms.append(
            {
                "code": room["code"],
                "pretty_code": pretty_code(room["code"]),
                "name": room["name"],
                "players": len(room["seats"]),
                "your_turn": room["turn"] == player["id"],
                "last_active": room["last_active"],
                "has_open_round": bool(rounds and rounds[-1].get("state") == G.OPEN),
            }
        )
    return JSONResponse(
        {"player": {"id": player["id"], "name": player["name"]}, "rooms": rooms}
    )


async def create_room(request: Request):
    body = await body_json(request)
    name = (body.get("name") or "Quote Club").strip()[:60]
    display = (body.get("player_name") or "").strip()[:40]

    player = current_player(request)
    if player is None:
        player = store.create_player(display or "player")
    elif display and display != player["name"]:
        store.rename_player(player["id"], display)
        player["name"] = display

    if len(store.rooms_for_player(player["id"])) >= MAX_ROOMS_PER_PLAYER:
        raise ApiError(429, "you are in too many rooms already")

    room = store.create_room(name)
    G.seat(room, player["id"], player["name"])
    store.save_room(room)
    resp = JSONResponse(
        {
            "code": room["code"],
            "pretty_code": pretty_code(room["code"]),
            "name": room["name"],
        }
    )
    return set_player_cookie(resp, player["id"])


async def join_room(request: Request):
    body = await body_json(request)
    display = (body.get("name") or "").strip()[:40]
    code = normalise_code(request.path_params.get("code", ""))
    room = store.room_by_code(code)
    if not room:
        raise ApiError(404, "no room with that code")

    player = current_player(request)
    if player is None:
        player = store.create_player(display or "player")
    elif display and display != player["name"]:
        store.rename_player(player["id"], display)
        player["name"] = display

    try:
        G.seat(room, player["id"], player["name"])
    except G.RuleError as e:
        raise ApiError(409, str(e))
    store.save_room(room)
    return set_player_cookie(
        JSONResponse(
            {
                "code": room["code"],
                "pretty_code": pretty_code(room["code"]),
                "name": room["name"],
                "you": player["id"],
            }
        ),
        player["id"],
    )


async def leave_room(request: Request):
    player = require_player(request)
    room = require_room(request, player)
    G.leave(room, player["id"])
    store.save_room(room)
    return JSONResponse({"ok": True})


async def room_state(request: Request):
    player = require_player(request)
    room = require_room(request, player)
    rounds = store.rounds(room["id"])
    me_id = player["id"]
    return JSONResponse(
        {
            "you": {"id": me_id, "name": player["name"]},
            "room": {
                "code": room["code"],
                "pretty_code": pretty_code(room["code"]),
                "name": room["name"],
                "players": [
                    {
                        "id": p,
                        "name": room["names"].get(p, "player"),
                        "you": p == me_id,
                        "on_turn": p == room["turn"],
                    }
                    for p in room["seats"]
                ],
                "turn": room["turn"],
                "turn_name": room["names"].get(room["turn"], ""),
                "your_turn": room["turn"] == me_id
                and len(room["seats"]) >= G.MIN_PLAYERS,
                "can_send": room["turn"] == me_id
                and len(room["seats"]) >= G.MIN_PLAYERS
                and G.open_round(rounds) is None,
                "max_players": G.MAX_PLAYERS,
            },
            "rounds": [G.round_view(r, me_id, room["names"]) for r in rounds[-25:]],
            "scoreboard": [
                {"id": p, "name": room["names"].get(p, "player"), "points": pts}
                for p, pts in sorted(
                    G.scoreboard(room["seats"], rounds).items(),
                    key=lambda kv: -kv[1],
                )
            ],
            "tools": X.tools_available(),
        }
    )


# --------------------------------------------------------------------------
# Search - streaming job
# --------------------------------------------------------------------------

class SearchJob:
    def __init__(self, phrase: str, room_id: str, player_id: str):
        self.id = new_id("s")
        self.phrase = phrase
        self.room_id = room_id
        self.player_id = player_id
        self.events: list[dict] = []
        self.done = False
        self.queue: asyncio.Queue = asyncio.Queue()
        self.workdir = Path(tempfile.mkdtemp(prefix="qc-job-"))
        self.created = time.time()

    def emit(self, kind: str, **data: Any) -> None:
        ev = {"kind": kind, "at": time.time(), **data}
        self.events.append(ev)
        self.queue.put_nowait(ev)


JOBS: dict[str, SearchJob] = {}
PREVIEWS: dict[str, dict] = {}

# A background task with no strong reference can be garbage collected
# mid-flight, which looks exactly like a search that silently stopped.
RUNNING: set[asyncio.Task] = set()


def _spawn(coro) -> asyncio.Task:
    task = asyncio.get_running_loop().create_task(coro)
    RUNNING.add(task)
    task.add_done_callback(RUNNING.discard)
    return task


async def start_search(request: Request):
    player = require_player(request)
    room = require_room(request, player)
    body = await body_json(request)
    phrase = (body.get("phrase") or "").strip()[:300]
    pasted = [u for u in (body.get("urls") or []) if isinstance(u, str)][:3]
    if not phrase and not pasted:
        raise ApiError(400, "type a quote, or paste a link")

    previous = ACTIVE_SEARCHES.get(player["id"])
    if previous and previous in JOBS and not JOBS[previous].done:
        raise ApiError(429, "you already have a search running")

    job = SearchJob(phrase, room["id"], player["id"])
    JOBS[job.id] = job
    ACTIVE_SEARCHES[player["id"]] = job.id
    _spawn(run_search(job, pasted))
    return JSONResponse({"job_id": job.id})


async def stream_search(request: Request):
    player = require_player(request)
    job = JOBS.get(request.path_params["job_id"])
    if not job or job.player_id != player["id"]:
        raise ApiError(404, "no such search")

    async def gen():
        sent = 0
        while True:
            while sent < len(job.events):
                yield f"data: {json.dumps(job.events[sent])}\n\n".encode()
                sent += 1
            if job.done:
                break
            try:
                await asyncio.wait_for(job.queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield b": keepalive\n\n"
        yield b'data: {"kind": "done"}\n\n'

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def run_search(job: SearchJob, pasted: list[str]) -> None:
    try:
        await _run_search(job, pasted)
    except Exception as exc:  # never leave the client hanging
        job.emit("error", message=str(exc)[:300])
    finally:
        job.done = True
        job.queue.put_nowait({"kind": "done"})
        ACTIVE_SEARCHES.pop(job.player_id, None)


async def _run_search(job: SearchJob, pasted: list[str]) -> None:
    import httpx

    missing = [k for k, v in X.tools_available().items() if not v]
    if missing:
        job.emit("warning", message=f"server is missing: {', '.join(missing)}")

    job.emit("phase", phase="Looking for the line…")
    candidates: list[Candidate] = []
    if pasted:
        candidates += websearch.from_pasted_urls(pasted)

    if job.phrase:
        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; QuoteClub/1.0)"},
        ) as http:
            for label, fn in (
                ("subtitle index", yarn.search),
                ("web search", websearch.search),
            ):
                try:
                    found = await fn(http, job.phrase)
                    job.emit("provider", provider=label, found=len(found))
                    candidates += found
                except Exception as exc:
                    job.emit("provider", provider=label, error=str(exc)[:160])

    ordered, rejected = plan(
        job.phrase, candidates, gate(), failure_memory, limit=MAX_ATTEMPTS
    )
    job.emit(
        "plan",
        considered=len(candidates),
        attempting=len(ordered),
        rejected=[{"reason": r} for _u, r in rejected[:25]],
    )

    if not ordered:
        job.emit(
            "empty",
            message=(
                f"Nothing playable. {len(rejected)} result(s) mention the line "
                "but cannot supply audio."
                if rejected
                else "No provider returned any candidates at all."
            ),
        )
        return

    got = 0
    for cand in ordered:
        if got >= TARGET_RESULTS:
            break
        job.emit("attempt", provider=cand.provider, score=round(cand.score, 2))
        try:
            res = await X.extract(cand.url, job.phrase, job.workdir / new_id("w"))
        except X.ExtractionError as exc:
            failure_memory.record(cand.url, exc.failure_class)
            job.emit("failed", reason=str(exc)[:200], failure_class=exc.failure_class)
            continue
        except Exception as exc:
            failure_memory.record(cand.url, X.classify(str(exc)))
            job.emit("failed", reason=str(exc)[:200])
            continue

        failure_memory.record_success(cand.url)
        token = new_id("pv")
        PREVIEWS[token] = {
            "path": res.wav_path,
            "player": job.player_id,
            "room": job.room_id,
            "phrase": job.phrase,
            "expires": time.time() + PREVIEW_TTL,
            "source_title": cand.source_title or res.source_title,
        }
        got += 1
        job.emit(
            "result",
            token=token,
            duration=round(res.duration, 3),
            suggested_answer=cand.source_title or res.source_title or "",
            notes=res.notes,
            provider_label=_provider_label(cand.provider),
            # Deliberately absent: the url, the page title, the filename.
        )

    if got == 0:
        job.emit(
            "empty",
            message=(
                "Everything that looked extractable failed when actually "
                "fetched - the reasons are above. If they all say sign-in or "
                "unavailable, the audio exists behind access this app does "
                "not go around."
            ),
        )


def _provider_label(p: str) -> str:
    return {
        "subtitle_index": "subtitle index",
        "video": "video source",
        "soundboard": "sound clip",
        "web": "web result",
    }.get(p, p)


def _preview_for(request: Request) -> dict:
    player = require_player(request)
    p = PREVIEWS.get(request.path_params["token"])
    if not p or p["player"] != player["id"] or p["expires"] < time.time():
        raise ApiError(404, "preview expired - search again")
    return p


async def preview_wav(request: Request):
    return FileResponse(_preview_for(request)["path"], media_type="audio/wav")


async def preview_peaks(request: Request):
    p = _preview_for(request)
    try:
        buckets = int(request.query_params.get("buckets", 700))
    except ValueError:
        buckets = 700
    samples, rate = au.decode_wav(Path(p["path"]).read_bytes())
    return JSONResponse(
        {
            "peaks": au.peaks(samples, max(64, min(2000, buckets))),
            "duration": len(samples) / rate,
            "rate": rate,
        }
    )


# --------------------------------------------------------------------------
# Cropping
# --------------------------------------------------------------------------

async def make_clip(request: Request):
    player = require_player(request)
    room = require_room(request, player)
    body = await body_json(request)
    p = PREVIEWS.get(body.get("token") or "")
    if not p or p["player"] != player["id"] or p["expires"] < time.time():
        raise ApiError(404, "preview expired - search again")

    samples, rate = au.decode_wav(Path(p["path"]).read_bytes())
    try:
        crop = au.resolve_crop(
            len(samples), rate, float(body.get("start", 0)), float(body.get("end", 0))
        )
    except (au.AudioError, TypeError, ValueError) as e:
        raise ApiError(400, str(e))
    selected = au.apply_crop(samples, crop)
    clip_id = store.put_clip(
        room["id"], player["id"], au.encode_wav(selected, rate), crop.duration
    )

    af = au.audio_features(selected, rate)
    rounds = store.rounds(room["id"])
    model = S.Model.from_json(room.get("model"))
    feats = S.build_features(
        af, p.get("source_title"), G.play_history(rounds), p.get("phrase") or ""
    )
    suggested = model.suggest(feats)
    return JSONResponse(
        {
            "clip_id": clip_id,
            "start": crop.start_seconds,
            "end": crop.end_seconds,
            "duration": crop.duration,
            "frames": crop.frames,
            "suggested_points": suggested,
            "difficulty": S.label_for(suggested),
            "suggested_answer": p.get("source_title") or "",
            "features": feats.__dict__,
            "heard": {
                "loudness": round(af.rms, 4),
                "speech_band": round(af.speech_band_ratio, 3),
                "silence": round(af.silence_ratio, 3),
                "events": af.onset_count,
            },
        }
    )


async def audio(request: Request):
    player = require_player(request)
    room = require_room(request, player)
    clip_id = request.path_params["clip_id"]
    meta = store.clip_meta(clip_id)
    if not meta or meta["room_id"] != room["id"]:
        raise ApiError(404, "no such clip")
    path = store.clip_path(clip_id)
    if not path.exists():
        raise ApiError(404, "clip file missing")
    # Neutral filename: nothing about the source travels in URL or headers.
    return FileResponse(
        path,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="clip.wav"'},
    )


# --------------------------------------------------------------------------
# Rounds
# --------------------------------------------------------------------------

async def send_round(request: Request):
    player = require_player(request)
    room = require_room(request, player)
    body = await body_json(request)
    rounds = store.rounds(room["id"])
    try:
        points = int(body.get("points", 25))
    except (TypeError, ValueError):
        points = 25
    try:
        rnd = G.create_round(
            room, player["id"], body.get("clip_id", ""), body.get("answer", ""),
            points, rounds=rounds,
        )
    except G.RuleError as e:
        raise ApiError(409, str(e))
    store.add_round(rnd)

    feats = body.get("features")
    suggested = body.get("suggested_points")
    if isinstance(feats, dict) and suggested is not None:
        model = S.Model.from_json(room.get("model"))
        try:
            f = S.Features(
                **{k: float(v) for k, v in feats.items() if k in S.Features.ORDER}
            )
        except (TypeError, ValueError):
            f = None
        if f is not None:
            model.learn(f, points)
            room["model"] = model.to_json()
            store.save_room(room)
            store.record_rating(room["id"], feats, int(suggested), points)
    else:
        store.save_room(room)
    return JSONResponse({"round": G.round_view(rnd, player["id"], room["names"])})


def _round_context(request: Request) -> tuple[dict, dict, dict]:
    player = require_player(request)
    rnd = store.get_round(request.path_params["round_id"])
    if not rnd:
        raise ApiError(404, "no such round")
    room = store.get_room(rnd["room_id"])
    if not room or not G.is_member(room, player["id"]):
        raise ApiError(403, "join this room first")
    return player, room, rnd


def _commit(player: dict, room: dict, rnd: dict) -> JSONResponse:
    G.close_round(rnd)
    store.save_round(rnd)
    if rnd.get("state") == G.CLOSED:
        G.advance_turn(room, rnd)
    store.save_room(room)
    return JSONResponse({"round": G.round_view(rnd, player["id"], room["names"])})


async def guess(request: Request):
    player, room, rnd = _round_context(request)
    body = await body_json(request)
    try:
        G.submit_guess(rnd, player["id"], body.get("text", ""))
    except G.RuleError as e:
        raise ApiError(409, str(e))
    return _commit(player, room, rnd)


async def hint_request(request: Request):
    player, room, rnd = _round_context(request)
    try:
        G.request_hint(rnd, player["id"])
    except G.RuleError as e:
        raise ApiError(409, str(e))
    return _commit(player, room, rnd)


async def hint_deliver(request: Request):
    player, room, rnd = _round_context(request)
    body = await body_json(request)
    clip_id = body.get("clip_id", "")
    meta = store.clip_meta(clip_id)
    if not meta or meta["room_id"] != room["id"]:
        raise ApiError(404, "no such clip")
    try:
        G.deliver_hint(rnd, player["id"], clip_id)
    except G.RuleError as e:
        raise ApiError(409, str(e))
    return _commit(player, room, rnd)


async def reveal(request: Request):
    player, room, rnd = _round_context(request)
    try:
        G.reveal(rnd, player["id"])
    except G.RuleError as e:
        raise ApiError(409, str(e))
    return _commit(player, room, rnd)


async def dispute(request: Request):
    player, room, rnd = _round_context(request)
    try:
        G.dispute(rnd, player["id"])
    except G.RuleError as e:
        raise ApiError(409, str(e))
    return _commit(player, room, rnd)


async def review(request: Request):
    player, room, rnd = _round_context(request)
    body = await body_json(request)
    try:
        G.resolve_dispute(
            rnd, player["id"], body.get("player_id", ""), bool(body.get("accept"))
        )
    except G.RuleError as e:
        raise ApiError(409, str(e))
    return _commit(player, room, rnd)


async def close_round(request: Request):
    player, room, rnd = _round_context(request)
    try:
        G.close_round(rnd, by=player["id"])
    except G.RuleError as e:
        raise ApiError(409, str(e))
    return _commit(player, room, rnd)


async def health(request: Request):
    return JSONResponse(
        {"ok": True, "tools": X.tools_available(), "stats": store.stats()}
    )


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

async def api_error(request: Request, exc: ApiError):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status)


async def audio_error(request: Request, exc: Exception):
    return JSONResponse({"detail": str(exc)}, status_code=400)


async def room_page(request: Request):
    """/JOINCODE deep link - serve the app; the client reads the path."""
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise ApiError(404, "client not built")
    return FileResponse(index, media_type="text/html")


routes = [
    Route("/api/health", health),
    Route("/api/session", session, methods=["POST"]),
    Route("/api/me", me),
    Route("/api/rooms", create_room, methods=["POST"]),
    Route("/api/rooms/{code}/join", join_room, methods=["POST"]),
    Route("/api/rooms/{code}/leave", leave_room, methods=["POST"]),
    Route("/api/rooms/{code}/state", room_state),
    Route("/api/rooms/{code}/search", start_search, methods=["POST"]),
    Route("/api/rooms/{code}/clips", make_clip, methods=["POST"]),
    Route("/api/rooms/{code}/audio/{clip_id}.wav", audio),
    Route("/api/rooms/{code}/rounds", send_round, methods=["POST"]),
    Route("/api/search/{job_id}/stream", stream_search),
    Route("/api/preview/{token}.wav", preview_wav),
    Route("/api/preview/{token}/peaks", preview_peaks),
    Route("/api/rounds/{round_id}/guess", guess, methods=["POST"]),
    Route("/api/rounds/{round_id}/hint-request", hint_request, methods=["POST"]),
    Route("/api/rounds/{round_id}/hint", hint_deliver, methods=["POST"]),
    Route("/api/rounds/{round_id}/reveal", reveal, methods=["POST"]),
    Route("/api/rounds/{round_id}/dispute", dispute, methods=["POST"]),
    Route("/api/rounds/{round_id}/review", review, methods=["POST"]),
    Route("/api/rounds/{round_id}/close", close_round, methods=["POST"]),
    Route("/r/{code}", room_page),
]

if WEB_DIR.exists():
    routes.append(Mount("/", app=StaticFiles(directory=str(WEB_DIR), html=True)))


async def _reaper():
    while True:
        await asyncio.sleep(600)
        now = time.time()
        for tok, p in list(PREVIEWS.items()):
            if p["expires"] < now:
                PREVIEWS.pop(tok, None)
        for jid, job in list(JOBS.items()):
            if job.done and now - job.created > PREVIEW_TTL:
                JOBS.pop(jid, None)
                shutil.rmtree(job.workdir, ignore_errors=True)
        try:
            store.purge_idle_rooms()
        except Exception:
            pass


@contextlib.asynccontextmanager
async def lifespan(_app):
    reaper = asyncio.create_task(_reaper())
    try:
        yield
    finally:
        reaper.cancel()


app = Starlette(
    routes=routes,
    lifespan=lifespan,
    exception_handlers={ApiError: api_error, au.AudioError: audio_error},
)
