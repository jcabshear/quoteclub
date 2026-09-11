"""Getting real audio out of a candidate URL.

Differences from the previous approach, in order of how much they matter:

* **HLS and DASH are not special.** yt-dlp hands segmented streams to
  ffmpeg and ffmpeg concatenates them. Refusing `.m3u8` / `.mpd` sources
  threw away a large share of the modern web for no reason.
* **Long sources are windowed, not rejected.** The old rule ("longer than
  three minutes with no captions -> reject") discarded usable material.
  Here, captions are used to *seek* when they exist, and when they do not
  we fetch a bounded window around the best guess and let the sender scrub.
* **"It decoded" is not success.** A file that parses but is silent, or is
  400 ms of a station ident, is a failure. `verify()` checks duration and
  actual signal energy before a candidate is ever shown.
* **Failures are classified**, so `discovery.FailureMemory` can tell
  "this clip is gone" from "this whole site wants a login".
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import audioutil as au
from .discovery import FailureClass

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")

MAX_SOURCE_SECONDS = float(os.environ.get("MAX_SOURCE_SECONDS", 900))   # 15 min
WINDOW_SECONDS = float(os.environ.get("WINDOW_SECONDS", 90))
MAX_BYTES = int(os.environ.get("MAX_SOURCE_BYTES", 80 * 1024 * 1024))
STEP_TIMEOUT = float(os.environ.get("EXTRACT_TIMEOUT", 55))

# A decoded file this quiet contains nothing worth guessing.
SILENCE_RMS_FLOOR = 0.0015
MIN_USEFUL_SECONDS = 0.35


class ExtractionError(Exception):
    def __init__(self, message: str, failure_class: str = FailureClass.URL_SPECIFIC):
        super().__init__(message)
        self.failure_class = failure_class


@dataclass
class Extracted:
    wav_path: Path
    duration: float
    rate: int
    source_title: Optional[str]
    caption_offset: Optional[float]
    notes: list[str]


# --------------------------------------------------------------------------
# Failure classification
# --------------------------------------------------------------------------

_HOST_BLOCKED_PATTERNS = (
    "sign in to confirm", "confirm you're not a bot", "login required",
    "private video", "members-only", "requires authentication",
    "not available in your country", "geo restricted", "geo-restricted",
    "blocked in your country", "403: forbidden", "http error 403",
    "robots.txt", "drm",
)
_TRANSIENT_PATTERNS = (
    "timed out", "timeout", "temporarily unavailable", "http error 5",
    "connection reset", "connection aborted", "too many requests",
    "http error 429", "name or service not known",
)


def classify(message: str) -> str:
    m = (message or "").lower()
    for p in _HOST_BLOCKED_PATTERNS:
        if p in m:
            return FailureClass.HOST_BLOCKED
    for p in _TRANSIENT_PATTERNS:
        if p in m:
            return FailureClass.TRANSIENT
    return FailureClass.URL_SPECIFIC


# --------------------------------------------------------------------------
# ffmpeg helpers
# --------------------------------------------------------------------------

def tools_available() -> dict[str, bool]:
    return {
        "ffmpeg": shutil.which(FFMPEG) is not None,
        "ffprobe": shutil.which(FFPROBE) is not None,
        "yt_dlp": _yt_dlp_available(),
    }


def _yt_dlp_available() -> bool:
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        return False
    return True


async def _run(cmd: list[str], timeout: float = STEP_TIMEOUT) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise ExtractionError("step timed out", FailureClass.TRANSIENT)
    return proc.returncode or 0, (out or b"").decode("utf-8", "replace")


async def probe_duration(path: Path) -> float:
    code, out = await _run([
        FFPROBE, "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ])
    if code != 0:
        raise ExtractionError(f"ffprobe failed: {out[-400:]}")
    try:
        return float(json.loads(out)["format"]["duration"])
    except (ValueError, KeyError, TypeError):
        raise ExtractionError("could not read a duration from the media")


async def to_canonical_wav(
    src: Path, dest: Path, start: float | None = None, length: float | None = None
) -> None:
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.3f}"]
    cmd += ["-i", str(src)]
    if length is not None:
        cmd += ["-t", f"{max(0.05, length):.3f}"]
    cmd += [
        "-vn", "-ac", str(au.CANON_CHANNELS), "-ar", str(au.CANON_RATE),
        "-acodec", "pcm_s16le", "-f", "wav", str(dest),
    ]
    code, out = await _run(cmd)
    if code != 0 or not dest.exists() or dest.stat().st_size < 1024:
        raise ExtractionError(f"ffmpeg could not produce audio: {out[-400:]}")


# --------------------------------------------------------------------------
# Verification - the honesty gate
# --------------------------------------------------------------------------

def verify(wav_path: Path) -> tuple[float, int]:
    """Confirm the file really holds audible source audio.

    Returns (duration, rate). Raises if the file is empty, unreadable,
    too short to be a clip, or effectively silent. A search result is not
    shown to the sender until this passes.
    """
    data = wav_path.read_bytes()
    try:
        samples, rate = au.decode_wav(data)
    except au.AudioError as exc:
        raise ExtractionError(f"decoded file is not usable audio: {exc}")
    duration = len(samples) / rate
    if duration < MIN_USEFUL_SECONDS:
        raise ExtractionError(f"only {duration:.2f}s of audio")
    f = au.audio_features(samples, rate)
    if f.rms < SILENCE_RMS_FLOOR:
        raise ExtractionError("audio track is silent")
    if f.silence_ratio > 0.98:
        raise ExtractionError("audio track is effectively silent")
    return duration, rate


# --------------------------------------------------------------------------
# Captions - used for seeking, never surfaced to the guesser
# --------------------------------------------------------------------------

_TS = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})"
)


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_cues(text: str) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        m = _TS.search(lines[i])
        if not m:
            i += 1
            continue
        start = _to_seconds(*m.group(1, 2, 3, 4))
        end = _to_seconds(*m.group(5, 6, 7, 8))
        body: list[str] = []
        i += 1
        while i < len(lines) and lines[i].strip():
            body.append(re.sub(r"<[^>]+>", "", lines[i]))
            i += 1
        cues.append((start, end, " ".join(" ".join(body).split())))
    return cues


def locate_phrase(cues: list[tuple[float, float, str]], phrase: str) -> Optional[tuple[float, float, str]]:
    from .discovery import phrase_containment

    best = None
    best_score = 0.0
    for start, end, text in cues:
        score = phrase_containment(phrase, text)
        if score > best_score:
            best_score, best = score, (start, end, text)
    if best_score >= 0.75:
        return best
    # Try adjacent pairs - a line is often split across two cues.
    for i in range(len(cues) - 1):
        joined = cues[i][2] + " " + cues[i + 1][2]
        score = phrase_containment(phrase, joined)
        if score > best_score:
            best_score = score
            best = (cues[i][0], cues[i + 1][1], joined)
    return best if best_score >= 0.75 else None


# --------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------

def _ydl_opts(workdir: Path) -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "outtmpl": str(workdir / "src.%(ext)s"),
        # Prefer an audio-only rendition; fall back to anything with sound.
        "format": "bestaudio/best[acodec!=none]/best",
        "noplaylist": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB", "en-orig"],
        "subtitlesformat": "vtt/srt/best",
        "skip_download": False,
        "max_filesize": MAX_BYTES,
        "socket_timeout": 20,
        "retries": 1,
        "fragment_retries": 2,
        "concurrent_fragment_downloads": 4,
        # No cookies, no impersonation, no DRM circumvention. If a site
        # wants an account, we take the failure and move on.
        "cookiefile": None,
        "ignoreerrors": False,
    }


async def extract(
    url: str,
    phrase: str = "",
    workdir: Optional[Path] = None,
) -> Extracted:
    """Download, window, transcode and verify. Raises ExtractionError."""
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ExtractionError(f"yt-dlp unavailable: {exc}", FailureClass.TRANSIENT)

    tmp = Path(workdir or tempfile.mkdtemp(prefix="qc-"))
    tmp.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    def _sync_download() -> dict:
        with yt_dlp.YoutubeDL(_ydl_opts(tmp)) as ydl:
            return ydl.extract_info(url, download=True)

    loop = asyncio.get_running_loop()
    try:
        info = await asyncio.wait_for(
            loop.run_in_executor(None, _sync_download), timeout=STEP_TIMEOUT
        )
    except asyncio.TimeoutError:
        raise ExtractionError("source download timed out", FailureClass.TRANSIENT)
    except Exception as exc:
        msg = str(exc)
        raise ExtractionError(msg, classify(msg))

    if isinstance(info, dict) and info.get("entries"):
        entries = [e for e in info["entries"] if e]
        if not entries:
            raise ExtractionError("no media entries at this url")
        info = entries[0]

    media = _pick_downloaded(tmp)
    if media is None:
        raise ExtractionError("nothing was downloaded")

    total = float(info.get("duration") or 0) or await probe_duration(media)
    if total > MAX_SOURCE_SECONDS:
        raise ExtractionError(
            f"source is {total / 60:.0f} minutes long; too big to work with",
            FailureClass.URL_SPECIFIC,
        )

    start: Optional[float] = None
    length: Optional[float] = None
    caption_offset: Optional[float] = None

    if total > WINDOW_SECONDS:
        hit = _locate_in_subs(tmp, phrase)
        if hit:
            cue_start, cue_end, _text = hit
            pad_before, pad_after = 2.5, 2.5
            start = max(0.0, cue_start - pad_before)
            length = min(WINDOW_SECONDS, (cue_end - cue_start) + pad_before + pad_after)
            caption_offset = cue_start - start
            notes.append("located via captions")
        else:
            # No captions to seek with: give the sender a bounded window
            # from the beginning rather than refusing the source outright.
            start, length = 0.0, WINDOW_SECONDS
            notes.append(
                f"no matching captions; showing the first {int(WINDOW_SECONDS)}s to scrub"
            )

    wav = tmp / "canonical.wav"
    await to_canonical_wav(media, wav, start, length)
    duration, rate = verify(wav)

    return Extracted(
        wav_path=wav,
        duration=duration,
        rate=rate,
        source_title=_title_from_info(info),
        caption_offset=caption_offset,
        notes=notes,
    )


def _pick_downloaded(tmp: Path) -> Optional[Path]:
    best: Optional[Path] = None
    for p in tmp.iterdir():
        if p.suffix.lower() in (".vtt", ".srt", ".json", ".wav"):
            continue
        if p.is_file() and (best is None or p.stat().st_size > best.stat().st_size):
            best = p
    return best


def _locate_in_subs(tmp: Path, phrase: str):
    if not phrase:
        return None
    cues: list[tuple[float, float, str]] = []
    for p in sorted(tmp.glob("*.vtt")) + sorted(tmp.glob("*.srt")):
        try:
            cues.extend(parse_cues(p.read_text("utf-8", "replace")))
        except OSError:
            continue
    if not cues:
        return None
    return locate_phrase(cues, phrase)


_TITLE_JUNK = re.compile(
    r"\s*[\|\-–]\s*(hd|4k|1080p|720p|full scene|scene|clip|movie clip"
    r"|official|subtitles?|english)\s*$",
    re.I,
)


def _title_from_info(info: dict) -> Optional[str]:
    """Best-effort source title from the extractor's own metadata.

    Ordered by how much the field is actually *about* the work rather than
    about the upload. This is still a heuristic - it is labelled as a
    suggestion in the UI and the sender can overwrite it.
    """
    for key in ("series", "album", "movie", "track"):
        v = info.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    title = info.get("title")
    if isinstance(title, str) and title.strip():
        t = title.strip()
        # "The Incredibles: You were this close..." -> "The Incredibles"
        if ":" in t:
            head = t.split(":", 1)[0].strip()
            if 2 <= len(head.split()) <= 6:
                return head
        return _TITLE_JUNK.sub("", t) or None
    return None
