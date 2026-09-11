"""Candidate discovery: turning a typed quote into URLs that plausibly
contain the original audio.

Design notes (this is the part that was broken before):

1. **Never decide what is playable by looking at words in the URL path.**
   `https://coub.com/view/5cru0` is a perfectly good media URL and contains
   none of "watch", "clip", "sound" or "video". The only authority on
   whether a URL can be extracted is the extractor itself, so the gate is
   `yt_dlp`'s own registry (`InfoExtractor.suitable`, excluding the
   generic fallback), plus direct media extensions, plus explicit media
   evidence advertised by the page (og:video / og:audio / enclosure).

2. **A failure is attached to a URL, not to a hostname.** One clip that
   404s does not prove the site is useless. Hosts are *demoted* in ranking
   as they accumulate distinct failures, and only hard host-level signals
   (auth wall, geo block, robots refusal) suspend a host, with a TTL.

3. **Subtitle-indexed corpora beat web search.** Searching the web for a
   quote finds pages that *mention* it. Searching a subtitle index finds
   the moment it was *said*, with the title attached - which also gives
   the answer autofill for free instead of scraping titles heuristically.
   Providers are ordered accordingly.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional
from urllib.parse import urlparse

DIRECT_MEDIA_EXT = (
    ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wav", ".flac", ".wma",
    ".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".m3u8", ".mpd",
)

# Hosts that are known to be article/pinboard/forum surfaces. They are not
# banned - if yt_dlp claims one, or the page advertises real media, it still
# gets through - but with nothing else going for them they rank last.
LOW_YIELD_HINTS = (
    "pinterest.", "livejournal.", "quotes.", "goodreads.", "reddit.com",
    "wikiquote.", "medium.com", "tumblr.com", "facebook.com", "x.com",
    "twitter.com",
)


class FailureClass:
    """Why an extraction attempt failed, and therefore what to remember."""

    TRANSIENT = "transient"        # timeout, 5xx, rate limit -> retry later
    URL_SPECIFIC = "url_specific"  # 404, removed, no audio stream -> this URL only
    HOST_BLOCKED = "host_blocked"  # login wall, geo block, robots -> suspend host


HOST_SUSPEND_SECONDS = 30 * 60
HOST_DEMOTE_AFTER = 3  # distinct failing URLs before a host loses rank


@dataclass
class Candidate:
    url: str
    title: str = ""
    snippet: str = ""
    provider: str = "web"
    # Exact line the corpus says is spoken here, when the provider knows it.
    transcript: Optional[str] = None
    # Source work the corpus attributes the line to, when it knows it.
    source_title: Optional[str] = None
    duration: Optional[float] = None
    media_evidence: bool = False   # page advertised og:video / og:audio / enclosure
    extra: dict = field(default_factory=dict)
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def host(self) -> str:
        return (urlparse(self.url).hostname or "").lower()


class FailureMemory:
    """Per-URL failure memory with soft host demotion.

    Replaces the old `failed_hosts` set, which let a single 422 delete an
    entire website from the search.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._urls: dict[str, tuple[str, float]] = {}
        self._host_failures: dict[str, set[str]] = {}
        self._host_suspended_until: dict[str, float] = {}

    def record(self, url: str, failure_class: str) -> None:
        now = self._clock()
        self._urls[url] = (failure_class, now)
        host = (urlparse(url).hostname or "").lower()
        if failure_class == FailureClass.TRANSIENT:
            return
        self._host_failures.setdefault(host, set()).add(url)
        if failure_class == FailureClass.HOST_BLOCKED:
            self._host_suspended_until[host] = now + HOST_SUSPEND_SECONDS

    def record_success(self, url: str) -> None:
        host = (urlparse(url).hostname or "").lower()
        self._urls.pop(url, None)
        self._host_failures.pop(host, None)
        self._host_suspended_until.pop(host, None)

    def url_blocked(self, url: str) -> bool:
        entry = self._urls.get(url)
        if not entry:
            return False
        cls, when = entry
        if cls == FailureClass.TRANSIENT:
            return self._clock() - when < 60  # brief cooloff only
        return True

    def host_suspended(self, host: str) -> bool:
        until = self._host_suspended_until.get(host)
        return bool(until and self._clock() < until)

    def host_penalty(self, host: str) -> float:
        n = len(self._host_failures.get(host, ()))
        if n < HOST_DEMOTE_AFTER:
            return 0.0
        return min(3.0, 0.5 * (n - HOST_DEMOTE_AFTER + 1))


# --------------------------------------------------------------------------
# The extractor gate
# --------------------------------------------------------------------------

def build_extractor_gate() -> Callable[[str], Optional[str]]:
    """Return f(url) -> extractor name, or None.

    Uses yt_dlp's real registry. The generic extractor is excluded on
    purpose: it matches everything, so including it would make the gate
    meaningless. Import is lazy so this module stays unit-testable with a
    stub gate and no yt_dlp installed.
    """
    from yt_dlp.extractor import gen_extractor_classes  # type: ignore

    classes = [
        ie for ie in gen_extractor_classes()
        if ie.ie_key() not in ("Generic",) and getattr(ie, "_WORKING", True)
    ]

    def gate(url: str) -> Optional[str]:
        for ie in classes:
            try:
                if ie.suitable(url):
                    return ie.ie_key()
            except Exception:  # pragma: no cover - a bad regex must not kill search
                continue
        return None

    return gate


def looks_like_direct_media(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    return path.endswith(DIRECT_MEDIA_EXT)


def is_public_http_url(url: str, resolve: Callable[[str], str] | None = None) -> bool:
    """Reject non-HTTP schemes and anything pointing at private space."""
    try:
        p = urlparse(url)
    except ValueError:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = p.hostname
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if resolve is None:
            return True
        try:
            ip = ipaddress.ip_address(resolve(host))
        except (ValueError, OSError, socket.gaierror):
            return False
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast or ip.is_unspecified
    )


def admissible(
    cand: Candidate,
    gate: Callable[[str], Optional[str]],
    memory: FailureMemory | None = None,
) -> tuple[bool, str]:
    """Can we plausibly get audio out of this? Returns (ok, reason)."""
    if not is_public_http_url(cand.url):
        return False, "not a public http(s) url"
    if memory is not None:
        if memory.url_blocked(cand.url):
            return False, "this exact url failed before"
        if memory.host_suspended(cand.host):
            return False, "host temporarily suspended"

    ie = gate(cand.url)
    if ie:
        cand.extra["extractor"] = ie
        cand.reasons.append(f"extractor:{ie}")
        return True, f"extractor {ie}"
    if looks_like_direct_media(cand.url):
        cand.reasons.append("direct-media-url")
        return True, "direct media url"
    if cand.media_evidence:
        cand.reasons.append("page-advertises-media")
        return True, "page advertises media"
    return False, "no extractor, no media extension, no media evidence"


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9']+")


def normalise(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def phrase_containment(query: str, text: str) -> float:
    """1.0 when the full query phrase appears; otherwise token overlap."""
    q = normalise(query)
    t = normalise(text)
    if not q or not t:
        return 0.0
    qs = " ".join(q)
    ts = " ".join(t)
    if qs in ts:
        return 1.0
    qset = set(q)
    hit = sum(1 for w in set(t) if w in qset)
    return hit / len(qset)


PROVIDER_PRIOR = {
    "subtitle_index": 3.0,  # knows the line and the title
    "soundboard": 1.5,      # short, already isolated audio
    "video": 1.0,
    "web": 0.0,
}


def rank(
    query: str,
    candidates: Iterable[Candidate],
    memory: FailureMemory | None = None,
) -> list[Candidate]:
    out: list[Candidate] = []
    for c in candidates:
        s = PROVIDER_PRIOR.get(c.provider, 0.0)

        # Strongest signal: the corpus itself says this line is spoken here.
        if c.transcript:
            s += 4.0 * phrase_containment(query, c.transcript)
        s += 2.0 * phrase_containment(query, c.title)
        s += 0.75 * phrase_containment(query, c.snippet)

        if c.source_title:
            s += 0.5
        if c.extra.get("extractor"):
            s += 0.5
        if looks_like_direct_media(c.url):
            s += 0.25

        # Short things are cheaper to fetch and likelier to be the moment
        # itself rather than a whole episode.
        if c.duration is not None:
            if c.duration <= 30:
                s += 1.0
            elif c.duration <= 180:
                s += 0.5
            elif c.duration > 1800:
                s -= 1.0

        host = c.host
        if any(h in host for h in LOW_YIELD_HINTS):
            s -= 1.25
        if memory is not None:
            s -= memory.host_penalty(host)

        c.score = s
        out.append(c)

    out.sort(key=lambda c: c.score, reverse=True)
    return out


def diversify(candidates: list[Candidate], per_host: int = 2) -> list[Candidate]:
    """Keep ranking order but stop one host monopolising the attempt budget."""
    seen: dict[str, int] = {}
    out: list[Candidate] = []
    for c in candidates:
        n = seen.get(c.host, 0)
        if n >= per_host:
            continue
        seen[c.host] = n + 1
        out.append(c)
    return out


def plan(
    query: str,
    candidates: Iterable[Candidate],
    gate: Callable[[str], Optional[str]],
    memory: FailureMemory | None = None,
    limit: int = 12,
    per_host: int = 2,
) -> tuple[list[Candidate], list[tuple[str, str]]]:
    """Full pipeline: admit, rank, diversify.

    Returns (ordered attempt list, rejected [(url, reason)]). The rejection
    list is kept and surfaced to the sender, because "we found nothing"
    with no explanation is what made the old behaviour impossible to debug.
    """
    admitted: list[Candidate] = []
    rejected: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    for c in candidates:
        if c.url in seen_urls:
            continue
        seen_urls.add(c.url)
        ok, reason = admissible(c, gate, memory)
        if ok:
            admitted.append(c)
        else:
            rejected.append((c.url, reason))
    ordered = diversify(rank(query, admitted, memory), per_host=per_host)
    return ordered[:limit], rejected
