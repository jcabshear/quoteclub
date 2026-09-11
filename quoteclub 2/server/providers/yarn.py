"""Yarn (getyarn.io) - a subtitle-indexed clip corpus.

Why this provider is first in the chain: a web search engine finds pages
that *mention* a quote. A subtitle index finds the moment the line was
*spoken*, and hands back the source title with it. That single difference
fixes both of the project's headline problems at once - discovery recall,
and answer autofill that does not depend on scraping a video title.

VERIFICATION STATUS: the clip-media URL shape (`https://y.yarn.co/<id>.mp4`)
is confirmed from a third-party ripper. The *search* response shape is not
confirmed - this module therefore parses defensively: JSON if the endpoint
returns JSON, otherwise clip ids and transcripts lifted out of the HTML.
Run `python -m server.providers.yarn "<quote>"` once from a machine with
network access to see what actually comes back before trusting it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional
from urllib.parse import quote_plus

from ..discovery import Candidate

SEARCH_URL = "https://getyarn.io/yarn-find?text={q}"
CLIP_PAGE = "https://getyarn.io/yarn-clip/{cid}"
CLIP_MEDIA = "https://y.yarn.co/{cid}.mp4"

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_CLIP_HREF = re.compile(r"/yarn-clip/(" + _UUID + r")", re.I)
_TRANSCRIPT_BLOCK = re.compile(
    r'class="[^"]*clip-transcript[^"]*"[^>]*>(.*?)</', re.I | re.S
)
_TAG = re.compile(r"<[^>]+>")


def _text(html_fragment: str) -> str:
    return " ".join(_TAG.sub(" ", html_fragment).split())


def parse_search(body: str, content_type: str = "") -> list[Candidate]:
    """Parse a Yarn search response into candidates.

    Handles a JSON payload if one is served, and falls back to pulling
    clip ids + transcripts out of the HTML. Unknown shapes yield an empty
    list rather than raising, so one provider cannot take down a search.
    """
    if "json" in (content_type or "").lower() or body.lstrip()[:1] in "[{":
        try:
            return _parse_json(json.loads(body))
        except (ValueError, TypeError, KeyError):
            pass
    return _parse_html(body)


def _parse_json(payload: Any) -> list[Candidate]:
    items = payload
    if isinstance(payload, dict):
        for key in ("clips", "results", "hits", "data", "items"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
    if not isinstance(items, list):
        return []
    out: list[Candidate] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = _first(it, ("id", "clip_id", "uuid", "hash"))
        if not cid or not re.fullmatch(_UUID, str(cid), re.I):
            continue
        out.append(
            Candidate(
                url=CLIP_MEDIA.format(cid=cid),
                provider="subtitle_index",
                transcript=_first(it, ("text", "transcript", "body", "subtitle")) or "",
                source_title=_source_title(it),
                duration=_duration(it),
                title=_first(it, ("title", "name")) or "",
                extra={"clip_page": CLIP_PAGE.format(cid=cid), "clip_id": cid},
            )
        )
    return out


def _first(d: dict, keys) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _source_title(it: dict) -> Optional[str]:
    for k in ("episode_title", "movie_title", "show_title", "source", "media_title"):
        v = it.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            inner = _first(v, ("title", "name"))
            if inner:
                return inner
    return None


def _duration(it: dict) -> Optional[float]:
    for k in ("duration", "length", "seconds"):
        v = it.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def _parse_html(html: str) -> list[Candidate]:
    ids: list[str] = []
    for m in _CLIP_HREF.finditer(html or ""):
        cid = m.group(1).lower()
        if cid not in ids:
            ids.append(cid)
    transcripts = [_text(t) for t in _TRANSCRIPT_BLOCK.findall(html or "")]
    out: list[Candidate] = []
    for i, cid in enumerate(ids):
        out.append(
            Candidate(
                url=CLIP_MEDIA.format(cid=cid),
                provider="subtitle_index",
                transcript=transcripts[i] if i < len(transcripts) else "",
                source_title=None,  # filled in from the clip page when needed
                extra={"clip_page": CLIP_PAGE.format(cid=cid), "clip_id": cid},
            )
        )
    return out


_OG_TITLE = re.compile(
    r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', re.I
)
# Yarn page titles look like:
#   YARN | <quote> | <Source Title> | Video clips by quotes | <id>
_YARN_TITLE = re.compile(r"YARN\s*\|(.*?)\|(.*?)\|", re.I | re.S)


def source_title_from_clip_page(html: str) -> Optional[str]:
    """Pull the work's title out of a clip page.

    This is where answer autofill comes from, and it is metadata the corpus
    publishes about itself - not a guess parsed out of an arbitrary video
    title.
    """
    m = _OG_TITLE.search(html or "")
    candidate = m.group(1) if m else ""
    if not candidate:
        tm = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.I | re.S)
        candidate = tm.group(1) if tm else ""
    candidate = _text(candidate)
    ym = _YARN_TITLE.search(candidate)
    if ym:
        title = ym.group(2).strip()
        # Strip a trailing "- S04E20 Comedy" style suffix but keep the show.
        title = re.sub(r"\s*-\s*S\d{1,2}E\d{1,2}.*$", "", title, flags=re.I).strip()
        return title or None
    return None


def search_url(phrase: str) -> str:
    return SEARCH_URL.format(q=quote_plus(phrase))


async def search(http, phrase: str, limit: int = 10) -> list[Candidate]:
    """Fetch and parse. `http` is an httpx.AsyncClient-like object."""
    resp = await http.get(search_url(phrase), follow_redirects=True)
    if resp.status_code >= 400:
        return []
    cands = parse_search(resp.text, resp.headers.get("content-type", ""))
    return cands[:limit]


async def enrich(http, cand: Candidate) -> Candidate:
    """Fill in source_title from the clip page when the search did not."""
    page = cand.extra.get("clip_page")
    if cand.source_title or not page:
        return cand
    try:
        resp = await http.get(page, follow_redirects=True)
    except Exception:
        return cand
    if resp.status_code < 400:
        cand.source_title = source_title_from_clip_page(resp.text)
    return cand


if __name__ == "__main__":  # pragma: no cover - manual verification helper
    import asyncio
    import sys

    async def _main() -> None:
        import httpx

        phrase = " ".join(sys.argv[1:]) or "you were this close to losing your job"
        async with httpx.AsyncClient(
            timeout=20, headers={"User-Agent": "Mozilla/5.0 QuoteClub/1.0"}
        ) as http:
            r = await http.get(search_url(phrase), follow_redirects=True)
            print("HTTP", r.status_code, r.headers.get("content-type"))
            print("--- first 1500 bytes of body ---")
            print(r.text[:1500])
            print("--- parsed ---")
            for c in parse_search(r.text, r.headers.get("content-type", "")):
                print(f"{c.url}\n  transcript: {c.transcript!r}\n  title: {c.source_title!r}")

    asyncio.run(_main())
