"""Web-search provider (Brave), used as a *widener*, not as the primary.

Web search answers "what pages mention this string". That is a different
question from "where can I hear this line", which is why it produced
Pinterest boards before. It stays in the chain because it is the only
provider that can reach the long tail - a fan upload, a soundboard, a
podcast clip - but it feeds the extractor gate rather than deciding for
itself what is playable.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from ..discovery import Candidate

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# Query shapes tried in order until enough candidates are admitted. The
# quoted phrase goes first because an exact-phrase hit is the strongest
# signal; the widened forms follow.
def query_variants(phrase: str) -> list[str]:
    p = phrase.strip().strip('"')
    return [
        f'"{p}"',
        f'"{p}" clip',
        f'"{p}" scene audio',
        f'{p} movie quote sound',
        p,
    ]


def parse_brave(payload: dict[str, Any]) -> list[Candidate]:
    out: list[Candidate] = []
    for section, provider in (("videos", "video"), ("web", "web")):
        block = payload.get(section) or {}
        for r in block.get("results") or []:
            url = r.get("url")
            if not isinstance(url, str):
                continue
            out.append(
                Candidate(
                    url=url,
                    title=_clean(r.get("title")),
                    snippet=_clean(r.get("description")),
                    provider=provider,
                    duration=_seconds(r),
                    media_evidence=_has_media_evidence(r, section),
                    extra={"brave_section": section},
                )
            )
    return out


def _clean(v: Any) -> str:
    if not isinstance(v, str):
        return ""
    return " ".join(v.replace("<strong>", "").replace("</strong>", "").split())


def _has_media_evidence(r: dict, section: str) -> bool:
    if section == "videos":
        return True
    meta = r.get("video") or r.get("meta_url") or {}
    if isinstance(meta, dict) and (meta.get("duration") or meta.get("embed_url")):
        return True
    props = r.get("properties") or {}
    return bool(isinstance(props, dict) and props.get("video"))


def _seconds(r: dict) -> float | None:
    v = (r.get("video") or {}) if isinstance(r.get("video"), dict) else {}
    raw = v.get("duration") or r.get("duration")
    if isinstance(raw, (int, float)) and raw > 0:
        return float(raw)
    if isinstance(raw, str) and ":" in raw:
        parts = raw.split(":")
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            return None
        total = 0
        for n in nums:
            total = total * 60 + n
        return float(total)
    return None


async def search(http, phrase: str, *, count: int = 20) -> list[Candidate]:
    key = os.environ.get("BRAVE_API_KEY", "").strip()
    if not key:
        return []
    seen: set[str] = set()
    out: list[Candidate] = []
    for q in query_variants(phrase)[:3]:
        try:
            resp = await http.get(
                BRAVE_ENDPOINT,
                params={"q": q, "count": count, "result_filter": "web,videos"},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": key,
                },
            )
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        for c in parse_brave(resp.json()):
            if c.url in seen:
                continue
            seen.add(c.url)
            out.append(c)
        if len(out) >= count:
            break
    return out


def from_pasted_urls(urls: Iterable[str]) -> list[Candidate]:
    """Escape hatch: the sender knows where the clip is and just pastes it."""
    return [Candidate(url=u.strip(), provider="video") for u in urls if u.strip()]
