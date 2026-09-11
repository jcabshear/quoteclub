import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.discovery import (  # noqa: E402
    Candidate,
    FailureClass,
    FailureMemory,
    admissible,
    diversify,
    is_public_http_url,
    looks_like_direct_media,
    phrase_containment,
    plan,
    rank,
)

# Real _VALID_URL patterns copied from yt-dlp so the gate can be exercised
# without the package installed. The production gate uses yt-dlp's own
# registry; this stub exists only so the *policy* is testable offline.
STUB_EXTRACTORS = {
    "Coub": re.compile(
        r"(?:coub:|https?://(?:coub\.com/(?:view|embed|coubs)/"
        r"|c-cdn\.coub\.com/fb-player\.swf\?.*\bcoub(?:ID|id)=))(?P<id>[\da-z]+)"
    ),
    "Youtube": re.compile(
        r"https?://(?:www\.|m\.)?youtube\.com/watch\?v=(?P<id>[\w-]{11})"
    ),
    "YoutubeShort": re.compile(r"https?://youtu\.be/(?P<id>[\w-]{11})"),
}


def stub_gate(url: str):
    for name, rx in STUB_EXTRACTORS.items():
        if rx.match(url):
            return name
    return None


class ExtractorGatePolicy(unittest.TestCase):
    def test_coub_view_url_is_admitted(self):
        """Regression: the Incredibles failure.

        https://coub.com/view/5cru0 contains none of the path words the old
        filter looked for ("watch", "clip", "sound", "video") and the search
        result carried no video metadata, so it was discarded before
        extraction was ever attempted - even though yt-dlp's Coub extractor
        matches it and the page serves a direct MP3.
        """
        c = Candidate(
            url="https://coub.com/view/5cru0",
            title="The Incredibles: You were this close to losing your job",
            provider="video",
        )
        ok, reason = admissible(c, stub_gate)
        self.assertTrue(ok, reason)
        self.assertEqual(c.extra["extractor"], "Coub")

    def test_article_page_without_media_is_rejected(self):
        c = Candidate(
            url="https://www.pinterest.com/pin/834995307/",
            title="You were this close to losing your job",
            provider="web",
        )
        ok, reason = admissible(c, stub_gate)
        self.assertFalse(ok)
        self.assertIn("no extractor", reason)

    def test_article_page_with_advertised_media_is_admitted(self):
        c = Candidate(
            url="https://example.org/posts/incredibles-huph",
            media_evidence=True,
            provider="web",
        )
        ok, _ = admissible(c, stub_gate)
        self.assertTrue(ok)

    def test_direct_media_extension_is_admitted_without_extractor(self):
        c = Candidate(url="https://cdn.example.com/a/b/1471771632_high.mp3")
        self.assertTrue(looks_like_direct_media(c.url))
        ok, _ = admissible(c, stub_gate)
        self.assertTrue(ok)

    def test_hls_and_dash_manifests_are_admitted(self):
        for u in (
            "https://cdn.example.com/stream/master.m3u8",
            "https://cdn.example.com/stream/manifest.mpd",
        ):
            ok, _ = admissible(Candidate(url=u), stub_gate)
            self.assertTrue(ok, u)

    def test_private_and_non_http_urls_are_rejected(self):
        for u in (
            "http://127.0.0.1:8000/a.mp3",
            "http://192.168.1.5/a.mp3",
            "file:///etc/passwd",
            "ftp://example.com/a.mp3",
            "http://169.254.169.254/latest/meta-data/",
        ):
            self.assertFalse(is_public_http_url(u), u)


class FailureMemoryPolicy(unittest.TestCase):
    def setUp(self):
        self.t = [1000.0]
        self.mem = FailureMemory(clock=lambda: self.t[0])

    def test_one_bad_clip_does_not_ban_the_host(self):
        """Regression: the old code added the host to failed_hosts on any
        422 and then skipped every other candidate from that host."""
        self.mem.record("https://coub.com/view/deadbeef", FailureClass.URL_SPECIFIC)
        good = Candidate(url="https://coub.com/view/5cru0")
        ok, reason = admissible(good, stub_gate, self.mem)
        self.assertTrue(ok, reason)
        self.assertFalse(self.mem.host_suspended("coub.com"))

    def test_the_same_url_is_not_retried(self):
        self.mem.record("https://coub.com/view/deadbeef", FailureClass.URL_SPECIFIC)
        again = Candidate(url="https://coub.com/view/deadbeef")
        ok, reason = admissible(again, stub_gate, self.mem)
        self.assertFalse(ok)
        self.assertIn("failed before", reason)

    def test_host_level_block_suspends_with_a_ttl(self):
        self.mem.record("https://walled.example/v/1", FailureClass.HOST_BLOCKED)
        self.assertTrue(self.mem.host_suspended("walled.example"))
        self.t[0] += 31 * 60
        self.assertFalse(self.mem.host_suspended("walled.example"))

    def test_repeated_distinct_failures_demote_but_do_not_ban(self):
        for i in range(5):
            self.mem.record(f"https://flaky.example/v/{i}.mp3", FailureClass.URL_SPECIFIC)
        self.assertGreater(self.mem.host_penalty("flaky.example"), 0.0)
        self.assertFalse(self.mem.host_suspended("flaky.example"))
        ok, reason = admissible(
            Candidate(url="https://flaky.example/v/99.mp3"), stub_gate, self.mem
        )
        self.assertTrue(ok, reason)
        # ...but it now ranks below an equivalent candidate from a clean host.
        demoted = Candidate(url="https://flaky.example/v/99.mp3", title="a line")
        clean = Candidate(url="https://clean.example/v/1.mp3", title="a line")
        ordered = rank("a line", [demoted, clean], self.mem)
        self.assertEqual(ordered[0].host, "clean.example")

    def test_transient_failure_clears_quickly(self):
        self.mem.record("https://slow.example/v/1", FailureClass.TRANSIENT)
        self.assertTrue(self.mem.url_blocked("https://slow.example/v/1"))
        self.t[0] += 61
        self.assertFalse(self.mem.url_blocked("https://slow.example/v/1"))

    def test_success_clears_prior_host_damage(self):
        for i in range(5):
            self.mem.record(f"https://site.example/v/{i}", FailureClass.URL_SPECIFIC)
        self.mem.record_success("https://site.example/v/7")
        self.assertEqual(self.mem.host_penalty("site.example"), 0.0)


class Ranking(unittest.TestCase):
    def test_transcript_match_outranks_title_match(self):
        q = "you were this close to losing your job"
        subtitle = Candidate(
            url="https://getyarn.io/yarn-clip/abc",
            provider="subtitle_index",
            transcript="You were THIS close to losing your job.",
            source_title="The Incredibles",
            duration=4.0,
        )
        webby = Candidate(
            url="https://coub.com/view/5cru0",
            provider="video",
            title="The Incredibles: You were this close to losing your job",
            duration=10.0,
        )
        ordered = rank(q, [webby, subtitle])
        self.assertEqual(ordered[0].url, subtitle.url)

    def test_full_episode_is_outranked_by_the_moment(self):
        q = "pull the lever kronk"
        short = Candidate(url="https://a.example/s.mp3", provider="video", title="Pull the lever Kronk", duration=6.0)
        long = Candidate(url="https://b.example/l.mp4", provider="video", title="Pull the lever Kronk full movie", duration=5400.0)
        ordered = rank(q, [long, short])
        self.assertEqual(ordered[0].url, short.url)

    def test_low_yield_hosts_are_demoted_not_removed(self):
        q = "im a little stitious"
        pin = Candidate(url="https://www.pinterest.com/pin/1", media_evidence=True, title="I'm a little stitious")
        vid = Candidate(url="https://c.example/x.mp4", title="I'm a little stitious - The Office")
        ordered = rank(q, [pin, vid])
        self.assertEqual(ordered[0].url, vid.url)
        self.assertIn(pin, ordered)

    def test_diversify_caps_one_host(self):
        cands = [Candidate(url=f"https://one.example/v/{i}", score=10 - i) for i in range(5)]
        cands.append(Candidate(url="https://two.example/v/1", score=0))
        kept = diversify(cands, per_host=2)
        self.assertEqual(sum(1 for c in kept if c.host == "one.example"), 2)
        self.assertEqual(len(kept), 3)

    def test_phrase_containment(self):
        self.assertEqual(phrase_containment("a b c", "x a b c y"), 1.0)
        self.assertAlmostEqual(phrase_containment("a b c d", "a b zz"), 0.5)
        self.assertEqual(phrase_containment("", "anything"), 0.0)


class Plan(unittest.TestCase):
    def test_plan_reports_why_things_were_dropped(self):
        cands = [
            Candidate(url="https://coub.com/view/5cru0", title="The Incredibles"),
            Candidate(url="https://www.livejournal.com/post/1"),
            Candidate(url="https://coub.com/view/5cru0"),  # duplicate
        ]
        ordered, rejected = plan("you were this close", cands, stub_gate)
        self.assertEqual([c.url for c in ordered], ["https://coub.com/view/5cru0"])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0][0], "https://www.livejournal.com/post/1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
