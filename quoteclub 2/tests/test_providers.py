import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.providers import websearch, yarn  # noqa: E402

YARN_HTML = """
<html><head>
<meta property="og:title" content="YARN | You were this close to losing your job | The Incredibles (2004) | Video clips by quotes | 3a0f">
</head><body>
<a href="/yarn-clip/11111111-2222-3333-4444-555555555555">
  <div class="clip-transcript">You were this close to losing your job.</div></a>
<a href="/yarn-clip/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee">
  <div class="clip-transcript">This close.</div></a>
<a href="/yarn-clip/11111111-2222-3333-4444-555555555555">duplicate</a>
</body></html>
"""

YARN_JSON = """
{"clips": [
  {"id": "11111111-2222-3333-4444-555555555555",
   "text": "You were this close to losing your job.",
   "episode_title": "The Incredibles", "duration": 4.2},
  {"id": "not-a-uuid", "text": "junk"},
  {"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
   "transcript": "This close.", "media_title": {"title": "The Incredibles"}}
]}
"""


class YarnParsing(unittest.TestCase):
    def test_html_yields_clip_media_urls(self):
        cands = yarn.parse_search(YARN_HTML, "text/html")
        self.assertEqual(len(cands), 2)
        self.assertEqual(
            cands[0].url,
            "https://y.yarn.co/11111111-2222-3333-4444-555555555555.mp4",
        )
        self.assertEqual(cands[0].provider, "subtitle_index")
        self.assertIn("this close", cands[0].transcript.lower())

    def test_duplicate_clip_ids_are_collapsed(self):
        self.assertEqual(len({c.url for c in yarn.parse_search(YARN_HTML)}), 2)

    def test_json_payload_is_parsed_when_served(self):
        cands = yarn.parse_search(YARN_JSON, "application/json")
        self.assertEqual(len(cands), 2)
        self.assertEqual(cands[0].source_title, "The Incredibles")
        self.assertEqual(cands[0].duration, 4.2)
        self.assertEqual(cands[1].source_title, "The Incredibles")

    def test_unknown_shapes_return_nothing_rather_than_raising(self):
        for body in ("", "<html></html>", "{}", "[1,2,3]", "not json at all"):
            self.assertEqual(yarn.parse_search(body), [])

    def test_source_title_comes_from_the_corpus_not_a_guess(self):
        self.assertEqual(
            yarn.source_title_from_clip_page(YARN_HTML), "The Incredibles (2004)"
        )

    def test_episode_suffix_is_trimmed_to_the_show(self):
        html = (
            '<meta property="og:title" content="YARN | Wheres my supersuit | '
            'The Office (2005) - S04E20 Comedy | Video clips by quotes | 3578">'
        )
        self.assertEqual(yarn.source_title_from_clip_page(html), "The Office (2005)")

    def test_missing_title_is_none(self):
        self.assertIsNone(yarn.source_title_from_clip_page("<html></html>"))

    def test_search_url_is_escaped(self):
        u = yarn.search_url('you were "this" close')
        self.assertIn("yarn-find", u)
        self.assertNotIn(" ", u)


BRAVE = {
    "videos": {
        "results": [
            {
                "url": "https://coub.com/view/5cru0",
                "title": "The Incredibles: You were this close to losing your job",
                "video": {"duration": "0:10"},
            }
        ]
    },
    "web": {
        "results": [
            {
                "url": "https://www.pinterest.com/pin/12345/",
                "title": "<strong>You were this close</strong>",
                "description": "a board",
            },
            {
                "url": "https://example.org/scene",
                "title": "scene",
                "properties": {"video": "https://example.org/v.mp4"},
            },
        ]
    },
}


class BraveParsing(unittest.TestCase):
    def test_video_and_web_sections_both_parsed(self):
        cands = websearch.parse_brave(BRAVE)
        self.assertEqual(len(cands), 3)
        by_url = {c.url: c for c in cands}
        self.assertTrue(by_url["https://coub.com/view/5cru0"].media_evidence)
        self.assertEqual(by_url["https://coub.com/view/5cru0"].duration, 10.0)
        self.assertFalse(by_url["https://www.pinterest.com/pin/12345/"].media_evidence)
        self.assertTrue(by_url["https://example.org/scene"].media_evidence)

    def test_markup_is_stripped_from_titles(self):
        c = [x for x in websearch.parse_brave(BRAVE) if "pinterest" in x.url][0]
        self.assertEqual(c.title, "You were this close")

    def test_duration_strings_are_converted(self):
        self.assertEqual(websearch._seconds({"video": {"duration": "1:02:03"}}), 3723.0)
        self.assertIsNone(websearch._seconds({}))

    def test_query_variants_put_the_exact_phrase_first(self):
        v = websearch.query_variants("pull the lever kronk")
        self.assertEqual(v[0], '"pull the lever kronk"')
        self.assertGreater(len(v), 2)

    def test_pasted_urls_become_candidates(self):
        c = websearch.from_pasted_urls(["  https://a.example/x.mp4 ", "", "  "])
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0].url, "https://a.example/x.mp4")

    def test_empty_payload_is_safe(self):
        self.assertEqual(websearch.parse_brave({}), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
