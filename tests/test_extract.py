import math
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import audioutil as au  # noqa: E402
from server import extract as X  # noqa: E402
from server.discovery import FailureClass  # noqa: E402


def tone(freq, seconds, rate=44100, amp=12000):
    n = int(seconds * rate)
    return array("h", (int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)))


def write_wav(samples, rate=44100) -> Path:
    p = Path(tempfile.mkdtemp()) / "a.wav"
    p.write_bytes(au.encode_wav(samples, rate))
    return p


class Classification(unittest.TestCase):
    def test_login_walls_and_geo_blocks_are_host_level(self):
        for msg in (
            "ERROR: Sign in to confirm you're not a bot",
            "This video is private video",
            "Video not available in your country",
            "HTTP Error 403: Forbidden",
        ):
            self.assertEqual(X.classify(msg), FailureClass.HOST_BLOCKED, msg)

    def test_timeouts_and_5xx_are_transient(self):
        for msg in (
            "The read operation timed out",
            "HTTP Error 503: Service Unavailable",
            "HTTP Error 429: Too Many Requests",
            "Connection reset by peer",
        ):
            self.assertEqual(X.classify(msg), FailureClass.TRANSIENT, msg)

    def test_everything_else_is_url_specific(self):
        """Regression: a 422 used to blacklist an entire hostname."""
        for msg in (
            "Unprocessable entity",
            "HTTP Error 404: Not Found",
            "Requested format is not available",
            "no audio stream found",
        ):
            self.assertEqual(X.classify(msg), FailureClass.URL_SPECIFIC, msg)


class Verification(unittest.TestCase):
    """'It decoded' is not the same as 'it contains the audio'."""

    def test_real_audio_passes(self):
        p = write_wav(tone(800, 2.0))
        duration, rate = X.verify(p)
        self.assertAlmostEqual(duration, 2.0, places=2)
        self.assertEqual(rate, 44100)

    def test_silent_file_is_rejected(self):
        p = write_wav(array("h", bytes(2 * 44100 * 2)))
        with self.assertRaises(X.ExtractionError) as cm:
            X.verify(p)
        self.assertIn("silent", str(cm.exception))

    def test_near_silent_file_is_rejected(self):
        p = write_wav(tone(800, 2.0, amp=20))
        with self.assertRaises(X.ExtractionError):
            X.verify(p)

    def test_too_short_to_be_a_clip_is_rejected(self):
        p = write_wav(tone(800, 0.1))
        with self.assertRaises(X.ExtractionError) as cm:
            X.verify(p)
        self.assertIn("0.10s", str(cm.exception))

    def test_non_audio_file_is_rejected(self):
        p = Path(tempfile.mkdtemp()) / "b.wav"
        p.write_bytes(b"<html>not audio</html>" * 100)
        with self.assertRaises(X.ExtractionError):
            X.verify(p)


VTT = """WEBVTT

00:00:01.000 --> 00:00:03.500
Where do you think you're going?

00:12:41.120 --> 00:12:43.400
You were <i>this</i> close

00:12:43.400 --> 00:12:45.000
to losing your job.

00:20:00.000 --> 00:20:02.000
Honey, where's my supersuit?
"""

SRT = """1
00:00:05,000 --> 00:00:07,250
Pull the lever, Kronk!

2
00:00:08,000 --> 00:00:09,000
Wrong lever!
"""


class Captions(unittest.TestCase):
    def test_parses_vtt(self):
        cues = X.parse_cues(VTT)
        self.assertEqual(len(cues), 4)
        self.assertAlmostEqual(cues[1][0], 761.12, places=2)
        self.assertEqual(cues[1][2], "You were this close")

    def test_parses_srt_with_comma_milliseconds(self):
        cues = X.parse_cues(SRT)
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[0][0], 5.0)
        self.assertEqual(cues[0][2], "Pull the lever, Kronk!")

    def test_finds_a_line_split_across_two_cues(self):
        cues = X.parse_cues(VTT)
        hit = X.locate_phrase(cues, "you were this close to losing your job")
        self.assertIsNotNone(hit)
        start, end, _ = hit
        self.assertAlmostEqual(start, 761.12, places=2)
        self.assertAlmostEqual(end, 765.0, places=2)

    def test_finds_a_single_cue_line(self):
        cues = X.parse_cues(VTT)
        hit = X.locate_phrase(cues, "honey wheres my supersuit")
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[0], 1200.0, places=2)

    def test_returns_none_when_the_line_is_not_there(self):
        self.assertIsNone(X.locate_phrase(X.parse_cues(VTT), "to infinity and beyond"))

    def test_empty_captions_are_safe(self):
        self.assertEqual(X.parse_cues(""), [])
        self.assertIsNone(X.locate_phrase([], "anything"))


class TitleInference(unittest.TestCase):
    def test_structured_metadata_wins(self):
        self.assertEqual(
            X._title_from_info({"series": "The Office", "title": "clip 12"}),
            "The Office",
        )

    def test_colon_prefix_is_treated_as_the_work(self):
        self.assertEqual(
            X._title_from_info(
                {"title": "The Incredibles: You were this close to losing your job"}
            ),
            "The Incredibles",
        )

    def test_upload_junk_is_trimmed(self):
        self.assertEqual(
            X._title_from_info({"title": "Pull the lever Kronk - HD"}),
            "Pull the lever Kronk",
        )

    def test_no_title_is_none_not_a_guess(self):
        self.assertIsNone(X._title_from_info({}))


class ToolReporting(unittest.TestCase):
    def test_tools_available_reports_honestly(self):
        t = X.tools_available()
        self.assertEqual(set(t), {"ffmpeg", "ffprobe", "yt_dlp"})
        self.assertTrue(all(isinstance(v, bool) for v in t.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
