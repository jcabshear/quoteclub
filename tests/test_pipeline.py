"""End-to-end checks that actually run ffmpeg and the real HTTP app.

These are the tests that distinguish "the code is plausible" from "the
bytes come out right". They do not reach the network: a source file is
synthesised locally with ffmpeg and extraction is stubbed at the one
boundary that needs the internet.
"""

import asyncio
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

from server import audioutil as au  # noqa: E402
from server import extract as X  # noqa: E402


def make_source(path: Path, seconds: float = 6.0) -> None:
    """A short AAC-in-MP4 file with a tone burst pattern."""
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i",
            f"sine=frequency=700:duration={seconds}:sample_rate=48000",
            "-af", "volume=0.6",
            "-c:a", "aac", "-b:a", "96k", str(path),
        ],
        check=True,
    )


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not installed")
class FfmpegPipeline(unittest.TestCase):
    """Real transcode, real probe, real verification."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="qc-pipe-"))
        cls.src = cls.tmp / "source.m4a"
        make_source(cls.src, 6.0)

    def test_source_was_actually_created(self):
        self.assertTrue(self.src.exists())
        self.assertGreater(self.src.stat().st_size, 2000)

    def test_transcode_to_canonical_wav(self):
        dest = self.tmp / "canon.wav"
        asyncio.run(X.to_canonical_wav(self.src, dest))
        duration, rate = X.verify(dest)
        self.assertEqual(rate, au.CANON_RATE)
        self.assertAlmostEqual(duration, 6.0, delta=0.15)

    def test_probe_duration_matches(self):
        d = asyncio.run(X.probe_duration(self.src))
        self.assertAlmostEqual(d, 6.0, delta=0.15)

    def test_windowed_transcode_respects_start_and_length(self):
        dest = self.tmp / "win.wav"
        asyncio.run(X.to_canonical_wav(self.src, dest, start=2.0, length=1.5))
        duration, rate = X.verify(dest)
        self.assertAlmostEqual(duration, 1.5, delta=0.08)

    def test_crop_of_a_real_transcode_is_sample_exact(self):
        dest = self.tmp / "canon2.wav"
        asyncio.run(X.to_canonical_wav(self.src, dest))
        samples, rate = au.decode_wav(dest.read_bytes())
        crop = au.resolve_crop(len(samples), rate, 1.2345, 1.9876)
        cut = au.apply_crop(samples, crop)
        self.assertEqual(len(cut), crop.frames)
        # Round-tripping the cut must not change a single sample.
        again, r2 = au.decode_wav(au.encode_wav(cut, rate))
        self.assertEqual(r2, rate)
        self.assertEqual(list(again), list(cut))
        # And the crop must line up with where we asked, to the sample.
        self.assertEqual(crop.start_index, round(1.2345 * rate))
        self.assertEqual(crop.end_index, round(1.9876 * rate))

    def test_a_silent_source_is_rejected_even_though_it_decodes(self):
        silent_src = self.tmp / "silent.m4a"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                "-i", "anullsrc=r=48000:cl=mono", "-t", "3",
                "-c:a", "aac", str(silent_src),
            ],
            check=True,
        )
        dest = self.tmp / "silent.wav"
        asyncio.run(X.to_canonical_wav(silent_src, dest))
        self.assertTrue(dest.exists() and dest.stat().st_size > 1000)
        with self.assertRaises(X.ExtractionError):
            X.verify(dest)

    def test_video_container_yields_audio_only(self):
        vid = self.tmp / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=4",
                "-f", "lavfi", "-i", "sine=frequency=900:duration=4",
                "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                "-shortest", str(vid),
            ],
            check=True,
        )
        dest = self.tmp / "fromvideo.wav"
        asyncio.run(X.to_canonical_wav(vid, dest))
        duration, rate = X.verify(dest)
        self.assertAlmostEqual(duration, 4.0, delta=0.2)
        self.assertEqual(rate, au.CANON_RATE)


def synth_wav(seconds=5.0, rate=au.CANON_RATE) -> bytes:
    n = int(seconds * rate)
    return au.encode_wav(
        array("h", (int(11000 * math.sin(2 * math.pi * 850 * i / rate)) for i in range(n))),
        rate,
    )


@unittest.skipUnless(
    __import__("importlib").util.find_spec("starlette") is not None,
    "starlette not installed",
)
class HttpFlow(unittest.TestCase):
    """The whole multi-room game over real HTTP, with extraction stubbed."""

    @classmethod
    def setUpClass(cls):
        import os

        cls.tmp = Path(tempfile.mkdtemp(prefix="qc-http-"))
        os.environ["QC_DATA_DIR"] = str(cls.tmp / "data")
        os.environ["QC_INSECURE_COOKIES"] = "1"
        for mod in [m for m in list(sys.modules) if m.startswith("server")]:
            del sys.modules[mod]

        from starlette.testclient import TestClient

        from server import extract as XX
        from server import main as M

        cls.M = M
        cls.wav = cls.tmp / "stub.wav"
        cls.wav.write_bytes(synth_wav(6.0))

        async def fake_extract(url, phrase="", workdir=None):
            if "broken" in url:
                raise XX.ExtractionError("HTTP Error 404: Not Found")
            return XX.Extracted(cls.wav, 6.0, au.CANON_RATE, "The Incredibles", None, [])

        # No network in tests: providers return a fixed candidate set and
        # extraction is stubbed. Everything between them is the real code.
        async def fake_yarn(http, phrase, limit=10):
            return [
                M.Candidate(
                    url="https://y.yarn.co/11111111-2222-3333-4444-555555555555.mp4",
                    provider="subtitle_index", transcript=phrase,
                    source_title="The Incredibles", duration=6.0,
                )
            ]

        async def fake_web(http, phrase, count=20):
            return [M.Candidate(url="https://broken.example/v/1.mp3", provider="video")]

        M.yarn.search = fake_yarn
        M.websearch.search = fake_web
        M.X.extract = fake_extract
        M.gate = lambda: (lambda url: "Stub")
        cls.TestClient = TestClient

    # -- helpers -----------------------------------------------------
    def client(self, name):
        """A fresh browser. Context-managed so the app lifespan runs."""
        c = self.enterContext(self.TestClient(self.M.app))
        c.post("/api/session", json={"name": name})
        return c

    def new_room(self, host_name="Ann"):
        host = self.client(host_name)
        r = host.post("/api/rooms", json={"player_name": host_name})
        self.assertEqual(r.status_code, 200, r.text)
        return host, r.json()["code"]

    def join(self, code, name):
        c = self.client(name)
        r = c.post(f"/api/rooms/{code}/join", json={"name": name})
        self.assertEqual(r.status_code, 200, r.text)
        return c

    def make_clip(self, client, code, start=0.5, end=1.25):
        r = client.post(f"/api/rooms/{code}/search", json={"phrase": "this close"})
        self.assertEqual(r.status_code, 200, r.text)
        job = r.json()["job_id"]
        token = None
        with client.stream("GET", f"/api/search/{job}/stream") as s:
            for line in s.iter_lines():
                if not line.startswith("data: "):
                    continue
                ev = __import__("json").loads(line[6:])
                if ev["kind"] == "result":
                    token = ev["token"]
                if ev["kind"] == "done":
                    break
        self.assertIsNotNone(token, "search produced no playable result")
        clip = client.post(
            f"/api/rooms/{code}/clips", json={"token": token, "start": start, "end": end}
        )
        self.assertEqual(clip.status_code, 200, clip.text)
        return clip.json()

    def send(self, client, code, answer="The Incredibles", points=40):
        clip = self.make_clip(client, code)
        r = client.post(
            f"/api/rooms/{code}/rounds",
            json={"clip_id": clip["clip_id"], "answer": answer, "points": points,
                  "features": clip["features"],
                  "suggested_points": clip["suggested_points"]},
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["round"]["id"], clip

    def state(self, client, code):
        r = client.get(f"/api/rooms/{code}/state")
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    # -- rooms and access --------------------------------------------
    def test_health_reports_tools(self):
        t = self.client("x").get("/api/health").json()
        self.assertIn("ffmpeg", t["tools"])
        self.assertIn("rooms", t["stats"])

    def test_room_codes_are_six_letters_and_unique(self):
        _, a = self.new_room("Ann")
        _, b = self.new_room("Bob")
        self.assertRegex(a, r"^[A-Z]{6}$")
        self.assertNotEqual(a, b)

    def test_no_login_is_required_anywhere(self):
        """A brand new browser can create a room and play with no account."""
        c = self.enterContext(self.TestClient(self.M.app))
        r = c.post("/api/rooms", json={"player_name": "Nobody"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.state(c, r.json()["code"])["you"]["name"], "Nobody")

    def test_codes_are_accepted_however_they_are_typed(self):
        host, code = self.new_room()
        pretty = f"{code[:3].lower()}-{code[3:].lower()}"
        c = self.client("Bob")
        self.assertEqual(
            c.post(f"/api/rooms/{pretty.replace('-', '')}/join", json={}).status_code, 200
        )

    def test_many_groups_play_at_once_without_seeing_each_other(self):
        h1, c1 = self.new_room("Ann")
        h2, c2 = self.new_room("Zoe")
        self.join(c1, "Bob")
        self.join(c2, "Yan")
        self.send(h1, c1, answer="The Incredibles")
        self.send(h2, c2, answer="Ratatouille")
        s1, s2 = self.state(h1, c1), self.state(h2, c2)
        self.assertEqual(len(s1["rounds"]), 1)
        self.assertEqual(len(s2["rounds"]), 1)
        self.assertEqual(s1["rounds"][0]["answer"], "The Incredibles")
        self.assertEqual(s2["rounds"][0]["answer"], "Ratatouille")

    def test_a_stranger_with_no_code_cannot_reach_a_room(self):
        _, code = self.new_room()
        stranger = self.client("Mallory")
        self.assertEqual(stranger.get(f"/api/rooms/{code}/state").status_code, 403)

    def test_a_wrong_code_is_a_404_not_a_leak(self):
        c = self.client("Mallory")
        self.assertEqual(c.post("/api/rooms/ZZZZZZ/join", json={}).status_code, 404)

    def test_one_person_can_be_in_several_rooms(self):
        host, c1 = self.new_room("Ann")
        _, c2 = self.new_room("Zoe")
        host.post(f"/api/rooms/{c2}/join", json={})
        mine = {r["code"] for r in host.get("/api/me").json()["rooms"]}
        self.assertIn(c1, mine)
        self.assertIn(c2, mine)

    def test_a_room_fills_up(self):
        _, code = self.new_room("Ann")
        for i in range(self.M.G.MAX_PLAYERS - 1):
            self.join(code, f"P{i}")
        extra = self.client("Late")
        self.assertEqual(
            extra.post(f"/api/rooms/{code}/join", json={}).status_code, 409
        )

    # -- clip mechanics ----------------------------------------------
    def test_clip_endpoint_returns_exact_duration_and_a_suggestion(self):
        host, code = self.new_room()
        self.join(code, "Bob")
        c = self.make_clip(host, code, 0.5, 1.25)
        self.assertAlmostEqual(c["duration"], 0.75, places=4)
        self.assertTrue(1 <= c["suggested_points"] <= 100)
        self.assertEqual(c["suggested_answer"], "The Incredibles")

    def test_the_sent_audio_is_byte_identical_to_the_selection(self):
        host, code = self.new_room()
        self.join(code, "Bob")
        c = self.make_clip(host, code, 1.0, 2.0)
        got = host.get(f"/api/rooms/{code}/audio/{c['clip_id']}.wav")
        self.assertEqual(got.status_code, 200)
        samples, rate = au.decode_wav(got.content)
        source, srate = au.decode_wav(self.wav.read_bytes())
        crop = au.resolve_crop(len(source), srate, 1.0, 2.0)
        self.assertEqual(list(samples), list(au.apply_crop(source, crop)))
        self.assertEqual(rate, srate)

    def test_audio_response_carries_no_identifying_filename(self):
        host, code = self.new_room()
        self.join(code, "Bob")
        c = self.make_clip(host, code)
        cd = host.get(f"/api/rooms/{code}/audio/{c['clip_id']}.wav").headers.get(
            "content-disposition", ""
        )
        self.assertIn("clip.wav", cd)
        self.assertNotIn("incredible", cd.lower())

    def test_audio_is_not_reachable_from_another_room(self):
        host, code = self.new_room("Ann")
        self.join(code, "Bob")
        c = self.make_clip(host, code)
        other, code2 = self.new_room("Zoe")
        self.assertEqual(
            other.get(f"/api/rooms/{code2}/audio/{c['clip_id']}.wav").status_code, 404
        )

    def test_search_stream_never_leaks_a_source_url(self):
        host, code = self.new_room()
        self.join(code, "Bob")
        r = host.post(f"/api/rooms/{code}/search", json={"phrase": "test"})
        body = ""
        with host.stream("GET", f"/api/search/{r.json()['job_id']}/stream") as s:
            for line in s.iter_lines():
                body += line + "\n"
                if '"done"' in line:
                    break
        self.assertIn('"result"', body)
        self.assertNotIn("yarn.co", body)
        self.assertNotIn("http", body.replace("https://", "").replace("http://", ""))

    # -- the multiplayer round ---------------------------------------
    def test_a_four_player_round_end_to_end(self):
        host, code = self.new_room("Ann")
        bob = self.join(code, "Bob")
        cal = self.join(code, "Cal")
        dee = self.join(code, "Dee")
        rid, _ = self.send(host, code, points=40)

        # Nobody but the sender sees the answer yet.
        for c in (bob, cal, dee):
            self.assertNotIn("answer", self.state(c, code)["rounds"][0])

        # Bob solves at full value.
        self.assertEqual(
            bob.post(f"/api/rounds/{rid}/guess", json={"text": "the incredibles"}).status_code,
            200,
        )
        self.assertEqual(self.state(bob, code)["rounds"][0]["your_awarded"], 40)
        self.assertEqual(self.state(bob, code)["rounds"][0]["answer"], "The Incredibles")
        # ...and Cal still cannot see it.
        self.assertNotIn("answer", self.state(cal, code)["rounds"][0])

        # Cal asks for a hint; asking is free, delivery costs 25%.
        cal.post(f"/api/rounds/{rid}/hint-request")
        self.assertEqual(self.state(cal, code)["rounds"][0]["reward_now"], 40)
        hint = self.make_clip(host, code, 3.0, 3.6)
        host.post(f"/api/rounds/{rid}/hint", json={"clip_id": hint["clip_id"]})
        self.assertEqual(self.state(cal, code)["rounds"][0]["reward_now"], 30)

        cal.post(f"/api/rounds/{rid}/guess", json={"text": "Incredibles"})
        dee.post(f"/api/rounds/{rid}/reveal")

        final = self.state(host, code)
        self.assertEqual(final["rounds"][0]["state"], "closed")
        scores = {p["name"]: p["points"] for p in final["scoreboard"]}
        self.assertEqual(scores["Bob"], 40)   # banked before the hint
        self.assertEqual(scores["Cal"], 30)   # after the hint
        self.assertEqual(scores["Dee"], 0)
        self.assertEqual(scores["Ann"], 0)    # the sender never scores
        self.assertEqual(final["room"]["turn_name"], "Bob")

    def test_the_sender_cannot_guess_their_own_clip(self):
        host, code = self.new_room("Ann")
        self.join(code, "Bob")
        rid, _ = self.send(host, code)
        r = host.post(f"/api/rounds/{rid}/guess", json={"text": "The Incredibles"})
        self.assertEqual(r.status_code, 409)

    def test_out_of_turn_sends_are_refused(self):
        host, code = self.new_room("Ann")
        bob = self.join(code, "Bob")
        clip = self.make_clip(bob, code)
        r = bob.post(
            f"/api/rooms/{code}/rounds",
            json={"clip_id": clip["clip_id"], "answer": "Up", "points": 10},
        )
        self.assertEqual(r.status_code, 409)
        self.assertIn("Ann", r.json()["detail"])

    def test_only_one_clip_in_play(self):
        host, code = self.new_room("Ann")
        self.join(code, "Bob")
        self.send(host, code)
        clip = self.make_clip(host, code)
        r = host.post(
            f"/api/rooms/{code}/rounds",
            json={"clip_id": clip["clip_id"], "answer": "Up", "points": 10},
        )
        self.assertEqual(r.status_code, 409)

    def test_sender_can_end_a_round_early(self):
        host, code = self.new_room("Ann")
        bob = self.join(code, "Bob")
        rid, _ = self.send(host, code)
        self.assertEqual(host.post(f"/api/rounds/{rid}/close").status_code, 200)
        s = self.state(bob, code)
        self.assertEqual(s["rounds"][0]["state"], "closed")
        self.assertIn("answer", s["rounds"][0])

    def test_a_guesser_cannot_end_a_round(self):
        host, code = self.new_room("Ann")
        bob = self.join(code, "Bob")
        rid, _ = self.send(host, code)
        self.assertEqual(bob.post(f"/api/rounds/{rid}/close").status_code, 409)

    def test_dispute_review_by_the_sender(self):
        host, code = self.new_room("Ann")
        bob = self.join(code, "Bob")
        rid, _ = self.send(host, code, points=50)
        bob.post(f"/api/rounds/{rid}/guess", json={"text": "that Pixar superhero one"})
        bob.post(f"/api/rounds/{rid}/dispute")
        bob_id = self.state(bob, code)["you"]["id"]
        disputes = self.state(host, code)["rounds"][0]["disputes"]
        self.assertEqual(disputes[0]["player_id"], bob_id)
        host.post(f"/api/rounds/{rid}/review", json={"player_id": bob_id, "accept": True})
        self.assertEqual(self.state(bob, code)["rounds"][0]["your_awarded"], 50)

    def test_someone_elses_wrong_guess_is_not_shown(self):
        host, code = self.new_room("Ann")
        bob = self.join(code, "Bob")
        cal = self.join(code, "Cal")
        rid, _ = self.send(host, code)
        bob.post(f"/api/rounds/{rid}/guess", json={"text": "Shrek Retold Boogaloo"})
        self.assertNotIn("boogaloo", repr(self.state(cal, code)).lower())

    def test_a_non_member_cannot_act_on_a_round(self):
        host, code = self.new_room("Ann")
        self.join(code, "Bob")
        rid, _ = self.send(host, code)
        stranger = self.client("Mallory")
        self.assertEqual(
            stranger.post(f"/api/rounds/{rid}/guess", json={"text": "x"}).status_code, 403
        )

    def test_corrections_train_the_rooms_model(self):
        host, code = self.new_room("Ann")
        bob = self.join(code, "Bob")
        before = self.make_clip(host, code, 0.5, 1.0)["suggested_points"]
        turn = [host, bob]
        for i in range(6):
            sender = turn[i % 2]
            clip = self.make_clip(sender, code, 0.5, 1.0)
            sender.post(
                f"/api/rooms/{code}/rounds",
                json={"clip_id": clip["clip_id"], "answer": "Ratatouille",
                      "points": min(100, before + 35),
                      "features": clip["features"],
                      "suggested_points": clip["suggested_points"]},
            )
            rid = self.state(sender, code)["rounds"][-1]["id"]
            turn[(i + 1) % 2].post(f"/api/rounds/{rid}/guess", json={"text": "Ratatouille"})
        after = self.make_clip(host, code, 0.5, 1.0)["suggested_points"]
        self.assertGreater(after, before)

    def test_index_and_deep_link_both_serve_the_app(self):
        c = self.client("Ann")
        self.assertIn("Quote", c.get("/").text)
        self.assertIn("Quote", c.get("/r/ABCDEF").text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
