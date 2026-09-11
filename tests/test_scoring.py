import math
import sys
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import audioutil as au  # noqa: E402
from server import scoring as S  # noqa: E402


def tone(freq, seconds, rate=44100, amp=12000):
    n = int(seconds * rate)
    return array("h", (int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)))


def features_for(samples, rate=44100, **kw):
    af = au.audio_features(samples, rate)
    return S.build_features(af, kw.pop("source_title", "The Office"), **kw)


class HintDecay(unittest.TestCase):
    def test_matches_the_stated_formula(self):
        for base in (5, 20, 40, 100):
            for n in range(0, 6):
                expected = max(1, round(base * 0.75 ** n)) if n else base
                self.assertEqual(S.reward_after_hints(base, n), expected)

    def test_floor_is_one(self):
        self.assertEqual(S.reward_after_hints(1, 10), 1)


class Bounds(unittest.TestCase):
    def test_points_stay_in_range(self):
        for v in (-500, -1, 0, 0.4, 50, 100.6, 10_000, float("nan")):
            p = S.clamp_points(v)
            self.assertTrue(S.MIN_POINTS <= p <= S.MAX_POINTS, p)

    def test_labels_cover_the_whole_range(self):
        for p in range(1, 101):
            self.assertTrue(S.label_for(p))


class SuggestionShape(unittest.TestCase):
    def setUp(self):
        self.m = S.Model()

    def test_a_tiny_fragment_scores_harder_than_a_full_line(self):
        long_line = features_for(tone(1000, 3.0), searched_phrase="you were this close to losing your job")
        fragment = features_for(tone(1000, 0.18), searched_phrase="you were this close to losing your job")
        self.assertGreater(self.m.suggest(fragment), self.m.suggest(long_line))

    def test_duration_not_release_year_drives_difficulty(self):
        """'Length of time' means clip duration."""
        short = features_for(tone(1000, 0.4))
        longer = features_for(tone(1000, 4.0))
        self.assertGreater(self.m.suggest(short), self.m.suggest(longer))

    def test_a_familiar_source_is_easier_than_an_unseen_one(self):
        hist = {"office": 9}
        familiar = features_for(tone(1000, 2.0), source_title="The Office", history=hist)
        unseen = features_for(tone(1000, 2.0), source_title="Ravenous", history=hist)
        self.assertLess(self.m.suggest(familiar), self.m.suggest(unseen))

    def test_near_silence_is_harder_than_clear_speech_band_audio(self):
        quiet = features_for(tone(1000, 2.0, amp=200))
        loud = features_for(tone(1000, 2.0, amp=20000))
        self.assertGreater(self.m.suggest(quiet), self.m.suggest(loud))

    def test_music_outside_the_speech_band_is_harder(self):
        speechy = features_for(tone(900, 2.0))
        musicy = features_for(tone(9000, 2.0))
        self.assertGreater(self.m.suggest(musicy), self.m.suggest(speechy))

    def test_it_actually_reads_the_audio(self):
        """Two crops with identical metadata but different content must
        not receive the same suggestion."""
        a = features_for(tone(1000, 1.0, amp=20000))
        b = features_for(array("h", bytes(2 * 44100)))  # 1s of silence
        self.assertNotEqual(self.m.suggest(a), self.m.suggest(b))


class Learning(unittest.TestCase):
    def test_corrections_move_future_suggestions(self):
        m = S.Model()
        f = features_for(tone(1000, 1.0))
        before = m.suggest(f)
        target = min(100, before + 30)
        for _ in range(40):
            m.learn(f, target)
        after = m.suggest(f)
        self.assertGreater(after, before)
        self.assertLessEqual(abs(after - target), 6)

    def test_learning_generalises_to_similar_clips(self):
        m = S.Model()
        train = features_for(tone(1000, 0.3), searched_phrase="hello there friend")
        for _ in range(60):
            m.learn(train, 95)
        similar = features_for(tone(1000, 0.35), searched_phrase="hello there friend")
        self.assertGreater(m.suggest(similar), S.Model().suggest(similar))

    def test_a_single_odd_correction_does_not_upend_the_scale(self):
        m = S.Model()
        f = features_for(tone(1000, 2.0))
        base = m.suggest(f)
        m.learn(f, 100)
        self.assertLess(abs(m.suggest(f) - base), 25)

    def test_model_round_trips_through_json(self):
        m = S.Model()
        f = features_for(tone(1000, 1.0))
        for _ in range(10):
            m.learn(f, 70)
        again = S.Model.from_json(m.to_json())
        self.assertEqual(again.samples, m.samples)
        self.assertEqual(again.suggest(f), m.suggest(f))

    def test_corrupt_saved_model_falls_back_to_the_prior(self):
        m = S.Model.from_json("{not json")
        self.assertEqual(m.bias, S.PRIOR_BIAS)
        self.assertEqual(m.samples, 0)

    def test_learned_points_are_still_clamped(self):
        m = S.Model()
        f = features_for(tone(1000, 1.0))
        for _ in range(500):
            m.learn(f, 100)
        self.assertLessEqual(m.suggest(f), 100)
        self.assertGreaterEqual(m.suggest(f), 1)


class PhraseFit(unittest.TestCase):
    def test_a_short_crop_cannot_hold_a_long_phrase(self):
        af = au.audio_features(tone(1000, 0.25), 44100)
        f = S.build_features(af, "The Incredibles", {}, "you were this close to losing your job")
        self.assertLess(f.word_fraction, 0.2)

    def test_a_long_crop_can_hold_the_whole_phrase(self):
        af = au.audio_features(tone(1000, 4.0), 44100)
        f = S.build_features(af, "The Incredibles", {}, "you were this close to losing your job")
        self.assertGreater(f.word_fraction, 0.9)

    def test_a_provided_transcript_overrides_the_duration_guess(self):
        af = au.audio_features(tone(1000, 4.0), 44100)
        f = S.build_features(
            af, "The Incredibles", {},
            searched_phrase="you were this close to losing your job",
            transcript_in_crop="you were this",
        )
        self.assertLess(f.word_fraction, 0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
