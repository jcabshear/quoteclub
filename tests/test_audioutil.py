import math
import sys
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import audioutil as au  # noqa: E402


def tone(freq, seconds, rate=44100, amp=12000):
    n = int(seconds * rate)
    return array("h", (int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)))


def silence(seconds, rate=44100):
    return array("h", bytes(2 * int(seconds * rate)))


class WavRoundTrip(unittest.TestCase):
    def test_encode_decode_is_lossless(self):
        src = tone(440, 0.25)
        data = au.encode_wav(src, 44100)
        back, rate = au.decode_wav(data)
        self.assertEqual(rate, 44100)
        self.assertEqual(len(back), len(src))
        self.assertEqual(list(back), list(src))

    def test_stereo_is_downmixed(self):
        stereo = array("h")
        for i in range(1000):
            stereo.append(100)
            stereo.append(300)
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(stereo.tobytes())
        mono, rate = au.decode_wav(buf.getvalue())
        self.assertEqual(rate, 22050)
        self.assertEqual(len(mono), 1000)
        self.assertEqual(mono[0], 200)

    def test_empty_wav_raises_rather_than_returning_silence(self):
        data = au.encode_wav(array("h"), 44100)
        with self.assertRaises(au.AudioError):
            au.decode_wav(data)

    def test_garbage_raises(self):
        with self.assertRaises(au.AudioError):
            au.decode_wav(b"this is not a wav file at all")


class CropExactness(unittest.TestCase):
    def test_crop_is_sample_exact(self):
        src = array("h", range(-1000, 1000))
        crop = au.resolve_crop(len(src), 1000, 0.250, 0.750)
        self.assertEqual(crop.start_index, 250)
        self.assertEqual(crop.end_index, 750)
        out = au.apply_crop(src, crop)
        self.assertEqual(len(out), 500)
        self.assertEqual(out[0], src[250])
        self.assertEqual(out[-1], src[749])

    def test_exported_bytes_match_the_selection_exactly(self):
        """The preview the sender hears and the file the guesser gets are
        produced from the same integer sample range."""
        src = tone(300, 2.0)
        crop = au.resolve_crop(len(src), 44100, 0.4321, 0.9876)
        selected = au.apply_crop(src, crop)
        wav = au.encode_wav(selected, 44100)
        decoded, rate = au.decode_wav(wav)
        self.assertEqual(rate, 44100)
        self.assertEqual(list(decoded), list(selected))
        self.assertAlmostEqual(crop.duration, len(decoded) / 44100, places=9)

    def test_reversed_handles_are_normalised(self):
        crop = au.resolve_crop(44100, 44100, 0.9, 0.2)
        self.assertLess(crop.start_index, crop.end_index)
        self.assertAlmostEqual(crop.start_seconds, 0.2, places=4)

    def test_tiny_crops_are_allowed(self):
        src = tone(1000, 1.0)
        crop = au.resolve_crop(len(src), 44100, 0.500, 0.530)
        self.assertAlmostEqual(crop.duration, 0.030, places=4)
        self.assertEqual(len(au.apply_crop(src, crop)), crop.frames)

    def test_degenerate_crop_is_grown_not_rejected(self):
        src = tone(1000, 1.0)
        crop = au.resolve_crop(len(src), 44100, 0.5, 0.5)
        self.assertGreaterEqual(crop.duration, au.MIN_CROP_SECONDS - 1e-9)

    def test_out_of_range_handles_clamp(self):
        src = tone(1000, 0.5)
        crop = au.resolve_crop(len(src), 44100, -5.0, 99.0)
        self.assertEqual(crop.start_index, 0)
        self.assertEqual(crop.end_index, len(src))


class Peaks(unittest.TestCase):
    def test_peaks_length_is_exactly_the_bucket_count(self):
        for n in (1, 7, 100, 44100):
            p = au.peaks(tone(440, n / 44100), buckets=64)
            self.assertEqual(len(p), 64)

    def test_peaks_are_normalised(self):
        p = au.peaks(tone(440, 0.5, amp=32000), buckets=32)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in p))
        self.assertGreater(max(p), 0.9)

    def test_silence_gives_zero_peaks(self):
        self.assertEqual(au.peaks(silence(0.2), 16), [0.0] * 16)


class Features(unittest.TestCase):
    def test_speech_band_tone_scores_high(self):
        f = au.audio_features(tone(1000, 0.5), 44100)
        self.assertGreater(f.speech_band_ratio, 0.8)

    def test_out_of_band_tone_scores_low(self):
        f = au.audio_features(tone(8000, 0.5), 44100)
        self.assertLess(f.speech_band_ratio, 0.2)

    def test_silence_is_detected(self):
        f = au.audio_features(silence(0.5), 44100)
        self.assertAlmostEqual(f.rms, 0.0, places=6)
        self.assertGreater(f.silence_ratio, 0.9)

    def test_duration_is_reported(self):
        f = au.audio_features(tone(440, 1.5), 44100)
        self.assertAlmostEqual(f.duration, 1.5, places=3)

    def test_onsets_counted_for_separated_bursts(self):
        sig = array("h")
        for _ in range(3):
            sig.extend(silence(0.12))
            sig.extend(tone(900, 0.08))
        f = au.audio_features(sig, 44100)
        self.assertGreaterEqual(f.onset_count, 2)

    def test_empty_crop_raises(self):
        with self.assertRaises(au.AudioError):
            au.audio_features(array("h"), 44100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
