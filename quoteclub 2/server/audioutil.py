"""Sample-exact WAV handling, waveform peaks, and audio feature extraction.

Everything here is pure stdlib so it can be unit-tested without ffmpeg,
numpy, or a network. ffmpeg is used elsewhere (extract.py) only to get
arbitrary source media *into* a canonical WAV; once it is a WAV, all
cropping is done here on integer sample indices so that the waveform,
the selection, the preview and the exported file can never disagree.
"""

from __future__ import annotations

import cmath
import io
import math
import struct
import wave
from array import array
from dataclasses import dataclass, asdict
from typing import Sequence

# Canonical internal format. Everything is normalised to this on ingest so
# that the editor never has to reason about rate/channel differences.
CANON_RATE = 44100
CANON_CHANNELS = 1
CANON_SAMPWIDTH = 2  # 16-bit PCM

MIN_CROP_SECONDS = 0.02  # 20 ms - deliberately below a single phoneme


class AudioError(ValueError):
    pass


# --------------------------------------------------------------------------
# WAV decode / encode
# --------------------------------------------------------------------------

def decode_wav(data: bytes) -> tuple[array, int]:
    """Decode a PCM WAV into (mono int16 samples, sample_rate).

    Accepts 8/16/32-bit PCM, any channel count, and downmixes to mono.
    Raises AudioError on anything it cannot read, rather than returning
    silence - a decoder that silently yields an empty buffer is how you
    end up shipping a clip that contains no audio.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            nchan = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            nframes = w.getnframes()
            raw = w.readframes(nframes)
    except (wave.Error, EOFError) as exc:  # pragma: no cover - defensive
        raise AudioError(f"not a readable PCM WAV: {exc}") from exc

    if nframes == 0 or not raw:
        raise AudioError("WAV contains zero frames")
    if rate <= 0:
        raise AudioError("WAV has an invalid sample rate")

    samples = _to_int16(raw, width)
    if nchan > 1:
        samples = _downmix(samples, nchan)
    return samples, rate


def _to_int16(raw: bytes, width: int) -> array:
    if width == 2:
        out = array("h")
        out.frombytes(raw[: len(raw) - (len(raw) % 2)])
        return out
    if width == 1:
        # 8-bit WAV is unsigned.
        return array("h", ((b - 128) << 8 for b in raw))
    if width == 4:
        n = len(raw) // 4
        vals = struct.unpack("<%di" % n, raw[: n * 4])
        return array("h", (v >> 16 for v in vals))
    raise AudioError(f"unsupported sample width: {width} bytes")


def _downmix(samples: array, nchan: int) -> array:
    usable = len(samples) - (len(samples) % nchan)
    out = array("h", bytes(2 * (usable // nchan)))
    for i in range(0, usable, nchan):
        total = 0
        for c in range(nchan):
            total += samples[i + c]
        out[i // nchan] = _clip16(total // nchan)
    return out


def encode_wav(samples: Sequence[int], rate: int = CANON_RATE) -> bytes:
    """Encode mono int16 samples to a WAV container."""
    if rate <= 0:
        raise AudioError("sample rate must be positive")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        arr = samples if isinstance(samples, array) else array("h", samples)
        w.writeframes(arr.tobytes())
    return buf.getvalue()


def _clip16(v: int) -> int:
    return -32768 if v < -32768 else (32767 if v > 32767 else v)


# --------------------------------------------------------------------------
# Cropping - the single source of truth for selection maths
# --------------------------------------------------------------------------

def seconds_to_index(t: float, rate: int, total: int) -> int:
    """Map a time in seconds to a sample index, clamped into range.

    The client and the server both call this (the client via the mirrored
    JS implementation) so a crop chosen on screen lands on exactly the
    same sample boundary on the server.
    """
    if t <= 0:
        return 0
    idx = int(round(t * rate))
    return total if idx > total else idx


@dataclass(frozen=True)
class Crop:
    start_index: int
    end_index: int
    rate: int

    @property
    def frames(self) -> int:
        return self.end_index - self.start_index

    @property
    def start_seconds(self) -> float:
        return self.start_index / self.rate

    @property
    def end_seconds(self) -> float:
        return self.end_index / self.rate

    @property
    def duration(self) -> float:
        return self.frames / self.rate


def resolve_crop(total: int, rate: int, start_s: float, end_s: float) -> Crop:
    """Turn a (start, end) in seconds into an exact, validated sample range."""
    if total <= 0:
        raise AudioError("source has no samples")
    a = seconds_to_index(start_s, rate, total)
    b = seconds_to_index(end_s, rate, total)
    if b < a:
        a, b = b, a
    min_frames = max(1, int(round(MIN_CROP_SECONDS * rate)))
    if b - a < min_frames:
        # Grow the selection rather than rejecting it: the user is allowed
        # to ask for a tiny fragment, they just cannot ask for nothing.
        b = min(total, a + min_frames)
        a = max(0, b - min_frames)
    if b <= a:
        raise AudioError("crop resolves to an empty range")
    return Crop(a, b, rate)


def apply_crop(samples: Sequence[int], crop: Crop) -> array:
    out = array("h", samples[crop.start_index : crop.end_index])
    return out


# --------------------------------------------------------------------------
# Waveform peaks
# --------------------------------------------------------------------------

def peaks(samples: Sequence[int], buckets: int = 512) -> list[float]:
    """Min/max-free RMS-ish peak envelope in 0..1, one value per bucket.

    Returned length is always exactly `buckets` so the client can map
    bucket i to x = i / buckets without any rounding disagreement.
    """
    n = len(samples)
    if buckets <= 0:
        raise AudioError("buckets must be positive")
    if n == 0:
        return [0.0] * buckets
    out: list[float] = []
    for i in range(buckets):
        lo = (i * n) // buckets
        hi = ((i + 1) * n) // buckets
        if hi <= lo:
            hi = min(n, lo + 1)
        m = 0
        for j in range(lo, hi):
            v = samples[j]
            if v < 0:
                v = -v
            if v > m:
                m = v
        out.append(m / 32768.0)
    return out


# --------------------------------------------------------------------------
# Feature extraction - this is the part that actually listens
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioFeatures:
    duration: float
    rms: float
    peak: float
    dynamic_range: float
    zero_crossing_rate: float
    onset_count: int
    speech_band_ratio: float
    silence_ratio: float
    voiced_estimate: float

    def as_dict(self) -> dict:
        return asdict(self)


def _fft(x: list[complex]) -> list[complex]:
    """Iterative radix-2 FFT. Input length must be a power of two."""
    n = len(x)
    if n & (n - 1):
        raise AudioError("FFT length must be a power of two")
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            x[i], x[j] = x[j], x[i]
    length = 2
    while length <= n:
        ang = -2 * math.pi / length
        wl = cmath.exp(1j * ang)
        for i in range(0, n, length):
            w = 1 + 0j
            half = length >> 1
            for k in range(half):
                u = x[i + k]
                v = x[i + k + half] * w
                x[i + k] = u + v
                x[i + k + half] = u - v
                w *= wl
        length <<= 1
    return x


def _downsample(samples: Sequence[int], rate: int, target: int = 8000) -> tuple[list[float], int]:
    if rate <= target:
        return [s / 32768.0 for s in samples], rate
    step = rate / target
    out = []
    i = 0.0
    n = len(samples)
    while i < n:
        out.append(samples[int(i)] / 32768.0)
        i += step
    return out, target


def audio_features(samples: Sequence[int], rate: int) -> AudioFeatures:
    """Cheap, dependency-free descriptors of what is actually in the crop.

    These are honest signal statistics, not semantic understanding. They
    tell the scorer whether a crop is a long line of clear speech, a
    half-word, a musical sting, or near-silence - which is most of what
    separates an easy clip from a brutal one.
    """
    n = len(samples)
    if n == 0:
        raise AudioError("cannot analyse an empty crop")
    duration = n / rate

    sq = 0
    peak = 0
    crossings = 0
    prev = samples[0]
    for s in samples:
        sq += s * s
        a = -s if s < 0 else s
        if a > peak:
            peak = a
        if (s >= 0) != (prev >= 0):
            crossings += 1
        prev = s
    rms = math.sqrt(sq / n) / 32768.0
    peakf = peak / 32768.0
    zcr = crossings / n

    mono, srate = _downsample(samples, rate)
    frame = 256
    hop = 128
    energies: list[float] = []
    for i in range(0, max(1, len(mono) - frame + 1), hop):
        seg = mono[i : i + frame]
        energies.append(sum(v * v for v in seg) / len(seg))
    if not energies:
        energies = [sum(v * v for v in mono) / max(1, len(mono))]

    emax = max(energies) or 1e-12
    silence_ratio = sum(1 for e in energies if e < emax * 0.02) / len(energies)

    onsets = 0
    for i in range(1, len(energies)):
        if energies[i] > emax * 0.15 and energies[i] > energies[i - 1] * 2.5:
            onsets += 1

    speech_band_ratio = _speech_band_ratio(mono, srate)

    emin = min(e for e in energies) if energies else 0.0
    dynamic_range = math.log10((emax + 1e-12) / (emin + 1e-12))

    # A rough "is someone talking" score: speech sits in 300-3400 Hz, has
    # moderate ZCR, and is not one continuous tone.
    voiced = speech_band_ratio * (1.0 - min(1.0, abs(zcr - 0.08) / 0.25))
    voiced = max(0.0, min(1.0, voiced * (1.0 - silence_ratio * 0.5)))

    return AudioFeatures(
        duration=duration,
        rms=rms,
        peak=peakf,
        dynamic_range=dynamic_range,
        zero_crossing_rate=zcr,
        onset_count=onsets,
        speech_band_ratio=speech_band_ratio,
        silence_ratio=silence_ratio,
        voiced_estimate=voiced,
    )


def _speech_band_ratio(mono: list[float], rate: int) -> float:
    """Fraction of spectral energy in the 300-3400 Hz speech band."""
    size = 1024
    if len(mono) < size:
        mono = mono + [0.0] * (size - len(mono))
    hop = size
    band = 0.0
    total = 0.0
    for start in range(0, len(mono) - size + 1, hop):
        seg = mono[start : start + size]
        # Hann window
        win = [seg[i] * (0.5 - 0.5 * math.cos(2 * math.pi * i / (size - 1))) for i in range(size)]
        spec = _fft([complex(v, 0.0) for v in win])
        for k in range(size // 2):
            mag = abs(spec[k]) ** 2
            freq = k * rate / size
            total += mag
            if 300.0 <= freq <= 3400.0:
                band += mag
    if total <= 0:
        return 0.0
    return band / total
