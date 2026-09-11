"""Difficulty and point suggestion.

Two honest statements about what this is:

* It DOES listen to the crop. The features come from `audioutil.audio_features`
  (duration, loudness, silence, onsets, speech-band energy, a voiced-speech
  estimate), so a 0.3 s musical sting and a 4 s clean line of dialogue are
  not treated the same way just because they came from the same search.
* It does NOT understand the crop. Nothing here recognises a voice, reads
  words, or knows that a line is famous. Familiarity comes from *your own
  play history* - which sources you two have sent and guessed before - plus
  the corrections you make to its suggestions.

The model is a small online linear regressor over interpretable features,
trained only on the pair's own accepted/adjusted values. It starts from a
hand-set prior so the first few clips are sensible, then moves towards
whatever this particular pair of brothers thinks is hard.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

MIN_POINTS = 1
MAX_POINTS = 100

LABELS = (
    (0, 15, "warm-up"),
    (15, 35, "fair"),
    (35, 60, "tricky"),
    (60, 85, "hard"),
    (85, 101, "brutal"),
)


def label_for(points: int) -> str:
    for lo, hi, name in LABELS:
        if lo <= points < hi:
            return name
    return "brutal"


@dataclass
class Features:
    """Interpretable inputs. Keep names stable - weights are persisted."""

    log_duration: float = 0.0       # ln(seconds), shorter -> harder
    voiced: float = 0.0             # 0..1, clear speech -> easier
    silence: float = 0.0            # 0..1 of the crop that is near-silent
    onsets: float = 0.0             # distinct sound events, normalised
    speech_band: float = 0.0        # 0..1 energy in 300-3400 Hz
    quietness: float = 0.0          # 1 - rms, quiet -> harder
    source_familiarity: float = 0.0 # 0..1 from play history
    source_unseen: float = 0.0      # 1.0 when we have never played this source
    word_fraction: float = 0.0      # 0..1 how much of the searched phrase fits

    ORDER = (
        "log_duration", "voiced", "silence", "onsets", "speech_band",
        "quietness", "source_familiarity", "source_unseen", "word_fraction",
    )

    def vector(self) -> list[float]:
        return [getattr(self, k) for k in self.ORDER]


# Prior weights, in "points" units, applied to the feature vector plus a bias.
# Signs are the interesting part: longer, clearly-voiced, familiar-source,
# full-phrase clips are easier (negative weight on difficulty).
PRIOR_BIAS = 52.0
PRIOR_WEIGHTS = {
    "log_duration": -11.0,
    "voiced": -16.0,
    "silence": 14.0,
    "onsets": -5.0,
    "speech_band": -8.0,
    "quietness": 12.0,
    "source_familiarity": -20.0,
    "source_unseen": 10.0,
    "word_fraction": -14.0,
}

LEARNING_RATE = 0.06
L2 = 0.002


@dataclass
class Model:
    bias: float = PRIOR_BIAS
    weights: dict[str, float] = field(
        default_factory=lambda: dict(PRIOR_WEIGHTS)
    )
    samples: int = 0

    # -- persistence -------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(
            {"bias": self.bias, "weights": self.weights, "samples": self.samples}
        )

    @classmethod
    def from_json(cls, raw: Optional[str]) -> "Model":
        if not raw:
            return cls()
        try:
            d = json.loads(raw)
        except (TypeError, ValueError):
            return cls()
        m = cls(
            bias=float(d.get("bias", PRIOR_BIAS)),
            weights=dict(PRIOR_WEIGHTS),
            samples=int(d.get("samples", 0)),
        )
        for k, v in (d.get("weights") or {}).items():
            if k in m.weights:
                try:
                    m.weights[k] = float(v)
                except (TypeError, ValueError):
                    pass
        return m

    # -- inference ---------------------------------------------------
    def raw_score(self, f: Features) -> float:
        total = self.bias
        for name in Features.ORDER:
            total += self.weights.get(name, 0.0) * getattr(f, name)
        return total

    def suggest(self, f: Features) -> int:
        return clamp_points(self.raw_score(f))

    # -- learning ----------------------------------------------------
    def learn(self, f: Features, chosen_points: int) -> None:
        """Nudge towards the value the sender actually chose.

        One gradient step of squared error with light L2 pull back to the
        prior. Deliberately slow: a pair of players generates tens of
        examples, not thousands, and a single weird clip should not
        rewrite the scale.
        """
        target = float(clamp_points(chosen_points))
        pred = self.raw_score(f)
        err = pred - target
        scale = LEARNING_RATE / (1.0 + 0.02 * self.samples)
        self.bias -= scale * err
        for name in Features.ORDER:
            x = getattr(f, name)
            g = err * x + L2 * (self.weights[name] - PRIOR_WEIGHTS[name])
            self.weights[name] -= scale * g
        self.samples += 1


def clamp_points(v: float) -> int:
    if v != v:  # NaN
        return 25
    p = int(round(v))
    return MIN_POINTS if p < MIN_POINTS else (MAX_POINTS if p > MAX_POINTS else p)


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

def build_features(
    af,
    source_title: Optional[str],
    history: dict[str, int] | None = None,
    searched_phrase: str = "",
    transcript_in_crop: str = "",
) -> Features:
    """Turn measured audio + play history into the model's inputs.

    `af` is an `audioutil.AudioFeatures`. `history` maps a normalised source
    title to how many times this pair has already played it.
    """
    history = history or {}
    key = normalise_title(source_title or "")
    seen = history.get(key, 0)
    familiarity = 1.0 - math.exp(-seen / 2.5) if seen else 0.0

    # How much of the phrase the sender typed actually survives in the crop.
    # If the provider gave us a transcript for the selection we use it;
    # otherwise we fall back to a duration-based guess, because a 0.2 s crop
    # cannot contain a nine-word sentence.
    if transcript_in_crop and searched_phrase:
        from .discovery import phrase_containment
        word_fraction = phrase_containment(searched_phrase, transcript_in_crop)
    elif searched_phrase:
        expected = max(1, len(searched_phrase.split()))
        # ~0.32 s per spoken word is a reasonable conversational rate.
        fits = af.duration / 0.32
        word_fraction = max(0.0, min(1.0, fits / expected))
    else:
        word_fraction = 0.0

    return Features(
        log_duration=math.log(max(af.duration, 0.02)),
        voiced=af.voiced_estimate,
        silence=af.silence_ratio,
        onsets=min(1.0, af.onset_count / 6.0),
        speech_band=af.speech_band_ratio,
        quietness=max(0.0, 1.0 - af.rms * 6.0),
        source_familiarity=familiarity,
        source_unseen=0.0 if seen else 1.0,
        word_fraction=word_fraction,
    )


def normalise_title(title: str) -> str:
    import re

    t = (title or "").lower().strip()
    t = re.sub(r"\b(the|a|an)\b", " ", t)
    t = re.sub(r"\(\d{4}\)", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


# --------------------------------------------------------------------------
# Hint decay
# --------------------------------------------------------------------------

HINT_DECAY = 0.75


def reward_after_hints(base_points: int, hints_delivered: int) -> int:
    """Points still on the table after N *delivered* hints.

    Requesting a hint costs nothing; the deduction lands when the extra
    audio actually arrives.
    """
    if hints_delivered <= 0:
        return clamp_points(base_points)
    value = base_points * (HINT_DECAY ** hints_delivered)
    return max(MIN_POINTS, int(round(value)))
