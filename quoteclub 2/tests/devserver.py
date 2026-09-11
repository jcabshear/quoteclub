"""Run the real app locally with extraction stubbed, for browser QA.

    python -m tests.devserver [port]

Everything except the two network-dependent providers and the yt-dlp
download is the production code path: the same routes, the same WAV
handling, the same crop maths, the same client bundle.
"""

import math
import os
import sys
import tempfile
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TMP = Path(tempfile.mkdtemp(prefix="qc-dev-"))
os.environ.setdefault("QC_DATA_DIR", str(TMP / "data"))
os.environ.setdefault("QC_INSECURE_COOKIES", "1")
os.environ.setdefault("QC_SECRET", "dev-secret-not-for-production")

from server import audioutil as au  # noqa: E402
from server import extract as XX  # noqa: E402
from server import main as M  # noqa: E402


def demo_wav(seconds=8.0, rate=au.CANON_RATE) -> bytes:
    """Speech-ish bursts separated by gaps, so the waveform has shape."""
    n = int(seconds * rate)
    out = array("h", bytes(2 * n))
    bursts = [(0.4, 1.5, 620), (2.0, 2.6, 900), (3.2, 4.9, 480), (5.6, 7.2, 760)]
    for start, end, freq in bursts:
        for i in range(int(start * rate), min(n, int(end * rate))):
            t = i / rate
            env = max(0.0, math.sin(math.pi * (t - start) / (end - start))) ** 0.5
            out[i] = int(
                11000 * env * (math.sin(2 * math.pi * freq * t)
                               + 0.35 * math.sin(2 * math.pi * freq * 2.1 * t))
            )
    return au.encode_wav(out, rate)


STUB = TMP / "stub.wav"
STUB.write_bytes(demo_wav())


async def fake_extract(url, phrase="", workdir=None):
    return XX.Extracted(STUB, 8.0, au.CANON_RATE, "The Incredibles", None, [])


async def fake_yarn(http, phrase, limit=10):
    return [
        M.Candidate(
            url="https://y.yarn.co/11111111-2222-3333-4444-555555555555.mp4",
            provider="subtitle_index",
            transcript=phrase,
            source_title="The Incredibles",
            duration=8.0,
        )
    ]


async def fake_web(http, phrase, count=20):
    return [M.Candidate(url="https://example.com/scene.mp4", provider="video",
                        title="scene", duration=8.0)]


M.X.extract = fake_extract
M.yarn.search = fake_yarn
M.websearch.search = fake_web
M.gate = lambda: (lambda url: "Stub")

app = M.app

if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
