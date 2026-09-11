# Quote Club

A party game of guessing films and shows from slivers of their audio.
Someone types a quote, the app finds the original sound, they crop it as
tight as they dare, and everyone else in the room guesses.

**One deployable service** — API, client, audio extraction and storage in a
single container. **No accounts**: a six-letter room code is the whole of
access control, so any number of unrelated groups can play at once and a
person can sit in several rooms at the same time.

---

## What is actually verified, and what is not

Read this before believing anything below it.

### Verified by tests that ran

165 unit/integration tests (`python -m unittest discover -s tests`) and 42
browser checks in real Chromium (`python -m tests.browser_qa`).

| Area | Evidence |
|---|---|
| Crop exactness | `test_audioutil.py`, `test_pipeline.py` — a real ffmpeg transcode is cropped, and the bytes served over HTTP are compared sample-by-sample with the selection. Identical. |
| ffmpeg pipeline | `test_pipeline.py::FfmpegPipeline` — real AAC and H.264+AAC files synthesised, transcoded, probed, windowed, verified. |
| "It decoded" ≠ "it has audio" | A silent AAC file decodes fine and is **rejected** by `extract.verify`. |
| Discovery policy | `test_discovery.py` — `https://coub.com/view/5cru0` is admitted; a 404 on one Coub clip does not remove coub.com from the search. |
| Multi-room isolation | `test_pipeline.py` — two rooms running at once never see each other's rounds or answers; a stranger with no code gets 403; a wrong code gets 404; clip audio is unreachable cross-room. |
| No login | A brand-new browser creates a room and plays with no account of any kind. |
| Rooms of 2–12 | Turn rotates round-robin; every non-sender guesses **independently** and banks points independently; the room fills at 12. |
| Hints | Asking is free; delivery costs 25% for everyone *still guessing* and never claws back points already banked. Floor of 1. |
| Answer privacy | The guesser's view is scanned for the answer, transcript, search phrase, source URL, page title and filename — none appear. One player solving does not reveal it to the others, and another player's wrong guess is never shown. |
| Scoring reads the audio | Two crops with identical metadata but different content get different suggestions; corrections move future suggestions. |
| Mobile layout | Chromium at 320/375/390/430 px, every view, no element past the viewport edge. |
| Handle alignment | Handle centres measured against the rendered selection rectangle: agreement within 1 device pixel, mid-drag and at the 20 ms floor. |
| Preview playback | The generated blob loads in a real media engine, its duration equals the selection, `currentTime` advances, changing the crop cancels it. |
| Two real browsers | Player A starts a room and sends a clip; player B joins from the shared link with no sign-in, gets audio with no answer visible, guesses correctly, and appears on the scoreboard. |

### NOT verified — do not assume these work

1. **Audible sound on an iPhone.** The playback path is the one built to
   survive iOS (a long-lived `<audio>` element, a blob built synchronously
   inside the tap handler, no `AudioContext` anywhere) and it plays in
   Chromium. Only a real phone settles the last step.
2. **Yarn's search endpoint.** The clip media URL shape
   (`https://y.yarn.co/<uuid>.mp4`) is confirmed from a third-party ripper,
   and the parser handles JSON and HTML defensively. The *search* response
   shape has never been observed. Run this once from a networked machine:
   ```bash
   python -m server.providers.yarn "you were this close to losing your job"
   ```
   It prints the status, content type, first 1500 bytes, and what the parser
   made of them. Adjust `parse_search` to match.
3. **Real-world search recall.** No live extraction ran during development —
   the build environment had no outbound network. The discovery *policy* is
   tested; how much of the internet it reaches is unmeasured.
4. **The Incredibles clip specifically.** `https://coub.com/view/5cru0` is
   real, its page serves a direct MP3, and yt-dlp's Coub extractor matches
   the URL. Three good reasons to expect it to work; zero proof that it does.

---

## Why the old search failed, concretely

Two independent defects, both fixed, both covered by regression tests.

**1. The media filter judged URLs by the words in their path** — `watch`,
`clip`, `sound`, `video`. `coub.com/view/5cru0` contains none of them, so an
indexed result with the exact quote in its title was discarded before
extraction was ever attempted.

Stop guessing. `discovery.build_extractor_gate()` asks yt-dlp's own registry:

```python
classes = [ie for ie in gen_extractor_classes() if ie.ie_key() != "Generic"]
```

A URL is admitted if a real extractor claims it, **or** it ends in a media
extension (`.m3u8` / `.mpd` included), **or** the result advertised media.
Everything else is rejected *with a reason that reaches the UI*.

**2. One failed clip deleted a whole website.** A 422 added the hostname to
`failed_hosts`, and every other candidate from that host was skipped.

`discovery.FailureMemory` classifies failures — `URL_SPECIFIC` (404, no audio
stream), `TRANSIENT` (timeout, 5xx, 429), `HOST_BLOCKED` (login wall, geo
block) — and then: a failed URL is not retried; a host accumulating distinct
failures is *demoted in ranking*, not banned; only a genuine host-level
signal suspends a host, for 30 minutes; one success clears the damage.

### Other limits that were self-inflicted

* **HLS/DASH.** yt-dlp hands segmented streams to ffmpeg, which concatenates
  them. There was never a reason to refuse `.m3u8`.
* **"Longer than three minutes without captions → reject."** Now captions are
  used to *seek* when they exist, and otherwise a bounded window is fetched
  and you scrub it. Only `MAX_SOURCE_SECONDS` (default 15 min) refuses a source.
* **Search as one blocking call.** It is a streaming job now. You watch each
  candidate get admitted, fetched, verified or fail, with the reason.

---

## How a game works

* Anyone starts a room and gets a six-letter code (`SUPWJY`, shown as
  `SUP-WJY`). Sharing the code — or the `/r/SUPWJY` link — is the invitation.
* 2 to 12 seats. Turn order is join order, rotating.
* The sender is the one person who cannot guess their own clip, and **the
  sender never scores**. Making a brutal clip just puts more on the table.
* Everyone else guesses independently. One person solving does not end the
  round for anybody else, and does not reveal the answer to them.
* Any guesser can ask for a hint. Asking is free. When the sender supplies
  the extra audio, the reward drops 25% for everyone still guessing —
  points already banked are safe.
* The round closes when everyone has solved or given up, or when the sender
  ends it early. Then the answer goes up for the room and the turn moves on.
* If the title matcher rejects a guess that should have counted, the guesser
  disputes it and the sender decides.

---

## Architecture

```
server/
  audioutil.py    WAV decode/encode, sample-exact crop, peaks, FFT features
  discovery.py    extractor gate, failure memory, ranking  ← the fixed part
  extract.py      yt-dlp + ffmpeg, caption seeking, verification
  scoring.py      audio-derived difficulty + online learning from corrections
  game.py         rooms, turn rotation, hints, disputes, answer-gated views
  store.py        sqlite + clip files + idle-room housekeeping
  main.py         Starlette app: API + static client
  providers/
    yarn.py       subtitle-indexed corpus (primary)
    websearch.py  Brave (widener, not primary)
web/              index.html + app.js + styles.css, no build step
```

### The one idea that matters most

**Search a subtitle index, not the web.** A web search answers "what pages
mention this string" — which is why it returned Pinterest boards. A subtitle
index answers "where was this line spoken", and hands back the source title
with it. That fixes discovery recall *and* answer autofill, which previously
depended on scraping movie names out of video titles. Providers are ordered
accordingly (`PROVIDER_PRIOR` in `discovery.py`); adding another corpus means
writing one function that returns `Candidate`s.

### Crop exactness

Sample index is the only truth. `audioutil.seconds_to_index` and the mirrored
`idx()` in `app.js` round identically, the SVG uses a `0..1000` viewBox across
the same box the handles are positioned in, and the client slices the same
`Int16Array` the server will slice. Waveform, selection, handles, readout,
preview and exported file are structurally incapable of disagreeing. Minimum
crop is 20 ms; a degenerate selection is grown, never rejected, so half-words
and single sounds are all fair game.

### Difficulty

`scoring.py` measures the crop — duration, RMS, silence ratio, onset count,
speech-band (300–3400 Hz) energy share, a voiced-speech estimate — plus how
much of the searched phrase could physically fit in it, plus how often the
room has played that source. A small online linear model turns those into
points and learns from every correction. It does not recognise voices or know
that a line is famous; the UI states what it heard so you can see when it is
wrong. Range 1–100; hint decay `round(points × 0.75ⁿ)`, floor 1.

---

## Running it

Local:

```bash
pip install -r requirements.txt        # ffmpeg must be on PATH
python -m uvicorn server.main:app --reload --port 8080
```

Browser QA against a stubbed extractor (no network needed):

```bash
python -m tests.devserver 8099         # then open http://127.0.0.1:8099
python -m tests.browser_qa             # needs playwright + chromium
```

Deploy — anything that builds a Dockerfile. On Railway: attach the repo, mount
a volume at `/data`, and set:

| Variable | Why |
|---|---|
| `QC_SECRET` | signs the player cookie. **Set it**, or everyone is logged out on each restart. |
| `QC_DATA_DIR` | `/data`, on a mounted volume, or rooms vanish on redeploy. |
| `BRAVE_API_KEY` | optional. Without it the web-search widener is skipped and only the subtitle index runs. |
| `QC_ROOM_TTL_DAYS` | idle rooms and their audio are purged after this (default 45). |
| `MAX_SOURCE_SECONDS` | default 900. |
| `QC_TARGET_RESULTS` | verified results to stop at, default 4. |

Check `GET /api/health` after deploying — it reports whether ffmpeg, ffprobe
and yt-dlp are present, plus room/round/clip counts. A green deploy with
`yt_dlp: false` will find nothing, and the UI says so rather than blaming the
internet.
