/* Quote Club client.
 *
 * Three things in here are deliberate and worth not "simplifying" later:
 *
 * 1. PLAYBACK NEVER USES AudioContext. The server always sends canonical
 *    16-bit mono PCM WAV, so the client parses it directly, slices the
 *    exact sample range, and plays a Blob through one long-lived <audio>
 *    element. On iOS an AudioContext can be created suspended and never
 *    resume, failing silently; an <audio> element whose .play() is called
 *    synchronously inside a tap handler is the reliable path. Everything
 *    needed to build the Blob is in memory before the tap, so nothing
 *    async sits between the gesture and .play().
 *
 * 2. ONE COORDINATE SYSTEM. Sample index is the truth. Seconds, pixels,
 *    the SVG viewBox and the handle offsets all derive from it with the
 *    same rounding the server uses (see idx()). Waveform, selection,
 *    handles, readout, preview and exported file cannot disagree.
 *
 * 3. NO ACCOUNTS. The browser holds a signed random player id in a
 *    cookie; a room code is the only thing that grants access to a room.
 */

const $ = (sel) => document.querySelector(sel);
const api = async (path, opts = {}) => {
  const r = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    const err = new Error(detail);
    err.status = r.status;
    throw err;
  }
  return r.status === 204 ? null : r.json();
};

const state = {
  me: null,
  code: null,      // normalised room code, or null when in the lobby
  room: null,
  rounds: [],
  scoreboard: [],
  source: null,    // {token, pcm, rate, frames, duration}
  sel: { a: 0, b: 0 },
  suggested: null,
  features: null,
  clip: null,
  es: null,
  hintFor: null,
  poll: 0,
  view: "play",
};

const normCode = (s) => (s || "").toUpperCase().replace(/[^A-Z]/g, "");
const pretty = (c) => (c && c.length === 6 ? c.slice(0, 3) + "-" + c.slice(3) : c || "");
const escapeHtml = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function show(el, text, cls) {
  if (!el) return;
  el.textContent = text;
  el.className = "msg" + (el.classList.contains("inline") ? " inline" : "") +
    (cls ? " " + cls : "");
  el.hidden = !text;
}

/* ------------------------------------------------------------------ WAV */

function parseWav(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, false) !== 0x52494646 || dv.getUint32(8, false) !== 0x57415645) {
    throw new Error("not a WAV");
  }
  let off = 12, rate = 44100, channels = 1, bits = 16, data = null;
  while (off + 8 <= dv.byteLength) {
    const id = dv.getUint32(off, false);
    const size = dv.getUint32(off + 4, true);
    const body = off + 8;
    if (id === 0x666d7420) {
      channels = dv.getUint16(body + 2, true);
      rate = dv.getUint32(body + 4, true);
      bits = dv.getUint16(body + 14, true);
    } else if (id === 0x64617461) {
      data = new Int16Array(buf.slice(body, body + size));
    }
    off = body + size + (size % 2);
  }
  if (!data) throw new Error("WAV has no data chunk");
  if (bits !== 16) throw new Error("expected 16-bit PCM");
  if (channels !== 1) {
    const mono = new Int16Array(Math.floor(data.length / channels));
    for (let i = 0; i < mono.length; i++) {
      let s = 0;
      for (let c = 0; c < channels; c++) s += data[i * channels + c];
      mono[i] = (s / channels) | 0;
    }
    data = mono;
  }
  return { pcm: data, rate };
}

function buildWav(pcm, rate) {
  const bytes = pcm.length * 2;
  const buf = new ArrayBuffer(44 + bytes);
  const dv = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); dv.setUint32(4, 36 + bytes, true); str(8, "WAVE");
  str(12, "fmt "); dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, rate, true); dv.setUint32(28, rate * 2, true);
  dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  str(36, "data"); dv.setUint32(40, bytes, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], { type: "audio/wav" });
}

/* --------------------------------------------------- selection maths */

// Mirrors server audioutil.seconds_to_index exactly.
const idx = (t, rate, total) => (t <= 0 ? 0 : Math.min(total, Math.round(t * rate)));
const MIN_CROP_SECONDS = 0.02;

function clampSel(a, b) {
  const { rate, frames } = state.source;
  let lo = Math.max(0, Math.min(frames, Math.round(a)));
  let hi = Math.max(0, Math.min(frames, Math.round(b)));
  if (hi < lo) [lo, hi] = [hi, lo];
  const min = Math.max(1, Math.round(MIN_CROP_SECONDS * rate));
  if (hi - lo < min) {
    hi = Math.min(frames, lo + min);
    lo = Math.max(0, hi - min);
  }
  return { a: lo, b: hi };
}

/* ------------------------------------------------------------ waveform */

function renderWave(peaks) {
  const g = $("#wave-bars");
  const n = peaks.length;
  const w = 1000 / n;
  // No rounded corners: preserveAspectRatio="none" scales x and y by
  // different factors, so any rx turns thin bars into ellipses.
  let html = "";
  for (let i = 0; i < n; i++) {
    const h = Math.max(1.5, peaks[i] * 92);
    html += `<rect x="${(i * w).toFixed(3)}" y="${((100 - h) / 2).toFixed(3)}" ` +
            `width="${(w * 0.82).toFixed(3)}" height="${h.toFixed(3)}"/>`;
  }
  g.innerHTML = html;
}

function bucketCount() {
  const w = $("#wave-wrap").getBoundingClientRect().width || 360;
  return Math.max(200, Math.min(1200, Math.round(w * 1.6)));
}

function paintSelection() {
  const { frames, rate } = state.source;
  const { a, b } = state.sel;
  const pa = frames ? a / frames : 0;
  const pb = frames ? b / frames : 0;
  $("#sel-rect").setAttribute("x", (pa * 1000).toFixed(3));
  $("#sel-rect").setAttribute("width", Math.max(0, (pb - pa) * 1000).toFixed(3));
  $("#h-start").style.left = (pa * 100).toFixed(4) + "%";
  $("#h-end").style.left = (pb * 100).toFixed(4) + "%";
  $("#h-start").setAttribute("aria-valuenow", (a / rate).toFixed(3));
  $("#h-end").setAttribute("aria-valuenow", (b / rate).toFixed(3));
  $("#sel-readout").textContent =
    `${(a / rate).toFixed(3)} – ${(b / rate).toFixed(3)} s  ·  ${((b - a) / rate).toFixed(3)}s`;
  stopPreview();
  scheduleSuggest();
}

/* ------------------------------------------------------------- dragging */

function bindDrag() {
  const wrap = $("#wave-wrap");
  const posToFrame = (clientX) => {
    const r = wrap.getBoundingClientRect();
    const p = r.width ? (clientX - r.left) / r.width : 0;
    return Math.max(0, Math.min(1, p)) * state.source.frames;
  };
  let active = null;

  const startDrag = (which) => (ev) => {
    if (!state.source) return;
    ev.preventDefault();
    active = which;
    ev.currentTarget.setPointerCapture?.(ev.pointerId);
  };
  $("#h-start").addEventListener("pointerdown", startDrag("a"));
  $("#h-end").addEventListener("pointerdown", startDrag("b"));

  const apply = (f) => {
    if (active === "a") state.sel = clampSel(f, state.sel.b);
    else state.sel = clampSel(state.sel.a, f);
    paintSelection();
  };

  wrap.addEventListener("pointerdown", (ev) => {
    if (!state.source || active) return;
    if (ev.target.closest(".handle")) return;
    const f = posToFrame(ev.clientX);
    active = Math.abs(f - state.sel.a) <= Math.abs(f - state.sel.b) ? "a" : "b";
    wrap.setPointerCapture?.(ev.pointerId);
    apply(f);
  });

  const move = (ev) => { if (active && state.source) apply(posToFrame(ev.clientX)); };
  const end = () => { active = null; };
  window.addEventListener("pointermove", move, { passive: true });
  window.addEventListener("pointerup", end);
  window.addEventListener("pointercancel", end);

  const keyStep = (which) => (ev) => {
    if (!state.source) return;
    const step = (ev.shiftKey ? 0.5 : 0.02) * state.source.rate;
    let d = 0;
    if (ev.key === "ArrowLeft") d = -step;
    else if (ev.key === "ArrowRight") d = step;
    else return;
    ev.preventDefault();
    if (which === "a") state.sel = clampSel(state.sel.a + d, state.sel.b);
    else state.sel = clampSel(state.sel.a, state.sel.b + d);
    paintSelection();
  };
  $("#h-start").addEventListener("keydown", keyStep("a"));
  $("#h-end").addEventListener("keydown", keyStep("b"));

  document.querySelectorAll("[data-nudge]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!state.source) return;
      const [which, secs] = btn.dataset.nudge.split(",");
      const d = parseFloat(secs) * state.source.rate;
      if (which === "start") state.sel = clampSel(state.sel.a + d, state.sel.b);
      else state.sel = clampSel(state.sel.a, state.sel.b + d);
      paintSelection();
    });
  });
}

/* -------------------------------------------------------------- preview */

const player = new Audio();
player.preload = "auto";
// Exposed so automated browser QA can measure what is actually loaded.
window.__qcPlayer = player;
let previewUrl = null;
let rafId = 0;

function stopPreview() {
  if (!player.paused) player.pause();
  player.removeAttribute("src");
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
  cancelAnimationFrame(rafId);
  $("#playhead")?.setAttribute("hidden", "");
}

function playSelection() {
  const msg = $("#preview-msg");
  msg.hidden = true;
  if (!state.source) return;
  const { pcm, rate, frames } = state.source;
  const { a, b } = state.sel;
  // Synchronous: slice -> blob -> src -> play(), all inside the tap.
  const slice = pcm.subarray(a, b);
  if (!slice.length) { show(msg, "Nothing selected.", "err"); return; }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(buildWav(slice, rate));
  player.src = previewUrl;
  const p = player.play();
  if (p && p.catch) {
    p.catch((e) => show(msg, "Could not play: " + (e.message || e.name), "err"));
  }
  const head = $("#playhead");
  head.removeAttribute("hidden");
  const tick = () => {
    const t = a + player.currentTime * rate;
    const x = frames ? (t / frames) * 1000 : 0;
    head.setAttribute("x1", x.toFixed(2));
    head.setAttribute("x2", x.toFixed(2));
    if (!player.paused && !player.ended) rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
}
player.addEventListener("ended", () => {
  cancelAnimationFrame(rafId);
  $("#playhead")?.setAttribute("hidden", "");
});
player.addEventListener("error", () => {
  show($("#preview-msg"), "The audio element rejected this clip.", "err");
});

/* --------------------------------------------------------------- search */

function logLine(text, cls) {
  const li = document.createElement("li");
  li.textContent = text;
  if (cls) li.className = cls;
  $("#progress-log").append(li);
  $("#progress-log").scrollTop = $("#progress-log").scrollHeight;
}

async function runSearch() {
  const phrase = $("#q").value.trim();
  const url = $("#q-url").value.trim();
  if (!phrase && !url) return;
  $("#results").innerHTML = "";
  $("#empty").hidden = true;
  $("#editor").hidden = true;
  $("#progress-log").innerHTML = "";
  $("#progress").hidden = false;
  $("#progress-phase").textContent = "Searching…";
  $("#search").disabled = true;
  state.es?.close();

  let job;
  try {
    job = await api(`/api/rooms/${state.code}/search`, {
      method: "POST",
      body: JSON.stringify({ phrase, urls: url ? [url] : [] }),
    });
  } catch (e) {
    $("#search").disabled = false;
    show($("#empty"), e.message, "err");
    return;
  }

  const es = new EventSource(`/api/search/${job.job_id}/stream`);
  state.es = es;
  es.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    switch (d.kind) {
      case "phase": $("#progress-phase").textContent = d.phase; break;
      case "provider":
        logLine(d.error ? `${d.provider}: ${d.error}`
                        : `${d.provider}: ${d.found} candidate(s)`,
                d.error ? "bad" : null);
        break;
      case "plan":
        logLine(`${d.considered} found, trying ${d.attempting}`);
        (d.rejected || []).forEach((r) => logLine(`skipped — ${r.reason}`, "bad"));
        break;
      case "attempt":
        $("#progress-phase").textContent = "Fetching audio…";
        logLine(`trying a ${d.provider === "subtitle_index" ? "subtitle-index" : d.provider} result`);
        break;
      case "failed": logLine(`no audio — ${d.reason}`, "bad"); break;
      case "result": addResult(d); logLine("got playable audio", "ok"); break;
      case "warning": logLine(d.message, "bad"); break;
      case "empty":
      case "error": show($("#empty"), d.message, "err"); break;
      case "done":
        es.close();
        $("#progress-phase").textContent = "Done";
        $("#search").disabled = false;
        break;
    }
  };
  es.onerror = () => { es.close(); $("#search").disabled = false; };
}

function addResult(d) {
  const btn = document.createElement("button");
  btn.className = "result";
  btn.type = "button";
  const label = d.suggested_answer
    ? `<strong>${escapeHtml(d.suggested_answer)}</strong>`
    : "<strong>Unlabelled clip</strong>";
  btn.innerHTML =
    `<span>${label}<br><span class="meta">${d.duration.toFixed(1)}s · ${escapeHtml(d.provider_label)}` +
    `${(d.notes || []).length ? " · " + escapeHtml(d.notes.join("; ")) : ""}</span></span>` +
    `<span class="meta">open ›</span>`;
  btn.addEventListener("click", () => loadSource(d, btn));
  $("#results").append(btn);
}

async function loadSource(d, btn) {
  document.querySelectorAll(".result").forEach((b) => b.classList.remove("is-active"));
  btn?.classList.add("is-active");
  const [wavResp, peaksResp] = await Promise.all([
    fetch(`/api/preview/${d.token}.wav`, { credentials: "same-origin" }),
    api(`/api/preview/${d.token}/peaks?buckets=${bucketCount()}`),
  ]);
  const { pcm, rate } = parseWav(await wavResp.arrayBuffer());
  state.source = { token: d.token, pcm, rate, frames: pcm.length, duration: pcm.length / rate };
  state.sel = clampSel(0, idx(Math.min(3, pcm.length / rate), rate, pcm.length));
  renderWave(peaksResp.peaks);
  $("#answer").value = d.suggested_answer || "";
  $("#points").dataset.touched = "";
  $("#editor").hidden = false;
  paintSelection();
  $("#editor").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ------------------------------------------------------- crop + suggest */

let suggestTimer = 0;
function scheduleSuggest() {
  clearTimeout(suggestTimer);
  suggestTimer = setTimeout(makeClip, 350);
}

async function makeClip() {
  if (!state.source || !state.code) return;
  const { rate } = state.source;
  try {
    const r = await api(`/api/rooms/${state.code}/clips`, {
      method: "POST",
      body: JSON.stringify({
        token: state.source.token,
        start: state.sel.a / rate,
        end: state.sel.b / rate,
      }),
    });
    state.clip = r;
    state.features = r.features;
    state.suggested = r.suggested_points;
    if (!$("#points").dataset.touched) {
      $("#points").value = r.suggested_points;
      $("#points-val").textContent = r.suggested_points;
      $("#points-label").textContent = r.difficulty;
    }
    const h = r.heard;
    $("#points-why").textContent =
      `Heard: ${r.duration.toFixed(2)}s, ${Math.round(h.speech_band * 100)}% speech-band energy, ` +
      `${Math.round(h.silence * 100)}% silence, ${h.events} distinct sound${h.events === 1 ? "" : "s"}.` +
      ` Move the slider if that is wrong — it learns from your corrections.`;
    if (!$("#answer").value && r.suggested_answer) $("#answer").value = r.suggested_answer;
  } catch (e) {
    show($("#send-msg"), e.message, "err");
  }
}

async function sendClip() {
  const msg = $("#send-msg");
  if (!state.clip) { show(msg, "Crop something first.", "err"); return; }
  const answer = $("#answer").value.trim();
  const points = parseInt($("#points").value, 10);
  try {
    if (state.hintFor) {
      await api(`/api/rounds/${state.hintFor}/hint`, {
        method: "POST", body: JSON.stringify({ clip_id: state.clip.clip_id }),
      });
      state.hintFor = null;
      show(msg, "Hint sent. Everyone still guessing dropped 25%.", "ok");
    } else {
      if (!answer) { show(msg, "Fill in the answer — nobody else sees it.", "err"); return; }
      await api(`/api/rooms/${state.code}/rounds`, {
        method: "POST",
        body: JSON.stringify({
          clip_id: state.clip.clip_id, answer, points,
          features: state.features, suggested_points: state.suggested,
        }),
      });
      show(msg, "Sent.", "ok");
    }
    resetEditor();
    await refresh();
    setView("play");
  } catch (e) {
    show(msg, e.message, "err");
  }
}

function resetEditor() {
  stopPreview();
  state.source = null;
  state.clip = null;
  state.features = null;
  $("#editor").hidden = true;
  $("#results").innerHTML = "";
  $("#progress").hidden = true;
  $("#q").value = "";
  $("#q-url").value = "";
}

/* ----------------------------------------------------------------- room */

function setView(name) {
  state.view = name;
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("is-active", t.dataset.view === name));
  ["make", "play", "score"].forEach((v) => { $("#view-" + v).hidden = v !== name; });
}

function showLobby() {
  clearInterval(state.poll);
  state.poll = 0;
  state.code = null;
  state.room = null;
  resetEditor();
  $("#view-lobby").hidden = false;
  $("#roombar").hidden = true;
  $("#tabs").hidden = true;
  ["make", "play", "score"].forEach((v) => { $("#view-" + v).hidden = true; });
  history.replaceState(null, "", "/");
  loadMe();
}

async function loadMe() {
  const d = await api("/api/me");
  state.me = d.player;
  if (d.player && !$("#me-name").value) $("#me-name").value = d.player.name;
  const host = $("#my-rooms-list");
  if (!d.rooms.length) { $("#my-rooms").hidden = true; return; }
  $("#my-rooms").hidden = false;
  host.innerHTML = "";
  d.rooms.forEach((r) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "result";
    b.innerHTML =
      `<span><strong>${escapeHtml(r.pretty_code)}</strong><br>` +
      `<span class="meta">${r.players} player${r.players === 1 ? "" : "s"}` +
      `${r.your_turn ? " · your turn" : ""}</span></span><span class="meta">open ›</span>`;
    b.addEventListener("click", () => enterRoom(r.code));
    host.append(b);
  });
}

async function enterRoom(code) {
  state.code = normCode(code);
  $("#view-lobby").hidden = true;
  $("#roombar").hidden = false;
  $("#tabs").hidden = false;
  history.replaceState(null, "", "/r/" + state.code);
  setView("play");
  await refresh();
  clearInterval(state.poll);
  state.poll = setInterval(() => {
    if (document.visibilityState === "visible" && !state.es) refresh().catch(() => {});
  }, 5000);
}

async function refresh() {
  if (!state.code) return;
  let s;
  try {
    s = await api(`/api/rooms/${state.code}/state`);
  } catch (e) {
    if (e.status === 401 || e.status === 403 || e.status === 404) { showLobby(); return; }
    throw e;
  }
  state.me = s.you;
  state.room = s.room;
  state.rounds = s.rounds;
  state.scoreboard = s.scoreboard;

  $("#room-code").textContent = s.room.pretty_code;
  $("#roster").textContent =
    s.room.players.map((p) => p.name + (p.on_turn ? " ●" : "")).join(", ");
  const open = s.rounds.find((r) => r.state === "open");
  let note = "";
  if (s.room.players.length < 2) {
    note = "Share the code — you need at least one more player.";
  } else if (open) {
    note = open.role === "sender"
      ? "Your clip is in play."
      : `${open.sender_name} sent a clip. Guess it on the Play tab.`;
  } else if (s.room.your_turn) {
    note = "Your turn to send a clip.";
  } else {
    note = `Waiting for ${s.room.turn_name} to send one.`;
  }
  show($("#turn-note"), note, s.room.can_send ? "ok" : null);

  renderRounds();
  renderScore();
  if (!s.tools.ffmpeg || !s.tools.yt_dlp) {
    show($("#empty"),
      "This server is missing " +
      Object.entries(s.tools).filter(([, v]) => !v).map(([k]) => k).join(" and ") +
      " — searching will not find audio until that is fixed.", "err");
  }
}

function renderRounds() {
  const host = $("#rounds");
  host.innerHTML = "";
  if (!state.rounds.length) {
    host.innerHTML = `<div class="card"><p class="muted">No clips yet. ` +
      `${state.room.your_turn ? "You're up — make one on the Make tab."
                              : "Waiting for " + escapeHtml(state.room.turn_name) + "."}</p></div>`;
    return;
  }
  [...state.rounds].reverse().forEach((r) => host.append(roundCard(r)));
}

function roundCard(r) {
  const el = document.createElement("div");
  el.className = "card round";
  const open = r.state === "open";
  const parts = [];

  parts.push(`<div class="round-head">
      <span class="chip">${open ? r.reward_now + " pts" : "finished"}${
        r.hints_delivered ? ` · ${r.hints_delivered} hint${r.hints_delivered > 1 ? "s" : ""}` : ""}</span>
      <span class="muted small">${r.role === "sender" ? "you sent this"
        : "from " + escapeHtml(r.sender_name)}</span>
    </div>`);
  parts.push(`<audio controls preload="none" src="/api/rooms/${state.code}/audio/${r.clip_id}.wav"></audio>`);
  r.hint_clip_ids.forEach((cid, i) => {
    parts.push(`<div class="muted small">hint ${i + 1}</div>
      <audio controls preload="none" src="/api/rooms/${state.code}/audio/${cid}.wav"></audio>`);
  });

  parts.push(`<ul class="table">` + r.table.map((t) =>
    `<li class="st-${t.status}">${escapeHtml(t.name)}${t.you ? " (you)" : ""}` +
    `<span>${t.status === "solved" ? "+" + t.awarded
            : t.status === "revealed" ? "gave up"
            : t.disputed ? "disputed" : "guessing…"}</span></li>`).join("") + `</ul>`);

  if ((r.your_guesses || []).length) {
    parts.push(`<ul class="guesses">` + r.your_guesses.map((g) =>
      `<li class="${g.correct ? "correct" : ""}">${escapeHtml(g.text)}${g.correct ? " ✓" : ""}</li>`
    ).join("") + `</ul>`);
  }
  if (r.answer !== undefined) {
    parts.push(`<div class="answer-reveal">${escapeHtml(r.answer)}</div>`);
  }
  el.innerHTML = parts.join("");

  if (open && r.role === "guesser" && r.your_status === "guessing") {
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Movie or show";
    input.autocomplete = "off";
    const row = document.createElement("div");
    row.className = "row";
    row.style.marginTop = "10px";
    row.append(
      button("Guess", "btn btn-primary", () =>
        act(`/api/rounds/${r.id}/guess`, { text: input.value })),
      button(r.hint_pending ? "Hint requested" : "Ask for a hint", "btn", () =>
        act(`/api/rounds/${r.id}/hint-request`, {}), r.hint_pending),
      button("Give up", "btn btn-ghost", () => act(`/api/rounds/${r.id}/reveal`, {})),
    );
    if ((r.your_guesses || []).length && !r.you_disputed) {
      row.append(button("That should count", "btn btn-ghost", () =>
        act(`/api/rounds/${r.id}/dispute`, {})));
    }
    el.append(input, row);
  }

  if (r.role === "sender" && open) {
    if (r.hint_pending) {
      const note = document.createElement("p");
      note.className = "msg";
      note.textContent =
        `${r.hint_pending_from || "Someone"} asked for a hint. Make another clip ` +
        `from the same source and send it — that is when points drop 25%.`;
      el.append(note,
        button("Send my next clip as the hint", "btn", () => {
          state.hintFor = r.id;
          setView("make");
          show($("#send-msg"), "Your next clip will be delivered as the hint.", "ok");
        }));
    }
    (r.disputes || []).forEach((d) => {
      const row = document.createElement("div");
      row.className = "row";
      row.style.marginTop = "10px";
      const label = document.createElement("span");
      label.className = "muted small";
      label.textContent = `${d.name} says "${d.guess}" should count.`;
      row.append(label,
        button("Allow it", "btn btn-primary", () =>
          act(`/api/rounds/${r.id}/review`, { player_id: d.player_id, accept: true })),
        button("No", "btn btn-ghost", () =>
          act(`/api/rounds/${r.id}/review`, { player_id: d.player_id, accept: false })));
      el.append(row);
    });
    el.append(button("End this round", "btn btn-ghost btn-sm", () =>
      act(`/api/rounds/${r.id}/close`, {})));
  }
  return el;
}

function button(label, cls, fn, disabled) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = cls;
  b.textContent = label;
  b.disabled = !!disabled;
  b.addEventListener("click", async () => {
    b.disabled = true;
    try { await fn(); } catch (e) { alert(e.message); b.disabled = false; }
  });
  return b;
}

async function act(path, body) {
  await api(path, { method: "POST", body: JSON.stringify(body) });
  await refresh();
}

function renderScore() {
  $("#score-body").innerHTML = state.scoreboard.map((p) =>
    `<div class="score-row"><span>${escapeHtml(p.name)}` +
    `${p.id === state.me.id ? " (you)" : ""}</span>` +
    `<span class="n">${p.points}</span></div>`).join("") ||
    `<p class="muted">Nobody has scored yet.</p>`;
}

/* ----------------------------------------------------------------- wire */

async function saveName() {
  const name = $("#me-name").value.trim();
  if (!name) return null;
  return api("/api/session", { method: "POST", body: JSON.stringify({ name }) });
}

async function createRoom() {
  const msg = $("#lobby-msg");
  const name = $("#me-name").value.trim();
  if (!name) { show(msg, "Pick a name first.", "err"); $("#me-name").focus(); return; }
  try {
    const r = await api("/api/rooms", {
      method: "POST", body: JSON.stringify({ player_name: name }),
    });
    await enterRoom(r.code);
    setView("make");
  } catch (e) { show(msg, e.message, "err"); }
}

async function joinRoom(codeFromUrl) {
  const msg = $("#lobby-msg");
  const code = normCode(codeFromUrl || $("#code").value);
  const name = $("#me-name").value.trim();
  if (code.length !== 6) { show(msg, "A room code is six letters.", "err"); return; }
  if (!name) { show(msg, "Pick a name first.", "err"); $("#me-name").focus(); return; }
  try {
    await api(`/api/rooms/${code}/join`, {
      method: "POST", body: JSON.stringify({ name }),
    });
    await enterRoom(code);
  } catch (e) { show(msg, e.message, "err"); }
}

async function shareRoom() {
  const url = location.origin + "/r/" + state.code;
  const text = `Join my Quote Club room: ${pretty(state.code)}`;
  try {
    if (navigator.share) { await navigator.share({ title: "Quote Club", text, url }); return; }
    await navigator.clipboard.writeText(url);
    show($("#turn-note"), "Link copied.", "ok");
  } catch (_) {
    show($("#turn-note"), url, null);
  }
}

function init() {
  bindDrag();
  $("#tabs").addEventListener("click", (e) => {
    const t = e.target.closest(".tab");
    if (t) setView(t.dataset.view);
  });
  $("#brand").addEventListener("click", showLobby);
  $("#make-room").addEventListener("click", createRoom);
  $("#join").addEventListener("click", () => joinRoom());
  $("#code").addEventListener("keydown", (e) => { if (e.key === "Enter") joinRoom(); });
  $("#me-name").addEventListener("change", () => saveName().catch(() => {}));
  $("#share").addEventListener("click", shareRoom);
  $("#leave").addEventListener("click", async () => {
    if (!confirm("Leave this room?")) return;
    try { await api(`/api/rooms/${state.code}/leave`, { method: "POST" }); } catch (_) {}
    showLobby();
  });
  $("#search").addEventListener("click", runSearch);
  $("#q").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
  $("#search-cancel").addEventListener("click", () => {
    state.es?.close();
    state.es = null;
    $("#search").disabled = false;
    $("#progress-phase").textContent = "Stopped";
  });
  $("#preview").addEventListener("click", playSelection);
  $("#points").addEventListener("input", (e) => {
    e.target.dataset.touched = "1";
    $("#points-val").textContent = e.target.value;
  });
  $("#send").addEventListener("click", sendClip);

  const deep = location.pathname.match(/^\/r\/([A-Za-z]{6})$/);
  loadMe()
    .then(() => {
      if (!deep) return;
      const code = normCode(deep[1]);
      $("#code").value = pretty(code);
      // Already seated? Walk straight in. Otherwise ask for a name first.
      return api(`/api/rooms/${code}/state`)
        .then(() => enterRoom(code))
        .catch(() => show($("#lobby-msg"),
          `Room ${pretty(code)} is ready — put your name in and press Join.`, "ok"));
    })
    .catch(() => {});
}

document.addEventListener("DOMContentLoaded", init);
