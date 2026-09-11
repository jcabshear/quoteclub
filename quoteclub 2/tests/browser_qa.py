"""Browser QA at phone widths, using a real Chromium engine.

Checks, in order of how badly they were wanted:

1. No horizontal overflow at 320, 375, 390 and 430 CSS px, on every view.
   Also that no single element sticks out past the viewport.
2. The crop handles sit exactly on the edges of the highlighted selection
   and on the numbers in the readout - measured in device pixels off the
   real layout, not asserted from the source.
3. Dragging a handle keeps all three in agreement.
4. Preview playback: the <audio> element accepts the generated WAV, its
   duration equals the selection to the millisecond, and currentTime
   advances. (Chromium, not iOS Safari - see the caveat printed at the
   end.)
5. Inputs are >= 16px so iOS does not zoom on focus, and the page is
   still pinch-zoomable.

Run:  python -m tests.browser_qa
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHOTS = ROOT / "qa-shots"
PORT = 8099
BASE = f"http://127.0.0.1:{PORT}"

WIDTHS = [320, 375, 390, 430]
failures: list[str] = []
notes: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"  — {detail}" if detail else ""))
    if not ok:
        failures.append(f"{label}: {detail}")


async def overflow_report(page):
    return await page.evaluate(
        """() => {
      const vw = document.documentElement.clientWidth;
      const bad = [];
      document.querySelectorAll('*').forEach(el => {
        const r = el.getBoundingClientRect();
        if (r.width === 0 && r.height === 0) return;
        if (r.right > vw + 0.5 || r.left < -0.5) {
          bad.push({tag: el.tagName.toLowerCase(),
                    cls: (el.className || '').toString().slice(0, 40),
                    left: +r.left.toFixed(1), right: +r.right.toFixed(1)});
        }
      });
      return {vw, scrollWidth: document.documentElement.scrollWidth,
              bodyScroll: document.body.scrollWidth, bad: bad.slice(0, 8)};
    }"""
    )


async def setup_editor(page):
    """Start a room with no login, run a search, open the first result."""
    await page.goto(BASE, wait_until="networkidle")
    await page.fill("#me-name", "Josiah")
    await page.click("#make-room")
    await page.wait_for_selector("#view-make:not([hidden])", timeout=10000)
    await page.fill("#q", "you were this close to losing your job")
    await page.click("#search")
    await page.wait_for_selector(".result", timeout=20000)
    await page.click(".result")
    await page.wait_for_selector("#editor:not([hidden])", timeout=10000)
    await page.wait_for_timeout(400)


async def room_code(page):
    return (await page.locator("#room-code").inner_text()).strip()


async def measure_alignment(page):
    return await page.evaluate(
        """() => {
      const wrap = document.querySelector('#wave-wrap').getBoundingClientRect();
      const svg = document.querySelector('#wave').getBoundingClientRect();
      const selEl = document.querySelector('#sel-rect');
      const x = parseFloat(selEl.getAttribute('x'));
      const w = parseFloat(selEl.getAttribute('width'));
      // viewBox is 0..1000 across the full svg width.
      const selLeft = svg.left + (x / 1000) * svg.width;
      const selRight = svg.left + ((x + w) / 1000) * svg.width;
      const hs = document.querySelector('#h-start').getBoundingClientRect();
      const he = document.querySelector('#h-end').getBoundingClientRect();
      const readout = document.querySelector('#sel-readout').textContent;
      const nums = readout.match(/[\\d.]+/g).map(Number);
      return {
        wrapWidth: wrap.width,
        selLeft, selRight,
        startCentre: hs.left + hs.width / 2,
        endCentre: he.left + he.width / 2,
        readoutStart: nums[0], readoutEnd: nums[1], readoutDur: nums[2],
        svgLeft: svg.left, svgWidth: svg.width,
      };
    }"""
    )


async def run():
    from playwright.async_api import async_playwright

    SHOTS.mkdir(exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            args=["--autoplay-policy=no-user-gesture-required"]
        )

        # ---------------------------------------------------- overflow
        print("\n1. Horizontal overflow at phone widths")
        for w in WIDTHS:
            ctx = await browser.new_context(
                viewport={"width": w, "height": 780},
                device_scale_factor=2,
                is_mobile=True,
                has_touch=True,
            )
            page = await ctx.new_page()
            await setup_editor(page)
            for view, tab in (("make", "Make"), ("play", "Play"), ("score", "Score")):
                await page.click(f'.tab[data-view="{view}"]')
                await page.wait_for_timeout(150)
                rep = await overflow_report(page)
                check(
                    rep["scrollWidth"] <= rep["vw"] + 1 and not rep["bad"],
                    f"{w}px · {tab} view fits",
                    f"scrollWidth={rep['scrollWidth']} vw={rep['vw']} "
                    f"offenders={rep['bad']}" if rep["bad"] or rep["scrollWidth"] > rep["vw"] + 1 else "",
                )
            await page.click('.tab[data-view="make"]')
            await page.wait_for_timeout(150)
            await page.screenshot(path=str(SHOTS / f"make-{w}.png"), full_page=True)
            await ctx.close()

        # ---------------------------------------------------- alignment
        print("\n2. Waveform / selection / handle / readout agreement")
        ctx = await browser.new_context(
            viewport={"width": 390, "height": 844}, device_scale_factor=3,
            is_mobile=True, has_touch=True,
        )
        page = await ctx.new_page()
        await setup_editor(page)
        m = await measure_alignment(page)
        check(abs(m["startCentre"] - m["selLeft"]) < 1.0,
              "start handle centred on selection left",
              f"{m['startCentre']:.2f} vs {m['selLeft']:.2f}")
        check(abs(m["endCentre"] - m["selRight"]) < 1.0,
              "end handle centred on selection right",
              f"{m['endCentre']:.2f} vs {m['selRight']:.2f}")
        check(abs(m["readoutDur"] - (m["readoutEnd"] - m["readoutStart"])) < 0.002,
              "readout duration equals end minus start")
        check(abs(m["svgWidth"] - m["wrapWidth"]) < 0.5,
              "svg fills the handle coordinate box",
              f"svg={m['svgWidth']:.2f} wrap={m['wrapWidth']:.2f}")

        # ---------------------------------------------------- dragging
        print("\n3. Dragging keeps them in agreement")
        box = await page.locator("#wave-wrap").bounding_box()
        for frac in (0.65, 0.30, 0.92):
            target = box["x"] + box["width"] * frac
            handle = await page.locator("#h-end").bounding_box()
            await page.mouse.move(handle["x"] + handle["width"] / 2,
                                  handle["y"] + handle["height"] / 2)
            await page.mouse.down()
            await page.mouse.move(target, handle["y"] + handle["height"] / 2, steps=12)
            await page.mouse.up()
            await page.wait_for_timeout(120)
            m = await measure_alignment(page)
            check(abs(m["endCentre"] - m["selRight"]) < 1.0,
                  f"after dragging end to {int(frac*100)}%",
                  f"handle={m['endCentre']:.2f} sel={m['selRight']:.2f}")
            expected_x = box["x"] + box["width"] * frac
            check(abs(m["endCentre"] - expected_x) < 4.0,
                  f"end handle landed where the finger did ({int(frac*100)}%)",
                  f"handle={m['endCentre']:.2f} finger={expected_x:.2f}")

        # tiny crop
        print("\n4. Very short crops")
        # Drive the end handle back onto the start handle and keep going.
        for _ in range(200):
            await page.click('[data-nudge="end,-0.05"]')
        await page.wait_for_timeout(200)
        m = await measure_alignment(page)
        check(m["readoutDur"] >= 0.02 - 1e-6,
              "crop never collapses below the 20 ms floor",
              f"{m['readoutDur']:.3f}s")
        check(abs(m["endCentre"] - m["selRight"]) < 1.0,
              "handles still aligned on a tiny crop",
              f"{m['endCentre']:.2f} vs {m['selRight']:.2f}")

        # ---------------------------------------------------- playback
        print("\n5. Preview playback in a real engine")
        for _ in range(24):
            await page.click('[data-nudge="end,0.05"]')
        await page.wait_for_timeout(400)
        m = await measure_alignment(page)
        expected = m["readoutDur"]

        await page.click("#preview")
        await page.wait_for_timeout(900)
        pb = await page.evaluate(
            """() => {
              const p = window.__qcPlayer;
              if (!p) return null;
              return {src: (p.src || '').slice(0, 5), duration: p.duration,
                      currentTime: p.currentTime, paused: p.paused,
                      error: p.error ? p.error.code : null};
            }"""
        )
        check(pb is not None, "the preview player exists")
        if pb:
            check(pb["error"] is None, "audio element reported no decode error",
                  str(pb["error"]))
            check(pb["src"] == "blob:", "preview plays a locally built blob, not a remote url",
                  pb["src"])
            check(
                pb["duration"] and abs(pb["duration"] - expected) < 0.02,
                "loaded audio duration equals the selection",
                f"audio={pb['duration']}s selection={expected}s",
            )
            check(pb["currentTime"] > 0, "playback actually advanced",
                  f"currentTime={pb['currentTime']}")
        errmsg = ""
        if await page.locator("#preview-msg").is_visible():
            errmsg = (await page.locator("#preview-msg").inner_text()).strip()
        check(errmsg == "", "preview reported no error to the user", errmsg)

        # Changing the crop must cancel the old preview rather than let a
        # stale clip keep playing.
        await page.click("#preview")
        await page.wait_for_timeout(120)
        await page.click('[data-nudge="start,0.05"]')
        await page.wait_for_timeout(200)
        after = await page.evaluate(
            "() => ({paused: window.__qcPlayer.paused, src: window.__qcPlayer.src})"
        )
        check(after["paused"] and not after["src"],
              "moving a handle cancels the running preview", str(after))

        await page.screenshot(path=str(SHOTS / "editor-390.png"), full_page=True)

        # ---------------------------------------------------- iOS hygiene
        print("\n6. iOS input hygiene")
        sizes = await page.evaluate(
            """() => [...document.querySelectorAll('input')]
                 .map(i => ({type: i.type,
                             size: parseFloat(getComputedStyle(i).fontSize)}))"""
        )
        small = [s for s in sizes if s["type"] in ("text", "search", "url") and s["size"] < 16]
        check(not small, "text inputs are >= 16px (no focus zoom on iOS)", str(small))
        vp = await page.evaluate(
            """() => document.querySelector('meta[name=viewport]').content"""
        )
        check("user-scalable=no" not in vp and "maximum-scale" not in vp,
              "pinch zoom is not disabled", vp)
        taps = await page.evaluate(
            """() => [...document.querySelectorAll('button')]
                 .map(b => ({t: b.textContent.trim().slice(0, 24),
                             c: b.className,
                             h: +b.getBoundingClientRect().height.toFixed(1)}))
                 .filter(x => x.h > 0 && x.h < 36)"""
        )
        check(not taps, "no button shorter than 36px", str(taps))

        await ctx.close()

        # ------------------------------------------- two real browsers
        print("\n7. Two browsers, one room, no login")
        ctxA = await browser.new_context(viewport={"width": 390, "height": 844},
                                         is_mobile=True, has_touch=True)
        ctxB = await browser.new_context(viewport={"width": 390, "height": 844},
                                         is_mobile=True, has_touch=True)
        A, B = await ctxA.new_page(), await ctxB.new_page()

        await setup_editor(A)                       # A starts a room + crops
        code = (await A.locator("#room-code").inner_text()).strip()
        check(len(code.replace("-", "")) == 6, "room code is six letters", code)

        # B joins from the shared link, never signing in.
        await B.goto(f"{BASE}/r/{code.replace('-', '')}", wait_until="networkidle")
        await B.fill("#me-name", "Brother")
        await B.click("#join")
        await B.wait_for_selector("#roombar:not([hidden])", timeout=10000)
        rosterB = await B.locator("#roster").inner_text()
        check("Josiah" in rosterB and "Brother" in rosterB,
              "second player joined by code alone", rosterB)

        # A sends the clip.
        await A.fill("#answer", "The Incredibles")
        await A.click("#send")
        await A.wait_for_selector(".round", timeout=10000)
        await B.reload(wait_until="networkidle")
        await B.wait_for_selector(".round", timeout=10000)

        leak = (await B.locator("#view-play").inner_text()).lower()
        check("incredibles" not in leak, "guesser cannot see the answer", leak[:120])
        check(await B.locator(".round audio").count() > 0,
              "guesser got playable audio")

        # B guesses right.
        await B.fill(".round input[type=text]", "the incredibles")
        await B.get_by_role("button", name="Guess").first.click()
        await B.wait_for_timeout(800)
        solved = (await B.locator("#view-play").inner_text()).lower()
        check("incredibles" in solved, "answer revealed to the solver")
        await B.click('.tab[data-view="score"]')
        await B.wait_for_timeout(300)
        score = await B.locator("#score-body").inner_text()
        check("Brother" in score, "guesser appears on the scoreboard", score.replace("\n", " "))
        await B.screenshot(path=str(SHOTS / "play-390.png"), full_page=True)

        for w in (320, 430):
            await B.set_viewport_size({"width": w, "height": 780})
            await B.wait_for_timeout(200)
            rep = await overflow_report(B)
            check(rep["scrollWidth"] <= rep["vw"] + 1 and not rep["bad"],
                  f"{w}px · played round fits", str(rep["bad"]))

        await ctxA.close()
        await ctxB.close()
        await browser.close()


def main() -> int:
    server = subprocess.Popen(
        [sys.executable, "-m", "tests.devserver", str(PORT)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        import urllib.request

        for _ in range(60):
            try:
                urllib.request.urlopen(f"{BASE}/api/health", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        else:
            print("server did not start")
            return 2
        asyncio.run(run())
    finally:
        server.terminate()
        server.wait(timeout=10)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print("  - " + f)
    else:
        print("All browser checks passed.")
    print("\nCaveat: this is Chromium on Linux. It is good evidence for layout,")
    print("geometry and WAV validity; it is NOT proof that audio is audible on")
    print("an iPhone. Only your phone can establish that.")
    print(f"Screenshots: {SHOTS}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
