// The entry point: pick the page for this path, draw it, and draw it again
// only when the reader asks. NOTHING POLLS. Every fetch is one request the
// reader caused — arriving, pressing r or the refresh control, or coming back
// to the tab after a while. Found live: pages ticking every 3 s filled the
// venue's one container with background polls, and the reader's own click
// waited a minute behind them. A sealed wave, as ever, is drawn once.
"use strict";

import {nav, route, stamp} from "./nav.js";
import {drawFleet} from "./fleet.js";
import {drawHost} from "./host.js";
import {drawCharts} from "./metrics.js";
import {drawIndex} from "./runs.js";
import {drawRun} from "./run.js";
import {drawWave} from "./wave.js";

const PAGES = {runs: drawIndex, run: drawRun, fleet: drawFleet,
               host: drawHost, wave: drawWave, charts: drawCharts};
// coming back to a tab HIDDEN for this long redraws it; sooner would only
// read the same bytes back out of the venue's memory. Time hidden, not time
// since the draw: a fresh tab announces itself visible as it loads, and that
// must not be a second fetch (seen: /api/runs twice, 150 ms apart)
const STALE_ON_RETURN_MS = 30000;

const draw = PAGES[route.page];
let drawnAt = 0;
let drawing = null;
let hiddenAt = null;

// ONE DRAW AT A TIME: a request already in flight IS the refresh
function refresh() {
  if (drawing !== null) return drawing;
  drawing = Promise.resolve(draw()).then(() => {
    // a draw that came back empty keeps the last render and says so
    // (dom.js's freshness contract); its stamp stays with the old bytes
    const lost = document.getElementById("stale");
    if (lost && lost.style.display === "block") return;
    drawnAt = Date.now();
    stamp(drawnAt);
  }).finally(() => { drawing = null; });
  return drawing;
}

function typing(target) {
  const tag = target && target.tagName;
  return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";
}

nav(refresh);
refresh();
addEventListener("keydown", event => {
  if (event.key === "r" && !event.metaKey && !event.ctrlKey && !event.altKey
      && !typing(event.target)) refresh();
});
addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") { hiddenAt = Date.now(); return; }
  const away = hiddenAt === null ? 0 : Date.now() - hiddenAt;
  hiddenAt = null;
  if (route.page !== "wave" && away > STALE_ON_RETURN_MS) refresh();
});
