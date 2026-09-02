// The entry point: pick the page for this path, draw it, and poll.
//
// THE POLL RULE (#50): a redraw never happens under the cursor. And a SEALED
// wave never polls at all — its bytes are immutable by the ledger, so the wave
// page draws once and stays put.
"use strict";

import {poll} from "./dom.js";
import {nav, route} from "./nav.js";
import {drawFleet} from "./fleet.js";
import {drawHost} from "./host.js";
import {drawCharts} from "./metrics.js";
import {drawIndex} from "./runs.js";
import {drawRun} from "./run.js";
import {drawWave} from "./wave.js";

const PAGES = {runs: drawIndex, run: drawRun, fleet: drawFleet,
               host: drawHost, wave: drawWave, charts: drawCharts};
const REFRESH_MS = 3000;

const draw = PAGES[route.page];
nav();
draw();
if (route.page !== "wave")
  setInterval(() => { if (!poll.hovering) draw(); }, REFRESH_MS);
