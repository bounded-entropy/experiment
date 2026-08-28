// One experiment. Panel priority IS the run's own dictionary.json (I11): the
// walkback from the loss first, then the two readings a run is judged by
// whatever its loss — PARITY (logprob_gap, the standing alarm) and STEP
// ECONOMICS (where each update's wall seconds went) — then the rails, the
// user's own panels, the measurements, and the tail of sealed waves.
"use strict";

import {C, brief, el, esc, getJSON, note, section} from "./dom.js";
import {card, emptyCard, stackedCard} from "./charts.js";
import {apiRun, ctx, drawAmbiguity, hostPath, legend, route, syncSwitcher,
        wavePath} from "./nav.js";

const GAP = "logprob_gap";      // the parity alarm (#25's certificate, running)
const PHASES = [{name: "collect", color: C.feed},
                {name: "post", color: C.derived},
                {name: "train", color: C.rail},
                {name: "seal", color: C.eval}];

export async function drawRun() {
  const runId = route.runId, folder = route.folder;
  const [data, runs, timing, waves] = await Promise.all([
    getJSON(apiRun(runId, "", folder)),
    getJSON("/api/runs"),
    getJSON(apiRun(runId, "/timing", folder)),
    getJSON(apiRun(runId, "/waves", folder)),
  ]);
  if (!data) { document.getElementById("page").textContent = "unknown run"; return; }
  if (data.ambiguous) { drawAmbiguity(runId, data.ambiguous); return; }
  syncSwitcher(runs || []);
  const mine = (runs || []).find(r => r.run_id === runId
      && (folder === null || r.folder === folder));
  const dict = data.dictionary || {columns: [], rails: []};
  // the NAME leads and the hex id is secondary: a run is a thing a human
  // named, addressed by (folder, run_id) underneath
  ctx(`<span>${esc((mine && mine.name) || runId)}</span>`
    + ((mine && mine.name) ? `<span class="k">${esc(runId)}</span>` : "")
    + ((mine && mine.folder) ? `<span class="k">${esc(mine.folder)}</span>` : "")
    + ((mine && mine.tags || []).map(t =>
        `<span class="tag">${esc(t)}</span>`).join("")
    + `<span class="meta">loss ${esc(dict.loss ?? "?")} · lag ${esc(dict.max_policy_lag ?? 0)}
     · post [${(dict.post_pipeline || []).map(esc).join(" → ")}]</span>
     <span class="${data.committed >= (data.target ?? 1e9) ? "meta" : "live"}">
     ${data.committed}/${esc(data.target ?? "?")} committed</span>`)
    + (mine ? `<span class="meta">on ${mine.hosts.map(h =>
        `<a href="${hostPath(h, folder)}" style="color:${C.feed}">${esc(h)}</a>`
      ).join(" + ") || "?"}</span>` : ""));
  document.getElementById("page").innerHTML = "";

  const cols = dict.columns || [];
  const postCols = cols.filter(c => c.phase === "post" && c.stored);
  const evalCols = cols.filter(c => c.phase === "eval" && c.stored);
  const feeding = postCols.filter(c => c.feeds_loss);
  const measure = postCols.filter(c => !c.feeds_loss);
  const trainOf = (sec, name) => data.updates
      .filter(u => u[sec][name] !== undefined && u[sec][name] !== null)
      .map(u => [u.update, u[sec][name]]);
  const evalOf = name => data.eval.filter(e => e.means[name] !== undefined)
      .map(e => [e.update, e.means[name]]);
  const pair = (name, color) => [
    {label: name, color: color, points: trainOf("post", name)},
    {label: "eval", color: C.eval, dash: true, points: evalOf(name)}];

  if (feeding.length) {
    const grid = section("feeds the loss", "walkback from " + (dict.loss ?? "?"));
    for (const c of feeding)
      grid.append(card(c.name, c.producer.replace("postprocessor:", ""),
                       pair(c.name, C.feed)));
  }
  drawParity(trainOf("train", GAP), dict);
  drawSteps(timing);

  const rails = (dict.rails || []).filter(r => r !== GAP);
  if (rails.length) {
    const grid = section("rails");
    for (const r of rails)
      grid.append(card(r, "loss:" + (dict.loss ?? "?"),
                       [{label: r, color: C.rail, points: trainOf("train", r)}]));
  }
  const derived = data.derived || [];
  if (derived.length) {
    const grid = section("derived", "your panels.json");
    for (const d of derived) {
      if (d.missing || d.error) {
        grid.append(emptyCard(d.name, d.expr, d.error ? d.error
            : "missing from this run's pipeline: " + d.missing.join(", ")));
      } else {
        grid.append(card(d.name, d.expr, [
          {label: d.name, color: C.derived, points: d.points},
          {label: "eval", color: C.eval, dash: true, points: d.eval}]));
      }
    }
  }
  if (measure.length || evalCols.length) {
    const grid = section("measurement");
    const seen = new Set();
    for (const c of measure) {
      seen.add(c.name);
      grid.append(card(c.name, c.producer.replace("postprocessor:", ""),
                       pair(c.name, C.dim)));
    }
    for (const c of evalCols)
      if (!seen.has(c.name))
        grid.append(card(c.name + " (eval)", c.producer.replace("postprocessor:", ""),
            [{label: "eval", color: C.eval, dash: true, points: evalOf(c.name)}]));
  }
  drawWaves(waves);
  legend("solid = training · <span style='color:#ffb86b'>dashed = held-out eval</span>"
    + " · hover any chart for the raw values · refreshes every 3s"
    + " · panels & priority from the run's own dictionary.json"
    + " · a sealed wave is read from the store, never from a live run");
}

// ---- parity: the alarm that is not about the objective --------------------

function drawParity(points, dict) {
  if (!points.length) return;
  const grid = section("parity", "logprob_gap — the running certificate (#25)");
  grid.append(card(GAP, "loss:" + (dict.loss ?? "?"),
      [{label: GAP, color: C.warn, points: points}],
      {reference: [{y: 0, label: "identical kernels", color: C.dim}]}));
}

// ---- step economics: where the wall clock went ----------------------------

function drawSteps(timing) {
  const updates = ((timing || {}).updates || []).filter(u => u.seconds > 0);
  const grid = section("step economics",
      "wall seconds per committed update, and the four phases inside them", "half");
  if (!updates.length) {
    grid.append(emptyCard("steps/s", "host journal",
        "no update events journaled for this run yet — a host emits one per "
        + "completed update"));
    return;
  }
  grid.append(card("steps/s", "1 / update seconds",
      [{label: "steps/s", color: C.rail,
        points: updates.map(u => [u.update, 1 / u.seconds])}],
      {y0: 0, unit: "/s"}));
  grid.append(stackedCard("update duration", "collect · post · train · seal",
      updates.map(u => ({x: u.update, at: u.t, total: u.seconds,
                         parts: u.phases || {}})),
      PHASES, {unit: "s"}));
}

// ---- the tail of sealed waves ---------------------------------------------

function drawWaves(waves) {
  const rows = (waves || {}).waves || [];
  document.getElementById("page").append(
      el("h2", {}, "sealed waves <span>the ledger's tail — click one to read it</span>"));
  if (!rows.length) { note("no committed waves yet"); return; }
  const table = el("table", {}, "<tr><th>update</th><th>trajectories</th>"
      + "<th>groups</th><th>post means</th><th>bundle</th></tr>");
  for (const wave of rows.slice().reverse()) {
    const row = el("tr", {});
    row.append(el("td", {},
        `<a href="${wavePath(route.runId, wave.update, route.folder)}">`
        + `wave ${wave.update}</a>`));
    row.append(el("td", {class: "n"}, esc(wave.trajectories ?? "?")));
    row.append(el("td", {class: "n"}, esc(wave.groups ?? "?")));
    row.append(el("td", {class: "k"}, Object.entries(wave.post || {})
        .map(([k, v]) => `${esc(k)} ${brief(v)}`).join(" · ") || "—"));
    row.append(el("td", {class: "k"}, esc(wave.bundle_id ?? "—")));
    table.append(row);
  }
  document.getElementById("page").append(table);
}
