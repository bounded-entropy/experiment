// One experiment. Panel priority IS the run's own dictionary.json (I11): the
// walkback from the loss first, then the two readings a run is judged by
// whatever its loss — PARITY (logprob_gap, the standing alarm) and STEP
// ECONOMICS (where each update's wall seconds went) — then the rails, the
// user's own panels, the measurements, and the tail of sealed waves.
"use strict";

import {C, brief, drawnOnce, el, esc, getAnswer, getJSON, lostTick, note,
        okTick, section} from "./dom.js";
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
  const [answer, runs, timing, waves] = await Promise.all([
    getAnswer(apiRun(runId, "", folder)),
    getJSON("/api/runs").then(d => (d && d.runs) || []),
    getJSON(apiRun(runId, "/timing", folder)),
    getJSON(apiRun(runId, "/waves", folder)),
  ]);
  const data = answer.data;
  if (!data) {
    // the freshness contract: a drawn page outranks a failed poll — even a
    // POSITIVE 404 after a good render reads as transient (a volume reload
    // can hide a run dir for one scan), never as the run vanishing
    if (drawnOnce()) { lostTick(); return; }
    document.getElementById("page").textContent =
        answer.missing ? "unknown run" : "observer unreachable — retrying";
    return;
  }
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
     ${data.committed}/${esc(data.target ?? "?")}
     ${data.extent === "rollout" ? "rollouts sealed" : "committed"}</span>`)
    + (mine ? `<span class="meta">on ${mine.hosts.map(h =>
        `<a href="${hostPath(h, folder)}">${esc(h)}</a>`
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
  // held-out overlays, both eras: the legacy in-run eval plus every NAMED
  // measurement (#70) carrying this metric — one dashed series each
  const measured = name => (data.measurements || []).map(m => ({
      label: m.name, color: C.eval, dash: true,
      points: m.points.filter(p => p.means[name] !== undefined)
                      .map(p => [p.update, p.means[name]])}))
      .filter(s => s.points.length);
  const pair = (name, color) => [
    {label: name, color: color, points: trainOf("post", name)},
    {label: "eval", color: C.eval, dash: true, points: evalOf(name)},
    ...measured(name)];

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
  // the bank's declared per-update stats: kind "stat" train-phase dictionary
  // nodes from adapter types (a latent KL, a posterior scale) — the trainer's
  // own bookkeeping (producer "trainer") already renders with the rails
  const adapterStats = cols.filter(c =>
      c.phase === "train" && c.kind === "stat" &&
      c.producer.startsWith("adapter:"));
  if (adapterStats.length) {
    const grid = section("adapter stats", "declared provides, per update");
    for (const c of adapterStats)
      grid.append(card(c.name, c.producer.replace("adapter:", ""),
                       [{label: c.name, color: C.feed,
                         points: trainOf("train", c.name)}]));
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
  drawWaves(waves, data.extent);
  legend(`solid = training · <span style="color:${C.eval}">dashed = held-out eval</span>`
    + " · hover any chart for the raw values · refreshes every 3s"
    + " · panels & priority from the run's own dictionary.json"
    + " · a sealed wave is read from the store, never from a live run");
  okTick();
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

function drawWaves(waves, extent) {
  const rows = (waves || {}).waves || [];
  // a run whose extent is ROLLOUTS has no Trainer and so no ledger at all:
  // its sealed waves are the rollouts themselves, each sealed the moment the
  // Generator's atomic write lands, and the observer lists THOSE (ADR 0006
  // Part B). Every panel fed by the ledger stays legitimately empty.
  const generated = extent === "rollout";
  document.getElementById("page").append(
      el("h2", {}, generated
        ? "sealed rollouts <span>the Generator's tail — click one to read it</span>"
        : "sealed waves <span>the ledger's tail — click one to read it</span>"));
  if (!rows.length && generated) {
    note("this run only generates: no Trainer, no ledger — its output is "
       + "the sealed rollouts, and none is sealed yet");
    return;
  }
  if (!rows.length) { note("no committed waves yet"); return; }
  const table = el("table", {}, `<tr><th>${generated ? "rollout" : "update"}</th>`
      + "<th>trajectories</th><th>groups</th><th>post means</th><th>bundle</th></tr>");
  for (const wave of rows.slice().reverse()) {
    const row = el("tr", {});
    row.append(el("td", {},
        `<a href="${wavePath(route.runId, wave.update, route.folder)}">`
        + `${generated ? "rollout" : "wave"} ${wave.update}</a>`));
    row.append(el("td", {class: "n"}, esc(wave.trajectories ?? "?")));
    row.append(el("td", {class: "n"}, esc(wave.groups ?? "?")));
    row.append(el("td", {class: "k"}, Object.entries(wave.post || {})
        .map(([k, v]) => `${esc(k)} ${brief(v)}`).join(" · ") || "—"));
    row.append(el("td", {class: "k"}, esc(wave.bundle_id ?? "—")));
    table.append(row);
  }
  document.getElementById("page").append(table);
}
