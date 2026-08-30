// One host, from hosts/<name>/log.jsonl alone: what it attested at birth, who
// has been resident, what it served (the traffic rails), and what its metal
// did (the gpu rails). Every gpu rail carries the MOMENTS beside it — boots,
// attaches, detaches, and sleeps the day the arbiter journals them — because a
// memory curve that falls off a cliff is only explained by the moment next to it.
"use strict";

import {C, WHEEL, brief, clock, el, esc, getJSON, gib, note, section, when}
  from "./dom.js";
import {card, plot, residencyTip, timeline} from "./charts.js";
import {ctx, legend, query, route, runPath, withHours} from "./nav.js";

const MOMENT_COLOR = {"host-up": C.teal, attach: C.rail, detach: C.feed,
                      sleep: C.derived, wake: C.eval};
// what the throughput and step sections already draw; the open slot below
// plots only what no reading claims
const CLAIMED_EVENTS = ["traffic.", "update."];

export async function drawHost() {
  const host = await getJSON(withHours("/api/host/" + encodeURIComponent(route.host)
                             + query(route.folder)));
  const holder = document.getElementById("page");
  if (!host) { holder.textContent = "unknown host"; return; }
  holder.innerHTML = "";
  const resident = host.tenancy.filter(t => t.detached === null).length;
  ctx(`<span>${esc(host.host)}</span>`
    + (host.folders.some(f => f)
        ? `<span class="k">${esc(host.folders.filter(f => f).join(" "))}</span>` : "")
    + `<span class="meta">engines ${esc(host.engines.join(", ") || "?")}`
    + ` · ${host.boots.length} boot${host.boots.length === 1 ? "" : "s"}`
    + ` · ${host.events} events · ${esc(host.journal_stores.map(esc).join(" "))}</span>`
    + `<span class="${resident ? "live" : "meta"}">${resident} resident</span>`);

  drawBirth(host);
  drawTenancy(host);
  drawThroughput(host);
  drawGpu(host);
  drawSlot(host);
  legend("this page is one file: hosts/" + esc(route.host) + "/log.jsonl · hover any"
    + " bar, point or ▲ moment for its raw values · refreshes every 3s");
}

// ---- birth: what a host IS (I12) ------------------------------------------

function drawBirth(host) {
  const holder = document.getElementById("page");
  holder.append(el("h2", {}, "the host <span>as it attested itself at birth</span>"));
  const partition = host.partition
    ? `${esc(host.partition.gpu || "?")} ${esc(host.partition.metal ?? host.partition.gpuset)} devices `
      + `[${(host.partition.devices || []).join(", ")}]`
      + ` · memory fraction ${esc(host.partition.memory)}`
    : "<span class='k'>not journaled (a host older than #43's partitions)</span>";
  const regimes = host.regimes.length
    ? host.regimes.map(r =>
        esc(`${r.name} = ${r.capability ?? r.kind} × ${r.base ?? "*"} × ${r.shape}`)).join("<br>")
    : "<span class='k'>not journaled (a host older than #43's regimes)</span>";
  holder.append(el("table", {},
    `<tr><th>partition</th><td>${partition}</td></tr>`
    + `<tr><th>regimes</th><td>${regimes}</td></tr>`
    + `<tr><th>engines</th><td>${esc(host.engines.join(", ") || "?")}</td></tr>`
    + `<tr><th>seen</th><td>${esc(when(host.first_seen))} → ${esc(when(host.last_seen))}`
    + ` <span class="k">(${esc(span(host))})</span></td></tr>`));
}

function span(host) {
  const seconds = (host.last_seen || 0) - (host.first_seen || 0);
  return seconds < 90 ? seconds.toFixed(0) + "s"
       : seconds < 5400 ? (seconds / 60).toFixed(1) + "m"
       : (seconds / 3600).toFixed(1) + "h";
}

// ---- tenancy: attach → detach ---------------------------------------------

function drawTenancy(host) {
  const holder = document.getElementById("page");
  holder.append(el("h2", {}, "tenancy <span>attach → detach, from the journal</span>"));
  if (!host.tenancy.length) {
    note("no tenant has ever attached here");
    return;
  }
  holder.append(timeline(host.tenancy.map(t => ({
      name: t.run_id, href: runPath(t.run_id, route.folder),
      bars: [{label: t.run_id, status: t.status, t0: t.attached, t1: t.detached,
              detail: (t.pools || []).join(", ")}]})),
    [host.first_seen, host.last_seen], residencyTip));
  const table = el("table", {}, "<tr><th>run</th><th>status</th><th>pools</th>"
      + "<th>remote pools</th><th>updates</th><th>attached</th><th>for</th>"
      + "<th>store</th></tr>");
  for (const t of host.tenancy.slice().reverse()) {
    const row = el("tr", {});
    row.append(el("td", {}, `<a href="${runPath(t.run_id, route.folder)}">${esc(t.run_id)}</a>`));
    row.append(el("td", {class: t.status === "running" ? "live" : ""}, esc(t.status)));
    row.append(el("td", {class: "k"}, esc(t.pools.join(", ") || "—")));
    row.append(el("td", {class: "k"}, esc(t.remotes.join(", ") || "—")));
    row.append(el("td", {}, esc(`${t.updates_completed ?? "?"}/${t.n_updates ?? "?"}`)));
    row.append(el("td", {class: "k"}, esc(when(t.attached))));
    row.append(el("td", {class: "k"}, esc(t.attached
        ? residency(t) : "—")));
    row.append(el("td", {class: "k"}, esc(t.store ?? "—")));
    table.append(row);
  }
  holder.append(table);
}

function residency(t) {
  const seconds = (t.detached || Date.now() / 1000) - t.attached;
  return seconds < 90 ? seconds.toFixed(0) + "s"
       : seconds < 5400 ? (seconds / 60).toFixed(1) + "m"
       : (seconds / 3600).toFixed(1) + "h";
}

// ---- the traffic rails: what this host served -----------------------------

function drawThroughput(host) {
  const ch = host.channels || {};
  const has = name => (ch[name] || []).length > 0;
  const marks = momentMarks(host.moments);
  const opts = extra => Object.assign(
      {xlabel: clock, breakGaps: true, y0: 0, markers: marks}, extra || {});
  if (!has("prefill_tok_s") && !has("requests_s") && !has("inflight")
      && !has("ttft_ms")) {
    document.getElementById("page").append(el("h2", {},
        "throughput <span>windowed traffic, as the host journals it</span>"));
    note("no traffic events journaled yet — a serving host emits one per stats "
       + "tick: prefill and decode tokens, requests, ttft, admission wait, inflight");
    return;
  }
  const grid = section("throughput", "windowed traffic, as the host journaled it");
  if (has("prefill_tok_s") || has("decode_tok_s"))
    grid.append(card("tokens/s", "prefill · decode", [
        {label: "prefill", color: C.feed, points: ch.prefill_tok_s || []},
        {label: "decode", color: C.rail, points: ch.decode_tok_s || []}],
      opts({unit: " tok/s"})));
  if (has("requests_s"))
    grid.append(card("requests/s", "completed per second",
        [{label: "requests/s", color: C.derived, points: ch.requests_s}],
        opts({unit: "/s"})));
  if (has("inflight"))
    grid.append(card("in flight", "requests resident in the engine",
        [{label: "inflight", color: C.teal, points: ch.inflight}], opts({})));
  if (has("ttft_ms") || has("admit_wait_ms"))
    grid.append(card("latency", "ttft · admission wait", [
        {label: "ttft", color: C.eval, points: ch.ttft_ms || []},
        {label: "admit wait", color: C.warn, points: ch.admit_wait_ms || []}],
      opts({unit: " ms"})));
}

function momentMarks(moments) {
  return (moments || []).map(m => ({
    t: m.t, color: MOMENT_COLOR[m.event] || C.dim,
    label: m.event + (m.run_id ? " " + m.run_id : "")
                   + (m.status ? " (" + m.status + ")" : "")}));
}

// ---- the gpu rails: one card per device, two rails, the totals as a line --

function drawGpu(host) {
  if (!host.gpus.length) {
    document.getElementById("page").append(el("h2", {}, "gpu"));
    note("no stats events journaled (run_stats not running, or no NVIDIA runtime)");
    return;
  }
  const marks = momentMarks(host.moments);
  const grid = section("gpu",
      "one card per device · utilization over HBM, with the device total as the line");
  for (const g of host.gpus) grid.append(deviceCard(g, marks));
}

function deviceCard(g, marks) {
  const color = WHEEL[g.device % WHEEL.length];
  const node = el("div", {class: "card"});
  node.append(el("div", {}, `<span class="name">gpu${g.device}</span>`
      + `<span class="who">${g.mem_total ? esc(gib(g.mem_total)) + " HBM" : "?"}</span>`));
  node.append(el("div", {class: "rail"}, "utilization %"));
  node.append(plot([{label: "gpu" + g.device, color: color, points: g.util}],
      {unit: "%", y0: 0, y1: 100, H: 110, xlabel: clock, breakGaps: true,
       markers: marks}));
  node.append(el("div", {class: "rail"}, "HBM MiB"));
  node.append(plot([{label: "mem", color: C.derived, points: g.mem_used}],
      {unit: " MiB", y0: 0, H: 110, xlabel: clock, breakGaps: true, markers: marks,
       reference: g.mem_total ? [{y: g.mem_total, label: "device total",
                                  color: C.dim}] : []}));
  const util = last(g.util), mem = last(g.mem_used);
  node.append(el("div", {class: "now"},
      `now ${brief(util)}% · ${gib(mem)}${g.mem_total ? " of " + gib(g.mem_total) : ""}`));
  return node;
}

function last(points) {
  return points && points.length ? points[points.length - 1][1] : null;
}

// ---- the open slot: any numeric field nothing above claims ----------------

function drawSlot(host) {
  const metrics = (host.metrics || []).filter(m =>
      !CLAIMED_EVENTS.some(prefix => m.key.startsWith(prefix)));
  if (!metrics.length) {
    document.getElementById("page").append(el("h2", {}, "journaled metrics"));
    note("none unclaimed — this page plots any numeric field a host event carries "
       + "that the readings above do not already draw");
    return;
  }
  const grid = section("journaled metrics",
      "any numeric field this host emits that no reading above claims");
  for (const m of metrics)
    grid.append(card(m.key, m.event, [{label: m.key, color: C.derived,
        points: m.points}], {xlabel: clock, breakGaps: true}));
}
