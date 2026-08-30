// The fleet: what is true GLOBALLY — who ran where, what the whole fleet is
// producing, and how busy the metal is. Per-run curves stay on the run page
// (#50's rule); the two aggregates here belong to no single host or run.
"use strict";

import {C, WHEEL, ago, brief, clock, el, esc, getJSON, pulseDot, raw, section,
        when} from "./dom.js";
import {card, emptyCard, residencyTip, timeline} from "./charts.js";
import {ctx, hostPath, legend, withHours} from "./nav.js";

export async function drawFleet() {
  const [fleet, flow] = await Promise.all([
    getJSON(withHours("/api/hosts")), getJSON(withHours("/api/fleet"))]);
  const holder = document.getElementById("page");
  holder.innerHTML = "";
  if (!fleet) { holder.textContent = "no stores"; return; }
  // a host name is unique per store, not per fleet: when two roots journal
  // the same name, the folder is what tells them apart
  const shared = new Set();
  const seen = new Set();
  for (const h of fleet.hosts) {
    if (seen.has(h.host)) shared.add(h.host);
    seen.add(h.host);
  }
  const label = h => shared.has(h.host) ? `${h.folder || "(top)"}/${h.host}` : h.host;
  const link = h => hostPath(h.host, shared.has(h.host) ? h.folder : null);
  ctx(`<span class="meta">${fleet.hosts.length} host${fleet.hosts.length === 1 ? "" : "s"}
     · ${fleet.runs.length} experiment${fleet.runs.length === 1 ? "" : "s"}
     · ${fleet.stores.map(esc).join(" ")}</span>`);
  if (!fleet.hosts.length) {
    holder.append(el("p", {style: "padding:22px"},
        "no hosts journaled in this store yet"));
    return;
  }
  holder.append(el("h2", {}, "placement <span>who ran where, over time</span>"));
  holder.append(timeline(fleet.hosts.map(h => ({
      name: label(h), href: link(h),
      bars: h.tenancy.map(t => ({label: t.run_id, status: t.status,
          t0: t.attached, t1: t.detached,
          detail: (t.pools || []).join(", ")}))})),
    fleet.window, residencyTip));

  drawAggregates(flow, fleet.now);

  const busy = fleet.hosts.filter(h => h.util.length);
  if (busy.length) {
    const grid = section("utilization", "busiest device per host", "wide");
    grid.append(card("gpu util", busy.map(label).join(" · "),
        busy.map((h, i) => ({label: label(h), color: WHEEL[i % WHEEL.length],
                             points: h.util})),
        {unit: "%", y0: 0, y1: 100, xlabel: clock, breakGaps: true, H: 200,
         asOf: fleet.now, freshS: 60}));
  }
  const table = el("table", {}, "<tr><th></th><th>host</th><th>engines</th><th>regimes</th>"
      + "<th>partition</th><th>tenants</th><th>boots</th><th>last seen</th></tr>");
  for (const h of fleet.hosts) {
    const row = el("tr", {});
    row.append(el("td", {}, pulseDot(h.pulse, fleet.now)));
    row.append(el("td", {}, `<a href="${link(h)}">${esc(label(h))}</a>`));
    row.append(el("td", {}, esc(h.engines.join(", ") || "?")));
    row.append(el("td", {}, (h.regimes || []).map(r =>
        esc(`${r.name}:${r.capability ?? r.kind}×${r.shape}`)).join(" ") || "<span class='k'>—</span>"));
    row.append(el("td", {}, h.partition
        ? esc(`${h.partition.metal ?? h.partition.gpuset} [${(h.partition.devices || []).join(",")}] `
              + `mem ${h.partition.memory}`)
        : "<span class='k'>—</span>"));
    const stalled = h.tenancy.filter(t => t.status === "stalled").length;
    const active = h.running.length - stalled;
    row.append(el("td", {},
        `<span class="${active > 0 ? "live" : "k"}">${Math.max(0, active)} running</span> `
      + (stalled ? `<span class="stall">${stalled} stalled</span> ` : "")
      + `<span class="k">${h.done} done, ${h.failed} failed</span>`));
    row.append(el("td", {class: "k"}, String(h.boots)));
    row.append(el("td", {class: "k", title: esc(when(h.last_seen))},
                  esc(ago(h.last_seen, fleet.now))));
    table.append(row);
  }
  document.getElementById("page").append(el("h2", {}, "hosts"));
  document.getElementById("page").append(table);
  legend("facts are host journals; ● liveness is a fleet-service PROBE where one"
    + " is connected, a journal-heartbeat PRESUMPTION otherwise (hover the dot)"
    + " · <span style='color:#9de08f'>green = resident</span>,"
    + " <span style='color:#5fb2ff'>blue = done</span>,"
    + " <span style='color:#ffb86b'>orange = stalled (running, host has no pulse)</span>,"
    + " <span style='color:#e07a7a'>red = failed</span> · aggregates sum per"
    + " bucket · hover anything for raw values");
}

// ---- the two aggregates: one per kind of partition ------------------------

function drawAggregates(flow, asOf) {
  const inference = (flow || {}).inference || [];
  const training = (flow || {}).training || [];
  const grid = section("fleet throughput",
      "summed per bucket — each host's own mean rate, added across hosts",
      "half");
  if (!inference.length && !training.length) {
    grid.append(emptyCard("inference", "traffic events",
        "no traffic events journaled yet — a serving host emits one per stats tick"));
    grid.append(emptyCard("training", "update events",
        "no update events journaled yet — a training host emits one per "
        + "completed update"));
    return;
  }
  grid.append(card("inference · tokens/s",
      `${(flow.hosts || []).length} serving host(s) · ${
        brief(flow.inference_bucket_s)}s buckets`,
      [{label: "prefill", color: C.feed,
        points: inference.map(b => [b.t, b.prefill_tok_s,
            `${raw(b.prefill_tokens)} prefill tokens · ${b.hosts} host(s)`
            + ` · ${b.samples} sample(s)`])},
       {label: "decode", color: C.rail,
        points: inference.map(b => [b.t, b.decode_tok_s,
            `${raw(b.decode_tokens)} decode tokens · ${raw(b.requests)} requests`])}],
      {unit: " tok/s", y0: 0, xlabel: clock, breakGaps: true, H: 190,
       asOf: asOf, freshS: flow.inference_bucket_s}));
  grid.append(card("training · updates/s",
      `${(flow.runs || []).length} run(s) · ${
        brief(flow.training_bucket_s)}s buckets`,
      [{label: "updates/s", color: C.derived,
        points: training.map(b => [b.t, b.updates_s,
            `${b.updates} update(s) from ${b.runs} run(s) · mean `
            + `${brief(b.seconds_mean)}s ${phaseWords(b.phases)}`])}],
      {unit: "/s", y0: 0, xlabel: clock, breakGaps: true, H: 190,
       asOf: asOf, freshS: flow.training_bucket_s}));
}

function phaseWords(phases) {
  return "(" + Object.entries(phases || {})
      .filter(([, v]) => v !== null && v !== undefined)
      .map(([k, v]) => `${k} ${brief(v)}s`).join(" · ") + ")";
}

