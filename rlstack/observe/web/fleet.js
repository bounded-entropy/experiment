// The fleet: what is true GLOBALLY — who ran where, what the whole fleet is
// producing, and how busy the metal is. Per-run curves stay on the run page
// (#50's rule); the two aggregates here belong to no single host or run.
"use strict";

import {C, WHEEL, brief, clock, el, esc, getJSON, raw, section, when} from "./dom.js";
import {card, emptyCard, residencyTip, timeline} from "./charts.js";
import {ctx, hostPath, legend} from "./nav.js";

export async function drawFleet() {
  const [fleet, flow] = await Promise.all([
    getJSON("/api/hosts"), getJSON("/api/fleet")]);
  const holder = document.getElementById("page");
  holder.innerHTML = "";
  if (!fleet) { holder.textContent = "no stores"; return; }
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
      name: h.host, href: hostPath(h.host),
      bars: h.tenancy.map(t => ({label: t.run_id, status: t.status,
          t0: t.attached, t1: t.detached,
          detail: (t.pools || []).join(", ")}))})),
    fleet.window, residencyTip));

  drawAggregates(flow);

  const busy = fleet.hosts.filter(h => h.util.length);
  if (busy.length) {
    const grid = section("utilization", "busiest device per host", "wide");
    grid.append(card("gpu util", busy.map(h => h.host).join(" · "),
        busy.map((h, i) => ({label: h.host, color: WHEEL[i % WHEEL.length],
                             points: h.util})),
        {unit: "%", y0: 0, y1: 100, xlabel: clock, breakGaps: true, H: 200}));
  }
  const table = el("table", {}, "<tr><th>host</th><th>engines</th><th>regimes</th>"
      + "<th>partition</th><th>tenants</th><th>boots</th><th>last seen</th></tr>");
  for (const h of fleet.hosts) {
    const row = el("tr", {});
    row.append(el("td", {}, `<a href="${hostPath(h.host)}">${esc(h.host)}</a>`));
    row.append(el("td", {}, esc(h.engines.join(", ") || "?")));
    row.append(el("td", {}, (h.regimes || []).map(r =>
        esc(`${r.name}:${r.kind}×${r.shape}`)).join(" ") || "<span class='k'>—</span>"));
    row.append(el("td", {}, h.partition
        ? esc(`${h.partition.gpuset} [${(h.partition.devices || []).join(",")}] `
              + `mem ${h.partition.memory}`)
        : "<span class='k'>—</span>"));
    row.append(el("td", {}, `<span class="${h.running.length ? "live" : "k"}">`
        + `${h.running.length} running</span> <span class="k">${h.done} done, `
        + `${h.failed} failed</span>`));
    row.append(el("td", {class: "k"}, String(h.boots)));
    row.append(el("td", {class: "k"}, esc(when(h.last_seen))));
    table.append(row);
  }
  document.getElementById("page").append(el("h2", {}, "hosts"));
  document.getElementById("page").append(table);
  legend("every fact here is a host journal (hosts/&lt;name&gt;/log.jsonl) — the observer"
    + " never attaches · <span style='color:#9de08f'>green = resident</span>,"
    + " <span style='color:#5fb2ff'>blue = done</span>,"
    + " <span style='color:#e07a7a'>red = failed</span> · the two aggregates sum"
    + " per bucket: each host's own mean rate, summed across hosts · hover a bar"
    + " or a point for the raw values");
}

// ---- the two aggregates: one per kind of partition ------------------------

function drawAggregates(flow) {
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
      {unit: " tok/s", y0: 0, xlabel: clock, breakGaps: true, H: 190}));
  grid.append(card("training · updates/s",
      `${(flow.runs || []).length} run(s) · ${
        brief(flow.training_bucket_s)}s buckets`,
      [{label: "updates/s", color: C.derived,
        points: training.map(b => [b.t, b.updates_s,
            `${b.updates} update(s) from ${b.runs} run(s) · mean `
            + `${brief(b.seconds_mean)}s ${phaseWords(b.phases)}`])}],
      {unit: "/s", y0: 0, xlabel: clock, breakGaps: true, H: 190}));
}

function phaseWords(phases) {
  return "(" + Object.entries(phases || {})
      .filter(([, v]) => v !== null && v !== undefined)
      .map(([k, v]) => `${k} ${brief(v)}s`).join(" · ") + ")";
}

