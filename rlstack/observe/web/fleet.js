// The fleet: what is true GLOBALLY — who ran where, what the whole fleet is
// producing, and how busy the metal is. Per-run curves stay on the run page
// (#50's rule); the two aggregates here belong to no single host or run.
"use strict";

import {C, WHEEL, ago, brief, clock, drawnOnce, el, esc, getJSON, lostTick,
        okTick, pulseDot, raw, section, storeTail, when} from "./dom.js";
import {card, emptyCard, residencyTip, timeline} from "./charts.js";
import {lazyDetails} from "./lazy.js";
import {ctx, hostPath, withHours, route, query} from "./nav.js";

const openHosts = new Set();

export async function drawFleet() {
  const page = await getJSON("/api/hosts/index" + query(route.folder));
  if (!page) {
    if (drawnOnce()) { lostTick(); return; }
    document.getElementById("page").textContent = "Could not load hosts — press r to retry.";
    return;
  }
  const holder = document.getElementById("page");
  holder.replaceChildren();
  ctx(`<span class="meta">${page.hosts.length} hosts · expand one to load its details</span>`);
  for (const h of page.hosts) {
    const key = JSON.stringify([h.folder, h.host]);
    const label = h.folder ? h.folder + "/" + h.host : h.host;
    holder.append(lazyDetails(esc(label), async body => {
      const host = await getJSON(withHours("/api/host/" + encodeURIComponent(h.host) + "/summary" + query(h.folder)));
      if (!host) throw new Error("Host read failed");
      body.replaceChildren();
      body.append(el("p", {}, pulseDot(host.pulse, host.now)
        + ` <a href="${hostPath(h.host, h.folder)}">Open host charts and history</a>`));
      const attached = host.open_attachments;
      body.append(el("p", {class: "k"}, `${attached} open journaled attachments · `
        + `${host.events} events · last seen ${esc(ago(host.last_seen, host.now))}`));
      body.append(el("p", {}, `Engines: ${esc(host.engines.join(", ") || "unknown")}`));
    }, key, openHosts));
  }
  holder.append(lazyDetails("Fleet-wide charts and placement", drawFleetHistory, "fleet-history", openHosts));
  okTick();
}

function localSection(holder, title, kind) {
  holder.append(el("h2", {}, esc(title)));
  const grid = el("div", {class: "grid " + kind});
  holder.append(grid);
  return grid;
}

async function drawFleetHistory(holder) {
  const page = await getJSON(withHours("/api/fleet/page"));   // ONE request
  const fleet = page && page.fleet, flow = page && page.flow;
  if (!fleet) {                        // the freshness contract (dom.js)
    throw new Error("Fleet read failed");
  }
  holder.innerHTML = "";
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
  /* Full fleet statistics are requested explicitly below. */
  holder.append(el("p", {}, `<span class="meta">${fleet.hosts.length} host${fleet.hosts.length === 1 ? "" : "s"}
     · ${fleet.runs.length} experiment${fleet.runs.length === 1 ? "" : "s"}</span>`
    + `<span class="meta" title="${fleet.stores.map(esc).join(" ")}">`
    + `${fleet.stores.map(s => esc(storeTail(s))).join(" ")}</span>`));
  if (!fleet.hosts.length) {
    holder.append(el("p", {style: "padding:22px"},
        "no hosts journaled in this store yet"));
    return;
  }
  holder.append(el("h2", {}, "placement"));
  holder.append(timeline(fleet.hosts.map(h => ({
      name: label(h), href: link(h),
      bars: h.tenancy.map(t => ({label: t.run_id, status: t.status,
          t0: t.attached, t1: t.detached,
          detail: (t.pools || []).join(", ")}))})),
    fleet.window, residencyTip));

  drawAggregates(flow, fleet.now, holder);

  const busy = fleet.hosts.filter(h => h.util.length);
  if (busy.length) {
    const grid = localSection(holder, "utilization", "wide");
    grid.append(card("gpu util", busy.map(label).join(" · "),
        busy.map((h, i) => ({label: label(h), color: WHEEL[i % WHEEL.length],
                             points: h.util})),
        {unit: "%", y0: 0, y1: 100, xlabel: clock, breakGaps: true, H: 200,
         asOf: fleet.now, freshS: 60}));
  }
  const table = el("table", {}, "<tr><th></th><th>host</th><th>engines</th><th>regimes</th>"
      + "<th>partition</th><th>residents</th><th>tenants</th><th>epoch</th>"
      + "<th>boots</th><th>last seen</th></tr>");
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
    // the processes the host was born with (ADR 0002): one per regime, label + pid
    row.append(el("td", {class: "k"}, (h.residents || []).length
        ? h.residents.map(r => esc(`${r.label} · pid ${r.pid}`)).join("<br>")
        : "<span class='k'>—</span>"));
    const stalled = h.tenancy.filter(t => t.status === "stalled").length;
    const parked = h.tenancy.filter(t => t.status === "parked").length;
    const active = h.running.length - stalled;
    row.append(el("td", {},
        `<span class="${active > 0 ? "live" : "k"}">${Math.max(0, active)} running</span> `
      + (stalled ? `<span class="stall">${stalled} stalled</span> ` : "")
      + (parked ? `<span class="stall">${parked} parked</span> ` : "")
      + `<span class="k">${h.done} done, ${h.failed} failed</span>`));
    // WHICH LIFE, and whether the desk could reach it (ADR 0008, F1/F2)
    row.append(el("td", {class: "k"},
        (h.epoch ? `<span title="the boot this journal's last host-up was">`
                 + `${esc(h.epoch)}</span>` : "—")
      + (h.unreachable ? ` <span class="warn" title="the desk's last ask of `
          + `this ${esc(h.unreachable.kind)} outlived its `
          + `${esc(String(h.unreachable.deadline_s))}s deadline">unreachable`
          + `</span>` : "")));
    row.append(el("td", {class: "k"}, String(h.boots)));
    row.append(el("td", {class: "k when", title: esc(when(h.last_seen))},
                  esc(ago(h.last_seen, fleet.now))));
    table.append(row);
  }
  holder.append(el("h2", {}, "hosts"));
  holder.append(table);
  okTick();
}

// ---- the two aggregates: one per kind of partition ------------------------

function drawAggregates(flow, asOf, holder) {
  const inference = (flow || {}).inference || [];
  const training = (flow || {}).training || [];
  const grid = localSection(holder, "fleet throughput", "half");
  if (!inference.length && !training.length) {
    grid.append(emptyCard("inference", "traffic events",
        "no traffic events journaled yet"));
    grid.append(emptyCard("training", "update events",
        "no update events journaled yet"));
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

