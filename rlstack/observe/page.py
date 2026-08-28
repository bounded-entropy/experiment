"""THE document: one self-contained HTML+CSS+JS page, served for every route.

No build step, no CDN, no framework — the whole UI is this string plus the JSON
the routes in ui.py serve, and JS routes on location.pathname so all four pages
(runs index, one run, the fleet, one host) are the same bytes. Three rules it
obeys: every plotted point is hoverable and the tooltip is the raw number as
journaled; GLOBAL facts render on the global pages (placement, residency,
utilization, timelines) while per-run curves stay on the run page; and a poll
never redraws a chart out from under the cursor.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>rlstack</title>
<style>
  :root { --bg:#101418; --card:#1a2027; --ink:#dde5ec; --dim:#8494a4;
          --line:#2a323c; --feed:#5fb2ff; --eval:#ffb86b; --rail:#9de08f;
          --derived:#c792ea; --warn:#e07a7a; }
  body { background:var(--bg); color:var(--ink); margin:0;
         font:14px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { padding:14px 22px; border-bottom:1px solid var(--line);
           display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  header a { color:var(--ink); text-decoration:none; font-weight:700; }
  header a.nav { color:var(--dim); font-weight:400; }
  header a.nav.on { color:var(--feed); }
  header .meta { color:var(--dim); }
  header #ctx { display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  h2 { font-size:13px; color:var(--dim); text-transform:uppercase;
       letter-spacing:.08em; margin:26px 22px 8px; }
  h2 span { text-transform:none; letter-spacing:0; opacity:.7; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
          gap:14px; padding:0 22px; }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:8px; padding:10px 12px 6px; }
  .card .name { font-weight:700; }
  .card .who { color:var(--dim); font-size:12px; float:right; }
  .card .now { color:var(--dim); font-size:12px; }
  .grid.wide { grid-template-columns:1fr; }
  .plot svg { width:100%; display:block; }
  .tl { background:var(--card); border:1px solid var(--line); border-radius:8px;
        margin:0 22px; padding:8px 10px; }
  .tl svg { width:100%; display:block; }
  .tl a text { text-decoration:none; }
  table { border-collapse:collapse; margin:6px 22px 18px; }
  td, th { padding:6px 14px 6px 0; text-align:left; border-bottom:1px solid var(--line); }
  th { color:var(--dim); font-weight:400; }
  td a { color:var(--feed); text-decoration:none; }
  td.k, span.k { color:var(--dim); }
  select { background:var(--card); color:var(--ink); border:1px solid var(--line);
           border-radius:6px; padding:3px 8px; font:inherit; font-size:12px;
           max-width:340px; }
  .legend { color:var(--dim); font-size:12px; padding:4px 22px 34px; }
  .live { color:var(--rail); }
  .warn { color:var(--warn); }
  .note { color:var(--dim); padding:2px 22px 10px; font-size:12px; }
  #tip { position:fixed; display:none; z-index:9; pointer-events:none;
         background:#0c1014f2; border:1px solid var(--line); border-radius:6px;
         padding:6px 9px; font-size:12px; line-height:1.5; white-space:nowrap;
         box-shadow:0 4px 14px #0008; }
</style>
<header id="hdr"></header>
<div id="page"></div>
<div class="legend" id="legend"></div>
<div id="tip"></div>
<script>
"use strict";
const path = location.pathname;
const runId   = path.startsWith("/run/")  ? decodeURIComponent(path.slice(5)) : null;
const hostName= path.startsWith("/host/") ? decodeURIComponent(path.slice(6)) : null;
const isFleet = path === "/hosts";
const C = {feed:"#5fb2ff", eval:"#ffb86b", rail:"#9de08f", derived:"#c792ea",
           dim:"#8494a4", warn:"#e07a7a"};
const WHEEL = ["#5fb2ff", "#9de08f", "#ffb86b", "#c792ea", "#e07a7a", "#6fd6c4"];
const STATUS_COLOR = {running:C.rail, done:C.feed, failed:C.warn};
let hovering = false;      // a poll never redraws under the cursor

// ---- small helpers --------------------------------------------------------

function el(tag, attrs, html) {
  const node = document.createElement(tag);
  for (const key in (attrs || {})) node.setAttribute(key, attrs[key]);
  if (html !== undefined) node.innerHTML = html;
  return node;
}
function esc(s) {
  return String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function fmt(v) {                        // axis labels: compact
  const a = Math.abs(v);
  return a === 0 ? "0" : a >= 1000 ? v.toFixed(0)
       : a >= 1 ? v.toFixed(2) : v.toPrecision(2);
}
function raw(v) {                        // hover: the number as journaled
  if (v === null || v === undefined) return "—";
  return Number.isInteger(v) ? String(v) : String(parseFloat(v.toPrecision(8)));
}
function brief(v) {                      // the card's own "now": four digits
  if (v === null || v === undefined) return "—";
  return Number.isInteger(v) ? String(v) : String(parseFloat(v.toPrecision(4)));
}
function when(t) {
  if (!t) return "—";
  return new Date(t * 1000).toLocaleString([], {month:"2-digit", day:"2-digit",
      hour:"2-digit", minute:"2-digit", second:"2-digit", hour12:false});
}
function clock(t) {
  if (!t) return "—";
  return new Date(t * 1000).toLocaleTimeString([], {hour:"2-digit",
      minute:"2-digit", second:"2-digit", hour12:false});
}
// #55 renamed Partition.gpuset -> .metal and Regime.kind -> .capability, but a
// journal is append-only history: pre-rename hosts still say the old words, so
// both readers take either spelling rather than blanking an older host.
function partMetal(p) { return (p && (p.metal || p.gpuset)) || "?"; }
function regimeCapability(r) { return r.capability || r.kind || "?"; }
function dur(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 90) return seconds.toFixed(0) + "s";
  if (seconds < 5400) return (seconds / 60).toFixed(1) + "m";
  return (seconds / 3600).toFixed(1) + "h";
}
function extent(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}
function showTip(event, html) {
  const tip = document.getElementById("tip");
  tip.innerHTML = html;
  tip.style.display = "block";
  const box = tip.getBoundingClientRect();
  let x = event.clientX + 16, y = event.clientY + 16;
  if (x + box.width > innerWidth - 8) x = event.clientX - box.width - 16;
  if (y + box.height > innerHeight - 8) y = event.clientY - box.height - 16;
  tip.style.left = Math.max(4, x) + "px";
  tip.style.top = Math.max(4, y) + "px";
}
function hideTip() { document.getElementById("tip").style.display = "none"; }

// ---- the chart: a polyline, and every point one cursor away from its number

function measured(build) {
  // draw at the container's own pixel width: the viewBox IS css pixels, so
  // nothing is letterboxed or stretched and the labels stay crisp
  const box = el("div", {class: "plot"});
  let drawnAt = 0;
  const render = () => {
    const width = Math.round(box.clientWidth);
    if (!width || Math.abs(width - drawnAt) < 4) return;
    drawnAt = width;
    build(box, width);
  };
  build(box, 340);       // drawn before layout, so a frozen tab still has it
  requestAnimationFrame(render);
  if (window.ResizeObserver) new ResizeObserver(render).observe(box);
  return box;
}

function plot(series, opts) {
  const o = Object.assign({H:150, pad:34, unit:"", xlabel: v => "u" + v},
                          opts || {});
  return measured((box, width) => drawPlot(box, Object.assign({W: width}, o), series));
}

function stretches(points, breakGaps) {
  // a gap in a host's samples IS downtime — never draw a line across one.
  // The cadence is whatever the journal actually held (its median step).
  if (!breakGaps || points.length < 3) return [points];
  const steps = points.slice(1).map((p, i) => p[0] - points[i][0])
                      .sort((a, b) => a - b);
  const limit = steps[Math.floor(steps.length / 2)] * 4;
  const runs = [[points[0]]];
  for (let i = 1; i < points.length; i++) {
    if (limit && points[i][0] - points[i - 1][0] > limit) runs.push([]);
    runs[runs.length - 1].push(points[i]);
  }
  return runs;
}

function drawPlot(box, o, series) {
  const drawn = (series || []).filter(s => s.points && s.points.length);
  if (!drawn.length) {
    box.innerHTML = `<svg viewBox="0 0 ${o.W} ${o.H}" style="height:${o.H}px"></svg>`;
    return;
  }
  const xs = [], ys = [];
  for (const s of drawn) for (const p of s.points) { xs.push(p[0]); ys.push(p[1]); }
  let [x0, x1] = extent(xs), [y0, y1] = extent(ys);
  if (o.y0 !== undefined) y0 = Math.min(y0, o.y0);
  if (o.y1 !== undefined) y1 = Math.max(y1, o.y1);
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  if (y0 === y1) { y0 -= Math.abs(y0) * 0.1 + 1e-3; y1 += Math.abs(y1) * 0.1 + 1e-3; }
  const sx = x => o.pad + (x - x0) / (x1 - x0) * (o.W - o.pad - 8);
  const sy = y => (o.H - 18) - (y - y0) / (y1 - y0) * (o.H - 30);
  const line = s => stretches(s.points, o.breakGaps).map(run =>
      run.length === 1
        ? `<circle cx="${sx(run[0][0]).toFixed(1)}" cy="${sy(run[0][1]).toFixed(1)}"
             r="1.6" fill="${s.color}"/>`
        : `<polyline fill="none" stroke="${s.color}" stroke-width="1.6"
             ${s.dash ? 'stroke-dasharray="5 4"' : ""} points="${
             run.map(p => sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join(" ")}"/>`
    ).join("")
    + s.points.slice(-1).map(p => `<circle cx="${sx(p[0]).toFixed(1)}"
      cy="${sy(p[1]).toFixed(1)}" r="2.5" fill="${s.color}"/>`).join("");
  box.innerHTML = `<svg viewBox="0 0 ${o.W} ${o.H}" style="height:${o.H}px">
    <text x="2" y="12" fill="${C.dim}" font-size="10">${fmt(y1)}</text>
    <text x="2" y="${o.H - 20}" fill="${C.dim}" font-size="10">${fmt(y0)}</text>
    <text x="${o.pad}" y="${o.H - 4}" fill="${C.dim}" font-size="10">${o.xlabel(x0)}</text>
    <text x="${o.W - 8}" y="${o.H - 4}" fill="${C.dim}" font-size="10"
          text-anchor="end">${o.xlabel(x1)}</text>
    <line x1="${o.pad}" y1="${o.H - 18}" x2="${o.W - 8}" y2="${o.H - 18}" stroke="${C.dim}33"/>
    ${drawn.map(line).join("")}
    <line class="cross" y1="4" y2="${o.H - 18}" stroke="${C.dim}" stroke-width="0.7" opacity="0"/>
    ${drawn.map(s => `<circle class="mk" r="3.4" fill="${s.color}" opacity="0"/>`).join("")}
    <rect x="0" y="0" width="${o.W}" height="${o.H}" fill="transparent"/></svg>`;

  const svg = box.querySelector("svg");
  const cross = svg.querySelector(".cross");
  const marks = Array.from(svg.querySelectorAll(".mk"));
  svg.addEventListener("mousemove", event => {
    hovering = true;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const at = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    const cursor = x0 + (at.x - o.pad) / (o.W - o.pad - 8) * (x1 - x0);
    let head = null;
    const rows = [];
    drawn.forEach((s, i) => {
      let best = null;
      for (const p of s.points)
        if (best === null || Math.abs(p[0] - cursor) < Math.abs(best[0] - cursor)) best = p;
      marks[i].setAttribute("cx", sx(best[0]).toFixed(1));
      marks[i].setAttribute("cy", sy(best[1]).toFixed(1));
      marks[i].setAttribute("opacity", 1);
      if (head === null) head = best[0];
      const elsewhere = best[0] === head ? ""
        : ` <span class="k">@${esc(o.xlabel(best[0]))}</span>`;
      rows.push(`<div><span style="color:${s.color}">${esc(s.label)}</span> `
                + raw(best[1]) + esc(o.unit) + elsewhere + "</div>");
    });
    cross.setAttribute("x1", sx(head).toFixed(1));
    cross.setAttribute("x2", sx(head).toFixed(1));
    cross.setAttribute("opacity", 1);
    showTip(event, `<div class="k">${esc(o.xlabel(head))}</div>` + rows.join(""));
  });
  svg.addEventListener("mouseleave", () => {
    hovering = false; hideTip();
    cross.setAttribute("opacity", 0);
    marks.forEach(m => m.setAttribute("opacity", 0));
  });
}

function card(title, who, series, opts) {
  const node = el("div", {class: "card"});
  node.append(el("div", {}, `<span class="name">${esc(title)}</span>`
                          + `<span class="who">${esc(who)}</span>`));
  node.append(plot(series, opts));
  const primary = (series || []).find(s => s.points && s.points.length);
  const last = primary ? primary.points[primary.points.length - 1] : null;
  node.append(el("div", {class: "now"}, last === null
      ? "no data yet" : "now " + brief(last[1]) + ((opts && opts.unit) || "")));
  return node;
}

function section(title, note, wide) {
  const holder = document.getElementById("page");
  holder.append(el("h2", {}, esc(title) + (note ? ` <span>${esc(note)}</span>` : "")));
  const grid = el("div", {class: wide ? "grid wide" : "grid"});
  holder.append(grid);
  return grid;
}

// ---- the timeline: one lane per host (fleet) or per residency (host) ------

function timeline(lanes, window_, describe) {
  const wrap = el("div", {class: "tl"});
  wrap.append(measured((box, width) =>
      drawTimeline(box, width, lanes, window_, describe)));
  return wrap;
}

function drawTimeline(box, W, lanes, window_, describe) {
  const PAD = Math.min(168, Math.round(W * 0.28)), ROW = 22;
  const H = lanes.length * ROW + 26;
  let t0 = window_ ? window_[0] : 0, t1 = window_ ? window_[1] : 1;
  if (!(t1 > t0)) { t1 = t0 + 1; }
  const sx = t => PAD + (Math.min(Math.max(t, t0), t1) - t0) / (t1 - t0) * (W - PAD - 14);
  const bars = [];
  let body = "";
  lanes.forEach((lane, row) => {
    const y = row * ROW + 6;
    const label = lane.href
      ? `<a href="${esc(lane.href)}"><text x="4" y="${y + 13}" fill="${C.feed}"
           font-size="11">${esc(lane.name)}</text></a>`
      : `<text x="4" y="${y + 13}" fill="#dde5ec" font-size="11">${esc(lane.name)}</text>`;
    body += label + `<line x1="${PAD}" y1="${y + 16}" x2="${W - 14}" y2="${y + 16}"
                       stroke="${C.dim}22"/>`;
    for (const bar of (lane.bars || [])) {
      const a = sx(bar.t0 === null || bar.t0 === undefined ? t0 : bar.t0);
      const b = sx(bar.t1 === null || bar.t1 === undefined ? t1 : bar.t1);
      const color = STATUS_COLOR[bar.status] || C.dim;
      body += `<rect class="bar" x="${a.toFixed(1)}" y="${y + 3}"
                 width="${Math.max(2, b - a).toFixed(1)}" height="10" rx="2"
                 fill="${color}" opacity="0.75"/>`;
      bars.push(bar);
    }
  });
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="height:${H}px">
    ${body}
    <text x="${PAD}" y="${H - 4}" fill="${C.dim}" font-size="10">${esc(when(t0))}</text>
    <text x="${W - 14}" y="${H - 4}" fill="${C.dim}" font-size="10"
          text-anchor="end">${esc(when(t1))}</text></svg>`;
  Array.from(box.querySelectorAll("rect.bar")).forEach((rect, i) => {
    rect.addEventListener("mousemove", event => {
      hovering = true; showTip(event, describe(bars[i]));
    });
    rect.addEventListener("mouseleave", () => { hovering = false; hideTip(); });
  });
}

function residencyTip(bar) {
  const span = (bar.t1 || Date.now() / 1000) - (bar.t0 || 0);
  return `<div><b>${esc(bar.label)}</b> <span class="k">${esc(bar.status)}</span></div>`
    + `<div><span class="k">from</span> ${esc(when(bar.t0))}</div>`
    + `<div><span class="k">to</span>   ${bar.t1 ? esc(when(bar.t1)) : "resident"}</div>`
    + `<div><span class="k">for</span>  ${esc(dur(bar.t0 ? span : null))}</div>`
    + (bar.detail ? `<div class="k">${esc(bar.detail)}</div>` : "");
}

// ---- header + the experiment switcher -------------------------------------

function nav() {
  const hdr = document.getElementById("hdr");
  hdr.innerHTML = `<a href="/">rlstack</a>`
    + `<a class="nav${runId || path === "/" ? " on" : ""}" href="/">runs</a>`
    + `<a class="nav${isFleet || hostName ? " on" : ""}" href="/hosts">hosts</a>`
    + `<span id="ctx"></span>`;
  if (runId) {
    const sel = el("select", {id: "switch", title: "switch experiment"});
    sel.addEventListener("change", () => {
      if (sel.value && sel.value !== runId) location.pathname = "/run/" + sel.value;
    });
    hdr.append(sel);
  }
}

let switcherSignature = null;
function syncSwitcher(runs) {
  // rebuilt only when the run list itself changes — a poll must not close
  // an open dropdown
  const sel = document.getElementById("switch");
  if (!sel) return;
  const known = runs.some(r => r.run_id === runId);
  const options = (known ? runs : [{run_id: runId, status: "?", committed: 0,
                                    target: "?"}].concat(runs));
  const signature = options.map(r => `${r.run_id}:${r.status}:${r.committed}`).join("|");
  if (signature === switcherSignature) return;
  switcherSignature = signature;
  sel.innerHTML = "";
  for (const r of options.slice().reverse()) {
    const opt = el("option", {value: r.run_id},
        `${esc(r.run_id)} · ${esc(r.status)} ${r.committed}/${esc(r.target)}`);
    sel.append(opt);
  }
  sel.value = runId;
}

// ---- page: one run --------------------------------------------------------

async function drawRun() {
  const [res, runsRes] = await Promise.all([
    fetch("/api/run/" + encodeURIComponent(runId)), fetch("/api/runs")]);
  if (!res.ok) { document.getElementById("page").textContent = "unknown run"; return; }
  const data = await res.json();
  const runs = runsRes.ok ? await runsRes.json() : [];
  syncSwitcher(runs);
  const mine = runs.find(r => r.run_id === runId);
  const dict = data.dictionary || {columns: [], rails: []};
  document.getElementById("ctx").innerHTML =
    `<span class="meta">loss ${esc(dict.loss ?? "?")} · lag ${esc(dict.max_policy_lag ?? 0)}
     · post [${(dict.post_pipeline || []).map(esc).join(" → ")}]</span>
     <span class="${data.committed >= (data.target ?? 1e9) ? "meta" : "live"}">
     ${data.committed}/${esc(data.target ?? "?")} committed</span>`
    + (mine ? `<span class="meta">on ${mine.hosts.map(h =>
        `<a href="/host/${encodeURIComponent(h)}" style="color:${C.feed}">${esc(h)}</a>`
      ).join(" + ") || "?"}</span>` : "");
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
  const rails = section("rails");
  for (const r of (dict.rails || []))
    rails.append(card(r, "loss:" + (dict.loss ?? "?"),
                      [{label: r, color: C.rail, points: trainOf("train", r)}]));
  const derived = data.derived || [];
  if (derived.length) {
    const grid = section("derived", "your panels.json");
    for (const d of derived) {
      if (d.missing || d.error) {
        const node = el("div", {class: "card"});
        node.append(el("div", {}, `<span class="name">${esc(d.name)}</span>`
                                + `<span class="who">${esc(d.expr)}</span>`));
        node.append(el("div", {class: "now warn"}, esc(d.error ? d.error
            : "missing from this run's pipeline: " + d.missing.join(", "))));
        grid.append(node);
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
  document.getElementById("legend").innerHTML =
    "solid = training · <span style='color:#ffb86b'>dashed = held-out eval</span>"
    + " · hover any chart for the raw values · refreshes every 3s"
    + " · panels & priority from the run's own dictionary.json";
}

// ---- page: the run index --------------------------------------------------

async function drawIndex() {
  const res = await fetch("/api/runs");
  const runs = await res.json();
  const holder = document.getElementById("page");
  holder.innerHTML = "";
  document.getElementById("ctx").innerHTML =
    `<span class="meta">${runs.length} experiment${runs.length === 1 ? "" : "s"}</span>`;
  if (!runs.length) {
    holder.append(el("p", {style: "padding:22px"},
        "no experiments journaled in this store yet"));
    return;
  }
  const table = el("table", {}, "<tr><th>run</th><th>status</th><th>committed</th>"
      + "<th>host(s)</th><th>store</th></tr>");
  for (const r of runs.slice().reverse()) {
    const row = el("tr", {});
    row.append(el("td", {}, `<a href="/run/${encodeURIComponent(r.run_id)}">${esc(r.run_id)}</a>`));
    row.append(el("td", {class: r.status === "running" ? "live" : ""},
                  esc(r.status) + (r.forked ? " <span class='warn'>⚠FORK</span>" : "")));
    row.append(el("td", {}, `${r.committed}/${esc(r.target)}`));
    row.append(el("td", {}, r.hosts.map(h =>
        `<a href="/host/${encodeURIComponent(h)}">${esc(h)}</a>`).join(" + ")));
    row.append(el("td", {class: "k"}, esc(r.store)));
    table.append(row);
  }
  holder.append(table);
  document.getElementById("legend").innerHTML =
    "one experiment, one store (I10) · ⚠FORK = the same run_id in two stores"
    + " · hosts link to their journals · refreshes every 3s";
}

// ---- page: the fleet ------------------------------------------------------

async function drawFleet() {
  const res = await fetch("/api/hosts");
  const fleet = await res.json();
  const holder = document.getElementById("page");
  holder.innerHTML = "";
  document.getElementById("ctx").innerHTML =
    `<span class="meta">${fleet.hosts.length} host${fleet.hosts.length === 1 ? "" : "s"}
     · ${fleet.runs.length} experiment${fleet.runs.length === 1 ? "" : "s"}
     · ${fleet.stores.map(esc).join(" ")}</span>`;
  if (!fleet.hosts.length) {
    holder.append(el("p", {style: "padding:22px"},
        "no hosts journaled in this store yet"));
    return;
  }
  holder.append(el("h2", {}, "placement <span>who ran where, over time</span>"));
  holder.append(timeline(fleet.hosts.map(h => ({
      name: h.host, href: "/host/" + encodeURIComponent(h.host),
      bars: h.tenancy.map(t => ({label: t.run_id, status: t.status,
          t0: t.attached, t1: t.detached,
          detail: (t.pools || []).join(", ")}))})),
    fleet.window, residencyTip));

  const busy = fleet.hosts.filter(h => h.util.length);
  if (busy.length) {
    const grid = section("utilization", "busiest device per host", true);
    grid.append(card("gpu util", busy.map(h => h.host).join(" · "),
        busy.map((h, i) => ({label: h.host, color: WHEEL[i % WHEEL.length],
                             points: h.util})),
        {unit: "%", y0: 0, y1: 100, xlabel: clock, breakGaps: true, H: 200}));
  }
  const table = el("table", {}, "<tr><th>host</th><th>engines</th><th>regimes</th>"
      + "<th>partition</th><th>tenants</th><th>boots</th><th>last seen</th></tr>");
  for (const h of fleet.hosts) {
    const row = el("tr", {});
    row.append(el("td", {}, `<a href="/host/${encodeURIComponent(h.host)}">${esc(h.host)}</a>`));
    row.append(el("td", {}, esc(h.engines.join(", ") || "?")));
    row.append(el("td", {}, (h.regimes || []).map(r =>
        esc(`${r.name}:${regimeCapability(r)}×${r.shape}`)).join(" ") || "<span class='k'>—</span>"));
    row.append(el("td", {}, h.partition
        ? esc(`${partMetal(h.partition)} [${(h.partition.devices || []).join(",")}] `
              + `mem ${h.partition.memory}`)
        : "<span class='k'>—</span>"));
    row.append(el("td", {}, `<span class="${h.running.length ? "live" : "k"}">`
        + `${h.running.length} running</span> <span class="k">${h.done} done, `
        + `${h.failed} failed</span>`));
    row.append(el("td", {class: "k"}, String(h.boots)));
    row.append(el("td", {class: "k"}, esc(when(h.last_seen))));
    table.append(row);
  }
  holder.append(el("h2", {}, "hosts"));
  holder.append(table);
  document.getElementById("legend").innerHTML =
    "every fact here is a host journal (hosts/&lt;name&gt;/log.jsonl) — the observer"
    + " never attaches · <span style='color:#9de08f'>green = resident</span>,"
    + " <span style='color:#5fb2ff'>blue = done</span>,"
    + " <span style='color:#e07a7a'>red = failed</span> · hover a bar or a point"
    + " for the raw values";
}

// ---- page: one host -------------------------------------------------------

async function drawHost() {
  const res = await fetch("/api/host/" + encodeURIComponent(hostName));
  if (!res.ok) { document.getElementById("page").textContent = "unknown host"; return; }
  const host = await res.json();
  const holder = document.getElementById("page");
  holder.innerHTML = "";
  const resident = host.tenancy.filter(t => t.detached === null).length;
  document.getElementById("ctx").innerHTML =
    `<span>${esc(host.host)}</span>`
    + `<span class="meta">engines ${esc(host.engines.join(", ") || "?")}`
    + ` · ${host.boots.length} boot${host.boots.length === 1 ? "" : "s"}`
    + ` · ${host.events} events · ${esc(host.journal_stores.map(esc).join(" "))}</span>`
    + `<span class="${resident ? "live" : "meta"}">${resident} resident</span>`;

  holder.append(el("h2", {}, "the host <span>as it attested itself at birth</span>"));
  const facts = el("table", {});
  const partition = host.partition
    ? `${esc(partMetal(host.partition))} devices [${(host.partition.devices || []).join(", ")}]`
      + ` · memory fraction ${esc(host.partition.memory)}`
    : "<span class='k'>not journaled (a host older than #43's partitions)</span>";
  const regimes = host.regimes.length
    ? host.regimes.map(r => esc(`${r.name} = ${regimeCapability(r)} × ${r.base ?? "*"} × ${r.shape}`)).join("<br>")
    : "<span class='k'>not journaled (a host older than #43's regimes)</span>";
  facts.innerHTML =
    `<tr><th>partition</th><td>${partition}</td></tr>`
    + `<tr><th>regimes</th><td>${regimes}</td></tr>`
    + `<tr><th>engines</th><td>${esc(host.engines.join(", ") || "?")}</td></tr>`
    + `<tr><th>seen</th><td>${esc(when(host.first_seen))} → ${esc(when(host.last_seen))}`
    + ` <span class="k">(${esc(dur(host.last_seen - host.first_seen))})</span></td></tr>`;
  holder.append(facts);

  holder.append(el("h2", {}, "tenancy <span>attach → detach, from the journal</span>"));
  if (host.tenancy.length) {
    holder.append(timeline(host.tenancy.map(t => ({
        name: t.run_id, href: "/run/" + encodeURIComponent(t.run_id),
        bars: [{label: t.run_id, status: t.status, t0: t.attached, t1: t.detached,
                detail: (t.pools || []).join(", ")}]})),
      [host.first_seen, host.last_seen], residencyTip));
    const table = el("table", {}, "<tr><th>run</th><th>status</th><th>pools</th>"
        + "<th>remote pools</th><th>updates</th><th>attached</th><th>for</th>"
        + "<th>store</th></tr>");
    for (const t of host.tenancy.slice().reverse()) {
      const row = el("tr", {});
      row.append(el("td", {}, `<a href="/run/${encodeURIComponent(t.run_id)}">${esc(t.run_id)}</a>`));
      row.append(el("td", {class: t.status === "running" ? "live" : ""}, esc(t.status)));
      row.append(el("td", {class: "k"}, esc(t.pools.join(", ") || "—")));
      row.append(el("td", {class: "k"}, esc(t.remotes.join(", ") || "—")));
      row.append(el("td", {}, esc(`${t.updates_completed ?? "?"}/${t.n_updates ?? "?"}`)));
      row.append(el("td", {class: "k"}, esc(when(t.attached))));
      row.append(el("td", {class: "k"}, esc(t.attached
          ? dur((t.detached || Date.now() / 1000) - t.attached) : "—")));
      row.append(el("td", {class: "k"}, esc(t.store ?? "—")));
      table.append(row);
    }
    holder.append(table);
  } else {
    holder.append(el("p", {class: "note"}, "no tenant has ever attached here"));
  }

  if (host.gpus.length) {
    const grid = section("gpu", "one line per device, from the stats events");
    grid.append(card("utilization", host.gpus.length + " device(s)",
        host.gpus.map(g => ({label: "gpu" + g.device,
                             color: WHEEL[g.device % WHEEL.length], points: g.util})),
        {unit: "%", y0: 0, y1: 100, xlabel: clock, breakGaps: true}));
    grid.append(card("memory used",
        host.gpus.map(g => g.mem_total ? g.mem_total + " MiB" : "?").join(" / "),
        host.gpus.map(g => ({label: "gpu" + g.device,
                             color: WHEEL[g.device % WHEEL.length], points: g.mem_used})),
        {unit: " MiB", y0: 0, xlabel: clock, breakGaps: true}));
  } else {
    holder.append(el("h2", {}, "gpu"));
    holder.append(el("p", {class: "note"},
        "no stats events journaled (run_stats not running, or no NVIDIA runtime)"));
  }

  const metrics = host.metrics || [];
  if (metrics.length) {
    const grid = section("journaled metrics", "any numeric field this host emits");
    for (const m of metrics)
      grid.append(card(m.key, m.event, [{label: m.key, color: C.derived,
                                         points: m.points}], {xlabel: clock, breakGaps: true}));
  } else {
    holder.append(el("h2", {}, "journaled metrics"));
    holder.append(el("p", {class: "note"},
        "none yet — this page plots any numeric field a host event carries that "
        + "the readings above do not claim, so throughput (and whatever comes "
        + "after it) appears here the day a host journals it"));
  }
  document.getElementById("legend").innerHTML =
    "this page is one file: hosts/" + esc(hostName) + "/log.jsonl · hover any bar"
    + " or point for its raw values · refreshes every 3s";
}

// ---- polling --------------------------------------------------------------

const draw = runId ? drawRun : hostName ? drawHost : isFleet ? drawFleet : drawIndex;
nav();
draw();
setInterval(() => { if (!hovering) draw(); }, 3000);   // never under the cursor
</script>
"""
