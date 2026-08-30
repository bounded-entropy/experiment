// The charts page: many runs, one metric, one overlaid graph — the reading
// for "which of my forty arms is doing anything".
//
// The query box speaks the shared grammar (regex terms AND, "|" for OR) and
// the server does the matching (observe/select.py), so the box and the runs
// page filter mean the same thing. Every series is one run, colored off the
// wheel, and its legend chip is the click-through to the run page — the
// overlay finds the interesting run, the run page explains it.
"use strict";

import {WHEEL, brief, el, esc, getJSON, hideTip, poll, showTip} from "./dom.js";
import {plot} from "./charts.js";
import {ctx, legend, runPath} from "./nav.js";

let metric = new URLSearchParams(location.search).get("metric") || "reward";
let expr = new URLSearchParams(location.search).get("q") || "";
let names = null;

export async function drawCharts() {
  frame();
  if (names === null) {
    const listed = await getJSON("/api/metrics");
    names = (listed && listed.metrics) || ["reward"];
    if (!names.includes(metric)) metric = names[0] || "reward";
    fillSelect();
  }
  const data = await getJSON("/api/series?metric=" + encodeURIComponent(metric)
                             + "&q=" + encodeURIComponent(expr));
  render(data);
}

// ---- the frame: built once, so the poll never yanks the controls ----------

function frame() {
  if (document.getElementById("overlay")) return;
  const holder = document.getElementById("page");
  holder.innerHTML = "";
  const bar = el("div", {class: "find"});
  const box = el("input", {id: "q", type: "search", autocomplete: "off",
      placeholder: "select runs: regex terms AND · \u201c | \u201d for OR "
                 + "· tag:lora exact (e.g. plora prior=0.3 | tag:lora r=4)"});
  box.value = expr;
  let debounce = null;
  box.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => { expr = box.value; remember(); drawCharts(); }, 250);
  });
  const pick = el("select", {id: "metric", title: "metric"});
  pick.addEventListener("change", () => {
    metric = pick.value; remember(); drawCharts();
  });
  bar.append(pick, box);
  holder.append(bar);
  holder.append(el("div", {id: "overlay"}));
  holder.append(el("div", {id: "chips", class: "chips"}));
}

function remember() {
  // the selection IS the address: reload, share, or bookmark it
  const params = new URLSearchParams(location.search);
  params.set("metric", metric);
  if (expr) params.set("q", expr); else params.delete("q");
  history.replaceState(null, "", location.pathname + "?" + params.toString());
}

function fillSelect() {
  const pick = document.getElementById("metric");
  pick.innerHTML = "";
  for (const name of names)
    pick.append(el("option", {value: name}, esc(name)));
  pick.value = metric;
}

// ---- the overlay ----------------------------------------------------------

function render(data) {
  const overlay = document.getElementById("overlay");
  const chips = document.getElementById("chips");
  overlay.innerHTML = "";
  chips.innerHTML = "";
  const series = (data && data.series) || [];
  ctx(`<span class="meta">${series.length} of ${data ? data.matched : 0} matching run`
    + `${(data && data.matched === 1) ? "" : "s"} · metric ${esc(metric)}`
    + (data && data.dropped ? ` · <span class="warn">${data.dropped} more matched — `
        + `narrow the query</span>` : "") + `</span>`);
  if (!series.length) {
    overlay.append(el("p", {class: "note"},
        expr ? `nothing matching “${esc(expr)}” carries ${esc(metric)}`
             : `no run carries ${esc(metric)} yet`));
    return;
  }
  const colored = series.map((s, i) => ({
      label: s.name || s.run_id, color: WHEEL[i % WHEEL.length],
      points: s.points, run: s}));
  overlay.append(plot(colored, {H: 320, xlabel: v => "u" + v}));
  for (const s of colored) {
    const chip = el("a", {class: "chip", href: runPath(s.run.run_id, s.run.folder),
        style: `border-color:${s.color}`,
        title: `${s.run.run_id} · ${s.run.status} · `
             + (s.run.tags || []).join(" ")});
    const last = s.points[s.points.length - 1];
    chip.innerHTML = `<span style="color:${s.color}">●</span> ${esc(s.label)}`
      + ` <span class="k">${brief(last[1])}</span>`;
    chip.addEventListener("mousemove", event => {
      poll.hovering = true;
      showTip(event, `<div><b>${esc(s.label)}</b></div>`
        + `<div class="k">${esc(s.run.run_id)} · ${esc(s.run.status)}</div>`
        + `<div>last ${brief(last[1])} @ u${last[0]}</div>`
        + `<div class="k">${esc((s.run.tags || []).join(" "))}</div>`);
    });
    chip.addEventListener("mouseleave", () => { poll.hovering = false; hideTip(); });
    chips.append(chip);
  }
  legend("one line per run, x = update · the query grammar is the runs page's"
    + " (regex terms AND, | for OR) · a chip links to its run page · metrics"
    + " are whatever the ledgers carry — rails, declared provides, postdata"
    + " columns · refreshes every 3s");
}
