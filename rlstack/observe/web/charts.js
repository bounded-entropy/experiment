// The charts. Four shapes — a line rail, a stacked bar, a histogram, a
// timeline — and one hover contract they all keep: EVERY drawn thing is one
// cursor away from the raw journaled number behind it (dom.raw, full logged
// precision). Charts are MEASURED: each draws at its container's own pixel
// width, viewBox == css pixels, so nothing is letterboxed at any card width.
"use strict";

import {C, STATUS_COLOR, ago, brief, clock, dur, el, esc, extent, fmt, hideTip,
        poll, raw, showTip} from "./dom.js";

export function measured(build) {
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

// ---- the line rail --------------------------------------------------------

export function plot(series, opts) {
  const o = Object.assign({H: 150, pad: 34, unit: "", xlabel: v => "u" + v},
                          opts || {});
  return measured((box, width) =>
      drawPlot(box, Object.assign({W: width}, o), series));
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
  const marks_ = (o.markers || []).slice();
  const refs = (o.reference || []).filter(r => Number.isFinite(r.y));
  const xs = [], ys = [];
  for (const s of drawn) for (const p of s.points) { xs.push(p[0]); ys.push(p[1]); }
  let [x0, x1] = extent(xs), [y0, y1] = extent(ys);
  for (const r of refs) { y0 = Math.min(y0, r.y); y1 = Math.max(y1, r.y); }
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
  const reference = refs.map(r => `
    <line x1="${o.pad}" y1="${sy(r.y).toFixed(1)}" x2="${o.W - 8}"
          y2="${sy(r.y).toFixed(1)}" stroke="${r.color || C.dim}"
          stroke-width="1" stroke-dasharray="3 4" opacity="0.75"/>
    <text x="${o.W - 10}" y="${(sy(r.y) - 3).toFixed(1)}" fill="${r.color || C.dim}"
          font-size="10" text-anchor="end">${esc(r.label || "")}</text>`).join("");
  const moments = marks_.filter(m => m.t >= x0 && m.t <= x1).map(m => `
    <line x1="${sx(m.t).toFixed(1)}" y1="4" x2="${sx(m.t).toFixed(1)}"
          y2="${o.H - 18}" stroke="${m.color}" stroke-width="1"
          stroke-dasharray="2 3" opacity="0.6"/>
    <circle cx="${sx(m.t).toFixed(1)}" cy="5" r="2.4" fill="${m.color}"/>`).join("");

  box.innerHTML = `<svg viewBox="0 0 ${o.W} ${o.H}" style="height:${o.H}px">
    <text x="2" y="12" fill="${C.dim}" font-size="10">${fmt(y1)}</text>
    <text x="2" y="${o.H - 20}" fill="${C.dim}" font-size="10">${fmt(y0)}</text>
    <text x="${o.pad}" y="${o.H - 4}" fill="${C.dim}" font-size="10">${o.xlabel(x0)}</text>
    <text x="${o.W - 8}" y="${o.H - 4}" fill="${C.dim}" font-size="10"
          text-anchor="end">${o.xlabel(x1)}</text>
    <line x1="${o.pad}" y1="${o.H - 18}" x2="${o.W - 8}" y2="${o.H - 18}" stroke="${C.dim}33"/>
    ${reference}${moments}
    ${drawn.map(line).join("")}
    <line class="cross" y1="4" y2="${o.H - 18}" stroke="${C.dim}" stroke-width="0.7" opacity="0"/>
    ${drawn.map(s => `<circle class="mk" r="3.4" fill="${s.color}" opacity="0"/>`).join("")}
    <rect x="0" y="0" width="${o.W}" height="${o.H}" fill="transparent"/></svg>`;

  const svg = box.querySelector("svg");
  const cross = svg.querySelector(".cross");
  const marks = Array.from(svg.querySelectorAll(".mk"));
  svg.addEventListener("mousemove", event => {
    poll.hovering = true;
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
      // a derived point carries the journaled counts it was summed from
      if (best[2] !== undefined) rows.push(`<div class="k">${esc(best[2])}</div>`);
    });
    // a moment within a few pixels of the cursor is part of the reading
    for (const m of marks_)
      if (Math.abs(sx(m.t) - at.x) < 6)
        rows.push(`<div><span style="color:${m.color}">▲ ${esc(m.label)}</span>`
                  + ` <span class="k">${esc(clock(m.t))}</span></div>`);
    cross.setAttribute("x1", sx(head).toFixed(1));
    cross.setAttribute("x2", sx(head).toFixed(1));
    cross.setAttribute("opacity", 1);
    showTip(event, `<div class="k">${esc(o.xlabel(head))}</div>` + rows.join(""));
  });
  svg.addEventListener("mouseleave", () => {
    poll.hovering = false; hideTip();
    cross.setAttribute("opacity", 0);
    marks.forEach(m => m.setAttribute("opacity", 0));
  });
}

export function card(title, who, series, opts) {
  const node = el("div", {class: "card"});
  node.append(el("div", {}, `<span class="name">${esc(title)}</span>`
                          + `<span class="who">${esc(who)}</span>`));
  node.append(plot(series, opts));
  const primary = (series || []).find(s => s.points && s.points.length);
  const last = primary ? primary.points[primary.points.length - 1] : null;
  // "now" is a claim about the present: on a time axis it holds only while
  // the last point is fresh against the server clock (opts.asOf); a stale
  // tail reads "last … Xh ago", and NOTHING recent reads as the truth it is
  let label = "no data yet";
  if (last !== null) {
    const value = brief(last[1]) + ((opts && opts.unit) || "");
    if (opts && opts.asOf) {
      const age = opts.asOf - last[0];
      const fresh = age < Math.max(120, (opts.freshS || 0) * 3);
      label = fresh ? "now " + value
                    : "last " + value + " · " + ago(last[0], opts.asOf);
    } else {
      label = "now " + value;
    }
  }
  node.append(el("div", {class: "now"}, label));
  return node;
}

export function emptyCard(title, who, why) {
  const node = el("div", {class: "card"});
  node.append(el("div", {}, `<span class="name">${esc(title)}</span>`
                          + `<span class="who">${esc(who)}</span>`));
  node.append(el("div", {class: "now"}, esc(why)));
  return node;
}

// ---- the stacked bar: one update, decomposed ------------------------------

export function stacked(rows, keys, opts) {
  const o = Object.assign({H: 190, pad: 40, unit: "s", xlabel: v => "u" + v},
                          opts || {});
  return measured((box, width) =>
      drawStacked(box, Object.assign({W: width}, o), rows, keys));
}

function drawStacked(box, o, rows, keys) {
  if (!rows.length) {
    box.innerHTML = `<svg viewBox="0 0 ${o.W} ${o.H}" style="height:${o.H}px"></svg>`;
    return;
  }
  const top = Math.max(...rows.map(r => r.total)) || 1;
  const inner = o.W - o.pad - 8;
  const step = inner / rows.length;
  const bar = Math.max(2, Math.min(26, step - 2));
  const sy = v => (o.H - 18) - v / top * (o.H - 30);
  let body = "";
  rows.forEach((row, i) => {
    const x = o.pad + step * i + (step - bar) / 2;
    let base = 0;
    keys.forEach((key, k) => {
      const value = row.parts[key.name] || 0;
      if (value <= 0) return;
      const y = sy(base + value), h = sy(base) - sy(base + value);
      body += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}"
                 width="${bar.toFixed(1)}" height="${Math.max(0.5, h).toFixed(1)}"
                 fill="${key.color}" opacity="0.85"/>`;
      base += value;
    });
    if (base < row.total) {                     // whatever the phases missed
      const y = sy(row.total), h = sy(base) - sy(row.total);
      body += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}"
                 width="${bar.toFixed(1)}" height="${Math.max(0.5, h).toFixed(1)}"
                 fill="${C.dim}" opacity="0.35"/>`;
    }
  });
  box.innerHTML = `<svg viewBox="0 0 ${o.W} ${o.H}" style="height:${o.H}px">
    <text x="2" y="12" fill="${C.dim}" font-size="10">${fmt(top)}${esc(o.unit)}</text>
    <text x="2" y="${o.H - 20}" fill="${C.dim}" font-size="10">0</text>
    <text x="${o.pad}" y="${o.H - 4}" fill="${C.dim}" font-size="10">${
      esc(o.xlabel(rows[0].x))}</text>
    <text x="${o.W - 8}" y="${o.H - 4}" fill="${C.dim}" font-size="10"
          text-anchor="end">${esc(o.xlabel(rows[rows.length - 1].x))}</text>
    <line x1="${o.pad}" y1="${o.H - 18}" x2="${o.W - 8}" y2="${o.H - 18}"
          stroke="${C.dim}33"/>
    <rect class="hi" y="4" height="${o.H - 22}" width="${step.toFixed(1)}"
          fill="#dde5ec" opacity="0"/>
    ${body}
    <rect x="0" y="0" width="${o.W}" height="${o.H}" fill="transparent"/></svg>`;

  const svg = box.querySelector("svg");
  const hi = svg.querySelector(".hi");
  svg.addEventListener("mousemove", event => {
    poll.hovering = true;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const at = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    const index = Math.max(0, Math.min(rows.length - 1,
        Math.floor((at.x - o.pad) / step)));
    const row = rows[index];
    hi.setAttribute("x", (o.pad + step * index).toFixed(1));
    hi.setAttribute("opacity", 0.07);
    const lines = keys.filter(key => row.parts[key.name] !== undefined).map(key =>
      `<div><span style="color:${key.color}">${esc(key.name)}</span> `
      + raw(row.parts[key.name]) + esc(o.unit) + "</div>");
    showTip(event, `<div class="k">${esc(o.xlabel(row.x))}${
        row.at ? " · " + esc(clock(row.at)) : ""}</div>`
      + `<div><b>total</b> ${raw(row.total)}${esc(o.unit)}</div>` + lines.join(""));
  });
  svg.addEventListener("mouseleave", () => {
    poll.hovering = false; hideTip(); hi.setAttribute("opacity", 0);
  });
}

export function stackedCard(title, who, rows, keys, opts) {
  const node = el("div", {class: "card"});
  node.append(el("div", {}, `<span class="name">${esc(title)}</span>`
                          + `<span class="who">${esc(who)}</span>`));
  node.append(stacked(rows, keys, opts));
  node.append(el("div", {class: "rail"}, keys.map(key =>
      `<span style="color:${key.color}">■</span> ${esc(key.name)}`).join(" ")));
  const last = rows.length ? rows[rows.length - 1] : null;
  node.append(el("div", {class: "now"}, last
      ? "now " + brief(last.total) + ((opts && opts.unit) || "") : "no data yet"));
  return node;
}

// ---- the histogram: a wave's own distribution -----------------------------

export function histogramCard(panel) {
  const node = el("div", {class: "card"});
  const h = panel.histogram;
  node.append(el("div", {}, `<span class="name">${esc(panel.name)}</span>`
                          + `<span class="who">${esc(panel.source || "")}</span>`));
  node.append(measured((box, width) => drawHistogram(box, width, h, panel)));
  node.append(el("div", {class: "now"}, h.n
      ? `n ${h.n} · mean ${brief(h.mean)}${esc(panel.unit || "")}`
      : "no values in this wave"));
  return node;
}

function drawHistogram(box, W, h, panel) {
  const H = 132, pad = 34;
  if (!h.n) {
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="height:${H}px"></svg>`;
    return;
  }
  const top = Math.max(...h.bins.map(b => b.count)) || 1;
  const inner = W - pad - 8;
  const step = inner / h.bins.length;
  const sy = count => (H - 18) - count / top * (H - 30);
  const body = h.bins.map((bin, i) => `
    <rect x="${(pad + step * i + 0.5).toFixed(1)}" y="${sy(bin.count).toFixed(1)}"
          width="${Math.max(1, step - 1).toFixed(1)}"
          height="${Math.max(0.5, (H - 18) - sy(bin.count)).toFixed(1)}"
          fill="${C.feed}" opacity="0.75"/>`).join("");
  const meanX = h.max === h.min ? pad + inner / 2
      : pad + (h.mean - h.min) / (h.max - h.min) * inner;
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="height:${H}px" class="hist">
    <text x="2" y="12" fill="${C.dim}" font-size="10">${top}</text>
    <text x="${pad}" y="${H - 4}" fill="${C.dim}" font-size="10">${fmt(h.min)}</text>
    <text x="${W - 8}" y="${H - 4}" fill="${C.dim}" font-size="10"
          text-anchor="end">${fmt(h.max)}</text>
    ${body}
    <line x1="${meanX.toFixed(1)}" y1="4" x2="${meanX.toFixed(1)}" y2="${H - 18}"
          stroke="${C.eval}" stroke-width="1" stroke-dasharray="3 3"/>
    <line x1="${pad}" y1="${H - 18}" x2="${W - 8}" y2="${H - 18}" stroke="${C.dim}33"/>
    <rect class="hi" y="4" height="${H - 22}" width="${Math.max(1, step - 1).toFixed(1)}"
          fill="${C.dim}" opacity="0"/>
    <rect x="0" y="0" width="${W}" height="${H}" fill="transparent"/></svg>`;

  const svg = box.querySelector("svg");
  const hi = svg.querySelector(".hi");
  svg.addEventListener("mousemove", event => {
    poll.hovering = true;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const at = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    const index = Math.max(0, Math.min(h.bins.length - 1,
        Math.floor((at.x - pad) / step)));
    const bin = h.bins[index];
    hi.setAttribute("x", (pad + step * index + 0.5).toFixed(1));
    hi.setAttribute("opacity", 0.18);
    showTip(event, `<div class="k">${esc(panel.name)}</div>`
      + `<div>${raw(bin.lo)} … ${raw(bin.hi)}${esc(panel.unit || "")}</div>`
      + `<div><b>${bin.count}</b> of ${h.n}</div>`
      + `<div class="k">mean ${raw(h.mean)} · min ${raw(h.min)} · max ${raw(h.max)}</div>`);
  });
  svg.addEventListener("mouseleave", () => {
    poll.hovering = false; hideTip(); hi.setAttribute("opacity", 0);
  });
}

// ---- the timeline: one lane per host (fleet) or per residency (host) ------

export function timeline(lanes, window_, describe) {
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
      const color = bar.color || STATUS_COLOR[bar.status] || C.dim;
      body += `<rect class="bar" x="${a.toFixed(1)}" y="${y + 3}"
                 width="${Math.max(2, b - a).toFixed(1)}" height="10" rx="2"
                 fill="${color}" opacity="0.75"/>`;
      bars.push(bar);
    }
  });
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="height:${H}px">
    ${body}
    <text x="${PAD}" y="${H - 4}" fill="${C.dim}" font-size="10">${esc(whenLabel(t0))}</text>
    <text x="${W - 14}" y="${H - 4}" fill="${C.dim}" font-size="10"
          text-anchor="end">${esc(whenLabel(t1))}</text></svg>`;
  Array.from(box.querySelectorAll("rect.bar")).forEach((rect, i) => {
    rect.addEventListener("mousemove", event => {
      poll.hovering = true; showTip(event, describe(bars[i]));
    });
    rect.addEventListener("mouseleave", () => { poll.hovering = false; hideTip(); });
  });
}

function whenLabel(t) {
  return new Date(t * 1000).toLocaleString([], {month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false});
}

export function residencyTip(bar) {
  const span = (bar.t1 || Date.now() / 1000) - (bar.t0 || 0);
  return `<div><b>${esc(bar.label)}</b> <span class="k">${esc(bar.status)}</span></div>`
    + `<div><span class="k">from</span> ${esc(whenLabel(bar.t0 || 0))}</div>`
    + `<div><span class="k">to</span>   ${bar.t1 ? esc(whenLabel(bar.t1)) : "resident"}</div>`
    + `<div><span class="k">for</span>  ${esc(dur(bar.t0 ? span : null))}</div>`
    + (bar.detail ? `<div class="k">${esc(bar.detail)}</div>` : "");
}
