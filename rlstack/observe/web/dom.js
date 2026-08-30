// The small shared vocabulary: element building, the number formats, and THE
// tooltip. Two rules live here. `raw` is the number AS JOURNALED (full logged
// precision) and is what every hover prints; `brief`/`fmt` are for labels only.
// And `poll.hovering` is the one flag the poll loop respects: a redraw never
// happens under the cursor.
"use strict";

export const C = {feed: "#5fb2ff", eval: "#ffb86b", rail: "#9de08f",
                  derived: "#c792ea", dim: "#8494a4", warn: "#e07a7a",
                  teal: "#6fd6c4", line: "#2a323c"};
export const WHEEL = [C.feed, C.rail, C.eval, C.derived, C.warn, C.teal];
export const STATUS_COLOR = {running: C.rail, done: C.feed,
                             failed: C.warn, stalled: C.eval};
export const poll = {hovering: false};

export function el(tag, attrs, html) {
  const node = document.createElement(tag);
  for (const key in (attrs || {})) node.setAttribute(key, attrs[key]);
  if (html !== undefined) node.innerHTML = html;
  return node;
}

export function esc(s) {
  return String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function fmt(v) {                    // axis labels: compact
  const a = Math.abs(v);
  return a === 0 ? "0" : a >= 1000 ? v.toFixed(0)
       : a >= 1 ? v.toFixed(2) : v.toPrecision(2);
}

export function raw(v) {                    // hover: the number as journaled
  if (v === null || v === undefined) return "—";
  return Number.isInteger(v) ? String(v) : String(parseFloat(v.toPrecision(8)));
}

export function brief(v) {                  // the card's own "now": four digits
  if (v === null || v === undefined) return "—";
  return Number.isInteger(v) ? String(v) : String(parseFloat(v.toPrecision(4)));
}

export function when(t) {
  if (!t) return "—";
  return new Date(t * 1000).toLocaleString([], {month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false});
}

export function clock(t) {
  if (!t) return "—";
  return new Date(t * 1000).toLocaleTimeString([], {hour: "2-digit",
      minute: "2-digit", second: "2-digit", hour12: false});
}

export function ago(t, now) {
  // relative age against the SERVER's clock when given (a page must not call
  // a host dead because the laptop's clock drifts)
  if (!t) return "never";
  const s = Math.max(0, (now || Date.now() / 1000) - t);
  if (s < 90) return s.toFixed(0) + "s ago";
  if (s < 5400) return (s / 60).toFixed(0) + "m ago";
  if (s < 129600) return (s / 3600).toFixed(1) + "h ago";
  return (s / 86400).toFixed(1) + "d ago";
}

export function pulseDot(pulse, now) {
  // the one liveness glyph, everywhere: probe or presumption, said as which
  if (!pulse || pulse.live === null || pulse.live === undefined)
    return `<span class="dot unknown" title="liveness unknown — journal too thin">○</span>`;
  const seen = ago(pulse.last, now);
  const how = pulse.source === "desk" ? "probed by the fleet service"
                                      : "presumed from the journal heartbeat";
  return pulse.live
    ? `<span class="dot live" title="alive (${how}) · last event ${seen}">●</span>`
    : `<span class="dot down" title="down (${how}) · last event ${seen}">●</span>`;
}

export function dur(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 90) return seconds.toFixed(0) + "s";
  if (seconds < 5400) return (seconds / 60).toFixed(1) + "m";
  return (seconds / 3600).toFixed(1) + "h";
}

export function gib(mib) {                  // MiB is journaled; GiB reads better
  return mib === null || mib === undefined ? "—" : (mib / 1024).toFixed(1) + " GiB";
}

export function extent(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

export function showTip(event, html) {
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

export function hideTip() {
  document.getElementById("tip").style.display = "none";
}

// a scroll moves the chart out from under the cursor: the tooltip goes with
// it, and the poll is free again
addEventListener("scroll", () => { poll.hovering = false; hideTip(); },
                 {passive: true});

export function section(title, note, kind) {
  const holder = document.getElementById("page");
  holder.append(el("h2", {}, esc(title) + (note ? ` <span>${esc(note)}</span>` : "")));
  const grid = el("div", {class: kind ? "grid " + kind : "grid"});
  holder.append(grid);
  return grid;
}

export function note(text) {
  document.getElementById("page").append(el("p", {class: "note"}, esc(text)));
}

export async function getJSON(path) {
  const res = await fetch(path);
  return res.ok ? await res.json() : null;
}
