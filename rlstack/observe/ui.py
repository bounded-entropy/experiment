"""The observer UI: a local wandb over the store, with zero core interference.

One stdlib WSGI app (no dependencies, no build step, no CDN) served two ways:
locally (`python -m rlstack ui <store-locator>`) or beside a remote store
(deploy wraps ui_app in a Modal web endpoint with the volume mounted). The
page is self-contained HTML+JS with hand-rolled SVG line charts, polling the
JSON API every few seconds — live monitoring is just committed state read
again.

Panel priority IS the flow graph (I11): the run's own dictionary.json orders
the page — columns that feed the loss (the walkback) first, the rails second,
measurement-only columns and eval series last. The UI never re-derives a
declaration and never reads anything but peeks and journals.

Routes:
    /                     run index (from the host journals)
    /run/<run_id>         the graphs page
    /api/runs             runs_data as JSON
    /api/run/<run_id>     run_series as JSON
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from rlstack.data.stores.base import Store
from rlstack.observe.series import run_series
from rlstack.observe.views import runs_data


def ui_app(stores: Sequence[Store],
           refresh: Callable[[], None] | None = None,
           panels: Callable[[], list[dict]] | None = None):
    """The WSGI app. `refresh` runs before each API read (a Modal volume
    needs .reload() to see commits from other containers; None for local).
    `panels` supplies derived-graph declarations (None → each store's own
    panels.json, re-read per request so edits appear live)."""

    def app(environ, start_response):
        path = environ.get("PATH_INFO", "/")
        if path.startswith("/api/"):
            if refresh is not None:
                refresh()
            if path == "/api/runs":
                return _json(start_response, runs_data(stores))
            if path.startswith("/api/run/"):
                run_id = path[len("/api/run/"):]
                for store in stores:
                    series = run_series(
                        store, run_id,
                        panels=panels() if panels is not None else None)
                    if series is not None:
                        return _json(start_response, series)
                return _json(start_response, {"error": "unknown run"}, "404 Not Found")
            return _json(start_response, {"error": "unknown route"}, "404 Not Found")
        # both pages are the same document; JS routes on location.pathname
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [PAGE.encode("utf-8")]

    return app


def _json(start_response, payload, status: str = "200 OK"):
    body = json.dumps(payload).encode("utf-8")
    start_response(status, [("Content-Type", "application/json"),
                            ("Cache-Control", "no-store")])
    return [body]


def serve(stores: Sequence[Store], port: int = 8321,
          panels_path: str | None = None) -> None:
    """Local serving: python -m rlstack ui <store> [--port N]
    [--panels panels.json] — the file re-reads per refresh, so editing a
    panel and saving shows up on the next poll."""
    import json as _json_mod
    from wsgiref.simple_server import make_server

    def panels():
        try:
            return _json_mod.loads(open(panels_path).read())
        except (OSError, ValueError):
            return []

    app = ui_app(stores, panels=panels if panels_path else None)
    with make_server("127.0.0.1", port, app) as httpd:
        print(f"rlstack ui: http://127.0.0.1:{port}  (ctrl-c to stop)")
        httpd.serve_forever()


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>rlstack</title>
<style>
  :root { --bg:#101418; --card:#1a2027; --ink:#dde5ec; --dim:#8494a4;
          --line:#2a323c; --feed:#5fb2ff; --eval:#ffb86b; --rail:#9de08f; }
  body { background:var(--bg); color:var(--ink); margin:0;
         font:14px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { padding:14px 22px; border-bottom:1px solid var(--line);
           display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }
  header a { color:var(--ink); text-decoration:none; font-weight:700; }
  header .meta { color:var(--dim); }
  h2 { font-size:13px; color:var(--dim); text-transform:uppercase;
       letter-spacing:.08em; margin:26px 22px 8px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
          gap:14px; padding:0 22px; }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:8px; padding:10px 12px 6px; }
  .card .name { font-weight:700; }
  .card .who { color:var(--dim); font-size:12px; float:right; }
  .card .now { color:var(--dim); font-size:12px; }
  svg { width:100%; height:150px; display:block; }
  table { border-collapse:collapse; margin:18px 22px; }
  td, th { padding:6px 14px 6px 0; text-align:left; border-bottom:1px solid var(--line); }
  th { color:var(--dim); font-weight:400; }
  td a { color:var(--feed); text-decoration:none; }
  .legend { color:var(--dim); font-size:12px; padding:4px 22px 30px; }
  .live { color:var(--rail); }
</style>
<header id="hdr"><a href="/">rlstack</a></header>
<div id="page"></div>
<div class="legend" id="legend"></div>
<script>
"use strict";
const runId = location.pathname.startsWith("/run/")
    ? location.pathname.slice(5) : null;

function el(tag, attrs, html) {
  const node = document.createElement(tag);
  for (const key in (attrs || {})) node.setAttribute(key, attrs[key]);
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function chart(points, evalPoints, color) {
  // points: [[x, y], ...] — a hand-rolled SVG polyline, nothing else
  const W = 340, H = 150, P = 30;
  const all = points.concat(evalPoints);
  if (!all.length) return "<svg></svg>";
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  if (y0 === y1) { y0 -= Math.abs(y0) * 0.1 + 1e-3; y1 += Math.abs(y1) * 0.1 + 1e-3; }
  const sx = x => P + (x - x0) / (x1 - x0) * (W - P - 8);
  const sy = y => (H - 18) - (y - y0) / (y1 - y0) * (H - 30);
  const line = (pts, dash, c) => pts.length
    ? `<polyline fill="none" stroke="${c}" stroke-width="1.6" ${dash}
         points="${pts.map(p => sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join(" ")}"/>`
      + pts.slice(-1).map(p =>
        `<circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="2.5" fill="${c}"/>`)
    : "";
  const fmt = v => Math.abs(v) >= 100 ? v.toFixed(0)
             : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2);
  return `<svg viewBox="0 0 ${W} ${H}">
    <text x="2" y="12" fill="#8494a4" font-size="10">${fmt(y1)}</text>
    <text x="2" y="${H - 20}" fill="#8494a4" font-size="10">${fmt(y0)}</text>
    <text x="${W - 8}" y="${H - 4}" fill="#8494a4" font-size="10" text-anchor="end">u${x1}</text>
    <line x1="${P}" y1="${H - 18}" x2="${W - 8}" y2="${H - 18}" stroke="#2a323c"/>
    ${line(points, "", color)}
    ${line(evalPoints, 'stroke-dasharray="5 4"', "#ffb86b")}</svg>`;
}

function seriesOf(data, section, name) {
  return data.updates.filter(u => u[section][name] !== undefined && u[section][name] !== null)
                     .map(u => [u.update, u[section][name]]);
}
function evalSeriesOf(data, name) {
  return data.eval.filter(e => e.means[name] !== undefined)
                  .map(e => [e.update, e.means[name]]);
}

function card(title, who, points, evalPoints, color) {
  const node = el("div", {class: "card"});
  const last = points.length ? points[points.length - 1][1] : null;
  node.append(el("div", {}, `<span class="name">${title}</span>` +
                            `<span class="who">${who}</span>`));
  node.append(el("div", {}, chart(points, evalPoints, color)));
  node.append(el("div", {class: "now"},
      last === null ? "no data yet" : "now " + last.toPrecision(4)));
  return node;
}

function section(title) {
  const holder = document.getElementById("page");
  holder.append(el("h2", {}, title));
  const grid = el("div", {class: "grid"});
  holder.append(grid);
  return grid;
}

async function drawRun() {
  const res = await fetch("/api/run/" + runId);
  if (!res.ok) { document.getElementById("page").textContent = "unknown run"; return; }
  const data = await res.json();
  const dict = data.dictionary || {columns: [], rails: []};
  const hdr = document.getElementById("hdr");
  hdr.innerHTML = `<a href="/">rlstack</a> <span>${runId}</span>
    <span class="meta">loss ${dict.loss ?? "?"} · lag ${dict.max_policy_lag ?? 0}
    · post [${(dict.post_pipeline || []).join(" → ")}]</span>
    <span class="${data.committed >= (data.target ?? 1e9) ? "meta" : "live"}">
    ${data.committed}/${data.target ?? "?"} committed</span>`;
  document.getElementById("page").innerHTML = "";

  const cols = dict.columns || [];
  const postCols = cols.filter(c => c.phase === "post" && c.stored);
  const evalCols = cols.filter(c => c.phase === "eval" && c.stored);
  const feeding = postCols.filter(c => c.feeds_loss);
  const measure = postCols.filter(c => !c.feeds_loss);

  if (feeding.length) {
    const grid = section("feeds the loss (walkback from " + dict.loss + ")");
    for (const c of feeding)
      grid.append(card(c.name, c.producer.replace("postprocessor:", ""),
                       seriesOf(data, "post", c.name),
                       evalSeriesOf(data, c.name), "#5fb2ff"));
  }
  const rails = section("rails");
  for (const r of (dict.rails || []))
    rails.append(card(r, "loss:" + (dict.loss ?? "?"),
                      seriesOf(data, "train", r), [], "#9de08f"));
  const derived = data.derived || [];
  if (derived.length) {
    const grid = section("derived (your panels.json)");
    for (const d of derived) {
      if (d.missing || d.error) {
        const node = el("div", {class: "card"});
        node.append(el("div", {}, `<span class="name">${d.name}</span>` +
                                  `<span class="who">${d.expr}</span>`));
        node.append(el("div", {class: "now", style: "color:#e07a7a"},
            d.error ? d.error :
            "missing from this run's pipeline: " + d.missing.join(", ")));
        grid.append(node);
      } else {
        grid.append(card(d.name, d.expr, d.points, d.eval, "#c792ea"));
      }
    }
  }
  if (measure.length || evalCols.length) {
    const grid = section("measurement");
    const seen = new Set();
    for (const c of measure) {
      seen.add(c.name);
      grid.append(card(c.name, c.producer.replace("postprocessor:", ""),
                       seriesOf(data, "post", c.name),
                       evalSeriesOf(data, c.name), "#8494a4"));
    }
    for (const c of evalCols)
      if (!seen.has(c.name))
        grid.append(card(c.name + " (eval)",
                         c.producer.replace("postprocessor:", ""),
                         [], evalSeriesOf(data, c.name), "#ffb86b"));
  }
  document.getElementById("legend").innerHTML =
    "solid = training · <span style='color:#ffb86b'>dashed = held-out eval</span>" +
    " · refreshes every 3s · panels & priority from the run's own dictionary.json";
}

async function drawIndex() {
  const res = await fetch("/api/runs");
  const runs = await res.json();
  const holder = document.getElementById("page");
  holder.innerHTML = "";
  if (!runs.length) { holder.append(el("p", {style: "padding:22px"},
      "no experiments journaled in this store yet")); return; }
  const table = el("table", {},
    "<tr><th>run</th><th>status</th><th>committed</th><th>host(s)</th><th>store</th></tr>");
  for (const r of runs.slice().reverse()) {
    const row = el("tr", {});
    row.append(el("td", {}, `<a href="/run/${r.run_id}">${r.run_id}</a>`));
    row.append(el("td", {class: r.status === "running" ? "live" : ""},
                  r.status + (r.forked ? " ⚠FORK" : "")));
    row.append(el("td", {}, `${r.committed}/${r.target}`));
    row.append(el("td", {}, r.hosts.join("+")));
    row.append(el("td", {}, r.store));
    table.append(row);
  }
  holder.append(table);
}

const draw = runId ? drawRun : drawIndex;
draw();
setInterval(draw, 3000);
</script>
"""
