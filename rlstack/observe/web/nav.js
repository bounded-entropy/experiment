// The route and the header. Six pages, one document: the path is parsed once,
// here, and every page module asks this record rather than re-reading
// location. The experiment switcher lives here too, because the run page and
// the wave page share it.
"use strict";

import {el, esc} from "./dom.js";

export const route = parse(location.pathname);

function parse(pathname) {
  const parts = pathname.split("/").filter(Boolean).map(decodeURIComponent);
  if (!parts.length) return {page: "runs"};
  if (parts[0] === "run" && parts.length === 2)
    return {page: "run", runId: parts[1]};
  if (parts[0] === "run" && parts[2] === "wave" && parts.length === 4)
    return {page: "wave", runId: parts[1], update: parseInt(parts[3], 10)};
  if (parts[0] === "hosts") return {page: "fleet"};
  if (parts[0] === "host" && parts.length === 2)
    return {page: "host", host: parts[1]};
  return {page: "runs"};
}

export function runPath(runId) { return "/run/" + encodeURIComponent(runId); }
export function hostPath(host) { return "/host/" + encodeURIComponent(host); }
export function wavePath(runId, update) {
  return runPath(runId) + "/wave/" + update;
}

export function nav() {
  const runs = route.page === "runs" || route.page === "run" || route.page === "wave";
  const hdr = document.getElementById("hdr");
  hdr.innerHTML = `<a href="/">rlstack</a>`
    + `<a class="nav${runs ? " on" : ""}" href="/">runs</a>`
    + `<a class="nav${runs ? "" : " on"}" href="/hosts">hosts</a>`
    + `<span id="ctx"></span>`;
  if (route.runId) {
    const sel = el("select", {id: "switch", title: "switch experiment"});
    sel.addEventListener("change", () => {
      if (sel.value && sel.value !== route.runId) location.href = runPath(sel.value);
    });
    hdr.append(sel);
  }
}

export function ctx(html) {
  document.getElementById("ctx").innerHTML = html;
}

let switcherSignature = null;

export function syncSwitcher(runs) {
  // rebuilt only when the run list itself changes — a poll must not close an
  // open dropdown under the cursor
  const sel = document.getElementById("switch");
  if (!sel) return;
  const known = runs.some(r => r.run_id === route.runId);
  const options = (known ? runs : [{run_id: route.runId, status: "?", committed: 0,
                                    target: "?"}].concat(runs));
  const signature = options.map(r => `${r.run_id}:${r.status}:${r.committed}`).join("|");
  if (signature === switcherSignature) return;
  switcherSignature = signature;
  sel.innerHTML = "";
  for (const r of options.slice().reverse())
    sel.append(el("option", {value: r.run_id},
        `${esc(r.run_id)} · ${esc(r.status)} ${r.committed}/${esc(r.target)}`));
  sel.value = route.runId;
}

export function legend(html) {
  document.getElementById("legend").innerHTML = html;
}
