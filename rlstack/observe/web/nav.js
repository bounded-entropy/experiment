// The route and the header. Six pages, one document: the path is parsed once,
// here, and every page module asks this record rather than re-reading
// location. The experiment switcher lives here too, because the run page and
// the wave page share it.
//
// ADDRESSING (#58). A folder is a store root, and the same spec submitted
// under two folders yields the SAME run_id in both — so a run is addressed by
// (folder, run_id), and every link this file builds carries ?root=<folder>
// when the reader can see more than one. No folder in the URL means "resolve
// against every root"; when several hold the id the API answers with the
// ambiguity and drawAmbiguity lists the folders. Nothing ever silently picks.
"use strict";

import {el, esc} from "./dom.js";

export const route = parse(location.pathname, location.search);

function parse(pathname, search) {
  const parts = pathname.split("/").filter(Boolean).map(decodeURIComponent);
  const params = new URLSearchParams(search);
  // "" is a folder (the top directory itself); absence is not emptiness
  const folder = params.has("root") ? params.get("root") : null;
  if (!parts.length) return {page: "runs", folder: folder};
  if (parts[0] === "run" && parts.length === 2)
    return {page: "run", runId: parts[1], folder: folder};
  if (parts[0] === "run" && parts[2] === "wave" && parts.length === 4)
    return {page: "wave", runId: parts[1], update: parseInt(parts[3], 10),
            folder: folder};
  if (parts[0] === "hosts") return {page: "fleet", folder: folder};
  if (parts[0] === "host" && parts.length === 2)
    return {page: "host", host: parts[1], folder: folder};
  return {page: "runs", folder: folder};
}

export function query(folder) {
  return (folder === null || folder === undefined) ? ""
       : "?root=" + encodeURIComponent(folder);
}

export function runPath(runId, folder) {
  return "/run/" + encodeURIComponent(runId) + query(folder);
}
export function hostPath(host, folder) {
  return "/host/" + encodeURIComponent(host) + query(folder);
}
export function wavePath(runId, update, folder) {
  return "/run/" + encodeURIComponent(runId) + "/wave/" + update + query(folder);
}
export function apiRun(runId, tail, folder) {
  return "/api/run/" + encodeURIComponent(runId) + (tail || "") + query(folder);
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
      if (sel.value && sel.value !== here()) location.href = sel.value;
    });
    hdr.append(sel);
  }
}

function here() {
  return runPath(route.runId, route.folder);
}

export function ctx(html) {
  document.getElementById("ctx").innerHTML = html;
}

let switcherSignature = null;

export function syncSwitcher(runs) {
  // rebuilt only when the run list itself changes — a poll must not close an
  // open dropdown under the cursor. An option's VALUE is its href, because
  // (folder, run_id) is the address and the href already spells it.
  const sel = document.getElementById("switch");
  if (!sel) return;
  const known = runs.some(r => r.run_id === route.runId
                               && (route.folder === null || r.folder === route.folder));
  const options = (known ? runs
    : [{run_id: route.runId, folder: route.folder === null ? "" : route.folder,
        name: "", status: "?", committed: 0, target: "?"}].concat(runs));
  const many = new Set(options.map(r => r.folder)).size > 1;
  const signature = options.map(r =>
      `${r.folder}/${r.run_id}:${r.status}:${r.committed}`).join("|");
  if (signature === switcherSignature) return;
  switcherSignature = signature;
  sel.innerHTML = "";
  const folders = new Map();
  for (const r of options.slice().reverse()) {
    if (!folders.has(r.folder)) folders.set(r.folder, []);
    folders.get(r.folder).push(r);
  }
  for (const [folder, rows] of folders) {
    const into = many ? el("optgroup", {label: folder || "(top)"}) : sel;
    for (const r of rows)
      into.append(el("option", {value: runPath(r.run_id, many ? r.folder : null)},
          `${esc(r.name || r.run_id)} · ${esc(r.status)} ${r.committed}/${esc(r.target)}`
          + (r.name ? ` · ${esc(r.run_id)}` : "")));
    if (many) sel.append(into);
  }
  sel.value = here();
}

export function drawAmbiguity(runId, folders) {
  // THE ONE AMBIGUITY, said out loud: this id names a run in more than one
  // folder, and picking one for the reader would be a guess about which
  // experiment they meant.
  const holder = document.getElementById("page");
  holder.innerHTML = "";
  const sel = document.getElementById("switch");
  if (sel) sel.remove();          // nothing to switch FROM: no run is selected
  ctx(`<span>${esc(runId)}</span>`
    + `<span class="warn">in ${folders.length} folders</span>`);
  holder.append(el("p", {class: "note"},
      "this run_id names a run in more than one folder — the same spec "
    + "submitted twice. Identity is content; the folder says which one."));
  const table = el("table", {}, "<tr><th>folder</th><th>run</th></tr>");
  for (const folder of folders) {
    const row = el("tr", {});
    row.append(el("td", {}, `<a href="${runPath(runId, folder)}">`
        + `${esc(folder || "(top)")}</a>`));
    row.append(el("td", {class: "k"}, esc(runId)));
    table.append(row);
  }
  holder.append(table);
  legend("a folder is a store root, fixed at birth · every link the index "
    + "emits carries its folder");
}

export function legend(html) {
  document.getElementById("legend").innerHTML = html;
}
