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
  if (parts[0] === "charts") return {page: "charts", folder: folder};
  if (parts[0] === "host" && parts.length === 2)
    return {page: "host", host: parts[1], folder: folder};
  return {page: "runs", folder: folder};
}

// The fleet pages read a TIME WINDOW (?hours=N; absent means one day, 0
// means everything). The picker writes it into the address, withHours carries
// it onto the API calls, and run pages never window — a curve is its whole
// story.
export function withHours(url) {
  const h = new URLSearchParams(location.search).get("hours") ?? "24";
  if (h === "0") return url;
  return url + (url.includes("?") ? "&" : "?") + "hours=" + encodeURIComponent(h);
}

function rangeLinks() {
  // ONE segmented control, not three loose links loose in the bar
  const current = new URLSearchParams(location.search).get("hours") ?? "24";
  const mk = (hours, label) => {
    const params = new URLSearchParams(location.search);
    params.set("hours", hours);
    return `<a${current === hours ? ' class="on"' : ""}`
         + ` href="${location.pathname}?${params.toString()}">${label}</a>`;
  };
  return `<span class="win">` + mk("1", "hour") + mk("24", "day")
       + mk("0", "all") + `</span>`;
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

export function keepHours(path) {
  // the window choice follows the reader across pages
  const h = new URLSearchParams(location.search).get("hours");
  return h === null ? path : path + "?hours=" + encodeURIComponent(h);
}

// THE BAR IS NAVIGATION ONLY: a wordmark, three tabs, and the two controls
// that steer a reading (the window, the switcher). It holds nothing about the
// thing being looked at, so it never reflows when a run is selected — that is
// the SUBJECT BAR's job (ctx below), one line underneath.
//
// Each tab lights for its OWN family and no other: runs/run/wave light "runs",
// fleet/host light "hosts", charts lights "charts".
export function nav(refresh) {
  const hdr = document.getElementById("hdr");
  const runs = route.page === "runs" || route.page === "run" || route.page === "wave";
  const fleetish = route.page === "fleet" || route.page === "host";
  const tab = (href, label, on) =>
      `<a${on ? ' class="on"' : ""} href="${keepHours(href)}">${label}</a>`;
  hdr.innerHTML = `<a class="brand" href="${keepHours("/")}">rlstack</a>`
    + `<nav class="tabs">`
    +   tab("/", "runs", runs)
    +   tab("/hosts", "hosts", fleetish)
    +   tab("/charts", "charts", route.page === "charts")
    + `</nav>`
    + `<span class="tools">${fleetish ? rangeLinks() : ""}</span>`;
  const tools = hdr.querySelector(".tools");
  if (route.runId) {
    const sel = el("select", {id: "switch", title: "switch experiment"});
    sel.append(el("option", {value: here()}, esc(route.runId)));
    // THE INDEX IS FETCHED WHEN THE SWITCHER IS REACHED FOR, not beside every
    // page: hovering or focusing it asks once (headerIndex dedupes)
    const fill = () => headerIndex(syncSwitcher);
    sel.addEventListener("mouseenter", fill);
    sel.addEventListener("focus", fill);
    sel.addEventListener("change", () => {
      if (sel.value && sel.value !== here()) location.href = sel.value;
    });
    tools.append(sel);
  }
  const again = el("button", {id: "refresh", title: "refresh (r)"}, "&#x21bb;");
  again.addEventListener("click", () => refresh());
  tools.append(again, el("span", {id: "asof"}));
}

// WHEN THE PAGE'S BYTES WERE FETCHED, said in the bar. The venue serves a
// reading from memory for up to 30 s and its snapshot is at most 12 s older,
// so this is the newest the page can honestly claim.
export function stamp(at) {
  const node = document.getElementById("asof");
  if (node) node.textContent = "as of " + new Date(at).toTimeString().slice(0, 8);
}

function here() {
  return runPath(route.runId, route.folder);
}

// THE SUBJECT BAR: what is being looked at, said once, under the nav. A page
// with no subject writes nothing and the bar is not there at all (#subject:empty).
export function ctx(html) {
  document.getElementById("subject").innerHTML = html;
}

let switcherSignature = null;

// THE INDEX IS HEADER FURNITURE, and never on a page's critical path.
// /api/runs walks every run in the store, and the only thing on a run or
// wave page that wants it is the switcher's list (the run's own name and
// tags now arrive with the page). So it is fetched when the switcher is
// reached for — at most one request at a time, at most once a minute — and
// the last good list is kept.
const INDEX_EVERY_MS = 60000;
let heldIndex = null;
let indexAt = 0;
let indexFlight = null;

export function headerIndex(use) {
  if (heldIndex !== null) use(heldIndex);
  if (indexFlight !== null || Date.now() - indexAt < INDEX_EVERY_MS) return;
  indexFlight = runsIndex().then(data => {
    indexFlight = null;
    if (data === null) return;       // a dead wire keeps the last good list
    indexAt = Date.now();
    heldIndex = data.runs;
    use(heldIndex);
  });
}

export async function runsIndex() {
  // /api/runs answers {now, runs}; every consumer reads it through here.
  // null means the POLL failed — a page keeps its last render and says so,
  // instead of mistaking a dead wire for an empty index
  const {getJSON} = await import("./dom.js");
  const data = await getJSON("/api/runs");
  return data ? {now: data.now, runs: data.runs || []} : null;
}

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
  const table = el("table", {}, "<tr><th>folder</th><th>run</th></tr>");
  for (const folder of folders) {
    const row = el("tr", {});
    row.append(el("td", {}, `<a href="${runPath(runId, folder)}">`
        + `${esc(folder || "(top)")}</a>`));
    row.append(el("td", {class: "k"}, esc(runId)));
    table.append(row);
  }
  holder.append(table);
}

