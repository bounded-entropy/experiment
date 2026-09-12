// The index: every experiment this reader's roots hold, newest first, under
// the FOLDER it was born in.
//
// A folder is a store root chosen at birth and never moved (#58), so the tree
// here is not a filing system the UI maintains — it is where the runs already
// are. The name, tags and note are annotations read from each root's own
// annotations.jsonl; the observer renders them and writes nothing. The filter
// speaks the shared query grammar (matchExpr below — regex terms AND, | for
// OR, the same words observe/select.py reads), over the rows already fetched,
// so typing costs no request and the poll cannot yank the box out from under
// the cursor.
"use strict";

import {ago, drawnOnce, el, esc, lostTick, okTick, storeTail}
  from "./dom.js";
import {getJSON} from "./dom.js";
import {lazyDetails} from "./lazy.js";
import {ctx, hostPath, runPath, runsIndex, route} from "./nav.js";
import {branchEntries, branchRows, runLabel, runTree} from "./run_tree.js";

let needle = "";
let latest = [];
let serverNow = null;
const collapsed = new Set();
const expanded = new Set(JSON.parse(sessionStorage.getItem("run-folders") || "[]"));

const openBranches = new Set();
let searchingAll = false;

export async function drawIndex() {
  if (searchingAll) return drawSearchIndex();
  const params = new URLSearchParams();
  if (route.folder !== null) params.set("root", route.folder);
  const data = await getJSON("/api/browse?" + params);
  if (!data) {
    if (drawnOnce()) { lostTick(); return; }
    document.getElementById("page").textContent = "Could not load folders — press r to retry.";
    return;
  }
  const holder = document.getElementById("page");
  holder.replaceChildren();
  const toolbar = el("div", {class: "find"});
  const search = el("button", {type: "button"}, "Search all experiments");
  search.addEventListener("click", async () => {
    search.disabled = true;
    search.textContent = "Loading search index…";
    searchingAll = true;
    await drawSearchIndex();
  });
  toolbar.append(search);
  holder.append(toolbar);
  ctx('<span class="meta">Experiments · expand a folder to load its contents</span>');
  const tree = el("div", {id: "browse-tree"});
  holder.append(tree);
  appendEntries(tree, data, route.folder, "");
  okTick();
}

function appendEntries(holder, data, folder, path) {
  for (const entry of data.entries) {
    if (entry.kind === "run") {
      holder.append(el("div", {style: "padding:8px 12px"},
        `<a href="${runPath(entry.path, entry.folder)}">${esc(entry.name)}</a>`
        + ' <span class="k">open results</span>'));
    } else {
      const key = JSON.stringify([entry.folder, entry.path]);
      holder.append(lazyDetails(esc(entry.name) + "/", async body => {
        const next = await browse(entry.folder, entry.path);
        body.replaceChildren();
        appendEntries(body, next, entry.folder, entry.path);
      }, key, openBranches));
    }
  }
  if (!data.entries.length) holder.append(el("p", {class: "k"}, "No experiments in this folder yet."));
  if (data.next !== null) {
    const more = el("button", {type: "button"}, "Load more");
    more.addEventListener("click", async () => {
      more.disabled = true;
      try {
        const next = await browse(folder, path, data.next);
        more.remove();
        appendEntries(holder, next, folder, path);
      } catch (_) { more.disabled = false; more.textContent = "Retry loading more"; }
    });
    holder.append(more);
  }
}

async function browse(folder, path, offset = 0) {
  const params = new URLSearchParams({path, offset: String(offset)});
  if (folder !== null) params.set("root", folder);
  const data = await getJSON("/api/browse?" + params);
  if (!data) throw new Error("Folder read failed");
  return data;
}

async function drawSearchIndex() {
  const data = await runsIndex();
  if (!data) {                         // the freshness contract (dom.js)
    if (drawnOnce()) { lostTick(); return; }
    document.getElementById("page").textContent =
        "observer unreachable — retrying";
    return;
  }
  latest = data.runs;
  serverNow = data.now;
  frame();
  render();
  okTick();
}

// ---- the frame: built ONCE, so a poll never rebuilds the search box -------

function frame() {
  if (document.getElementById("tree")) return;
  const holder = document.getElementById("page");
  holder.innerHTML = "";
  const find = el("div", {class: "find"});
  const back = el("button", {type: "button"}, "Browse folders");
  back.addEventListener("click", () => { searchingAll = false; drawIndex(); });
  find.append(back);
  const box = el("input", {id: "q", type: "search", autocomplete: "off",
      placeholder: "Search runs or folders · tag:lora · name: / id: / note:"});
  box.setAttribute("aria-label", "Search runs or folders");
  box.value = needle;
  box.addEventListener("input", () => { needle = box.value; render(); });
  find.append(box);
  const close = el("button", {type: "button"}, "Collapse all");
  close.addEventListener("click", () => {
    expanded.clear();
    collapsed.clear();
    saveFolders();
    needle = "";
    box.value = "";
    render();
  });
  find.append(close);
  holder.append(find);
  holder.append(el("div", {id: "tree"}));
}

// THE QUERY GRAMMAR, same as the server's (observe/select.py): "|" separates
// OR clauses, whitespace is AND, each term is a case-insensitive regex over
// run_id + name + note + tags, falling back to substring when it does not
// compile.
export function matchExpr(run, q) {
  q = (q || "").trim();
  if (!q) return true;
  // unscoped terms NEVER search the note (prose poisons family filters);
  // tag: matches a whole tag exactly, name:/id:/note: search their field;
  // a pipe WITH SPACES is OR; an unspaced | stays inside its regex term
  const hay = [run.run_id, run.name || "", run.subdir || "", run.folder || ""]
      .concat(run.tags || []).join(" ");
  const search = (term, text) => {
    try { return new RegExp(term, "i").test(text); }
    catch (_) { return text.toLowerCase().includes(term.toLowerCase()); }
  };
  const whole = (term, tag) => {
    try { return new RegExp("^(?:" + term + ")$", "i").test(tag); }
    catch (_) { return tag.toLowerCase() === term.toLowerCase(); }
  };
  const one = term => {
    const at = term.indexOf(":");
    const scope = at > 0 ? term.slice(0, at) : "";
    const rest = at > 0 ? term.slice(at + 1) : "";
    if (rest && scope === "tag")
      return (run.tags || []).some(tag => whole(rest, tag));
    if (rest && scope === "name") return search(rest, run.name || "");
    if (rest && scope === "id") return search(rest, run.run_id || "");
    if (rest && scope === "note") return search(rest, run.note || "");
    return search(term, hay);
  };
  return q.split(/\s+\|\s+/).some(clause => {
    const terms = clause.trim().split(/\s+/).filter(Boolean);
    return terms.length > 0 && terms.every(one);
  });
}

// ---- the tree: one block per folder ---------------------------------------

function render() {
  const rows = latest.filter(r => matchExpr(r, needle));
  const folders = [...new Set(latest.map(r => r.folder))].sort();
  const many = folders.length > 1 || (folders.length === 1 && folders[0]);
  const groupCount = folders.reduce((n, folder) => {
    const tree = runTree(latest.filter(r => r.folder === folder));
    return n + tree.children.size + (tree.rows.length ? 1 : 0);
  }, 0);
  ctx(`<span class="meta">${latest.length} experiment${latest.length === 1 ? "" : "s"}`
    + (needle ? ` · ${rows.length} matching` : "")
    + ` · ${groupCount} folder${groupCount === 1 ? "" : "s"}`
    + `</span>`);
  const tree = document.getElementById("tree");
  tree.innerHTML = "";
  if (!latest.length) {
    tree.append(el("p", {style: "padding:22px"},
        "no experiments journaled in these stores yet"));
    return;
  }
  if (!rows.length) {
    tree.append(el("p", {class: "note"}, `nothing matching “${esc(needle)}”`));
  }
  for (const folder of folders) {
    const mine = rows.filter(r => r.folder === folder);
    if (!mine.length) continue;
    const all = latest.filter(r => r.folder === folder);
    tree.append(many ? folderBlock(folder, all) : dirBlocks(all, false, ""));
  }
}

function folderBlock(folder, rows) {
  const block = groupBlock(folder || "(top)", rows, JSON.stringify([folder]));
  block.append(dirBlocks(rows, true, folder));
  return block;
}

// ---- the filing inside one store: runs/<subdir>/<run_id> ------------------
// SUBDIRs keep their original storage addresses. The display nests slash
// paths and related named series, and turns single-run leaves into rows.
// Every link still uses the original (store folder, run_id).

function dirBlocks(rows, inFolder, folderKey) {
  return treeBlocks(runTree(rows), inFolder, [folderKey], true);
}

function treeBlocks(node, inFolder, path, top = false) {
  const frag = document.createDocumentFragment();
  const direct = node.rows.slice();
  const branches = [];
  for (const child of branchEntries(node)) {
    // A singleton leaf is a named experiment, not another folder to open.
    if (!top && !child.children.size && child.rows.length === 1)
      direct.push(...child.rows);
    else branches.push(child);
  }
  const visible = direct.filter(r => matchExpr(r, needle));
  if (visible.length) {
    if (top) {
      const block = groupBlock("Unfiled", direct, JSON.stringify([...path, ""]));
      block.append(table(visible, inFolder));
      frag.append(block);
    } else frag.append(table(visible, inFolder));
  }
  for (const child of branches) {
    const mine = branchRows(child);
    if (!mine.some(r => matchExpr(r, needle))) continue;
    const childPath = [...path, child.name];
    const block = groupBlock(child.name, mine, JSON.stringify(childPath));
    block.append(treeBlocks(child, inFolder, childPath));
    frag.append(block);
  }
  return frag;
}

function groupBlock(name, rows, key) {
  const mine = rows.filter(r => matchExpr(r, needle));
  const open = needle ? !collapsed.has(key) : expanded.has(key);
  const block = el("details", open ? {class: "group", open: "open"}
                                   : {class: "group"});
  const summary = el("summary", {},
      `<span class="key">${esc(name)}/</span>`
    + `<span class="k">${mine.length} experiment${mine.length === 1 ? "" : "s"}</span>`);
  const statuses = [...new Set(mine.map(r => r.status))].sort();
  for (const status of statuses) {
    const count = mine.filter(r => r.status === status).length;
    summary.append(el("span", {class: "k"}, `${count} ${esc(status)}`));
  }
  block.append(summary);
  // Native toggle events also fire when a render sets `open`. Save only a
  // connected group's actual change, so removed nodes cannot reset choices.
  block.addEventListener("toggle", () => {
    if (!block.isConnected) return;
    if (needle) {
      if (block.open) collapsed.delete(key); else collapsed.add(key);
    } else {
      if (block.open) expanded.add(key); else expanded.delete(key);
      saveFolders();
    }
  });
  return block;
}

function saveFolders() {
  sessionStorage.setItem("run-folders", JSON.stringify([...expanded]));
}

function table(rows, inFolder) {
  // inside a folder block the store column is the folder said twice — and
  // with ONE root it is the same absolute path on every row, which is a
  // column that carries no information. It appears when it distinguishes.
  const showStore = !inFolder && new Set(latest.map(r => r.store)).size > 1;
  const node = el("table", {}, "<tr><th>experiment</th><th>tags</th>"
      + "<th>status</th><th>committed</th><th>host(s)</th><th>last</th>"
      + (showStore ? "<th>store</th>" : "") + "</tr>");
  for (const r of rows.slice().sort((a, b) => runLabel(a).localeCompare(runLabel(b))
                                               || b.t - a.t)) {
    const folder = inFolder ? r.folder : null;
    const row = el("tr", {});
    const label = runLabel(r);
    const cell = el("td", {},
        `<a href="${runPath(r.run_ref || r.run_id, folder)}" title="${esc(r.note || "")}">`
      + `${esc(label)}</a>`
      + (label !== r.run_id ? ` <span class="k">${esc(r.run_id)}</span>` : ""));
    cell.setAttribute("title", [r.subdir, r.note].filter(Boolean).join(" · "));
    row.append(cell);
    row.append(el("td", {}, (r.tags || []).map(t =>
        `<span class="tag">${esc(t)}</span>`).join("") || "<span class='k'>—</span>"));
    const badge = r.status === "running" ? "live"
                : ["stalled", "parked", "lost"].includes(r.status) ? "stall" : "";
    row.append(el("td", {class: badge},
                  esc(r.status)
                + (r.stalled_hosts ? ` <span class="k" title="no heartbeat from: ${
                      esc(r.stalled_hosts.join(", "))}">⚠</span>` : "")
                // PARKED SAYS WHAT IT IS WAITING FOR (ADR 0008, F6): a run
                // the desk is holding is not a mystery, it is a shopping list
                + (r.parked ? ` <span class="k" title="${esc(parkedWhy(r.parked))}">⏸</span>` : "")
                + (r.lost_hosts ? ` <span class="k" title="alive but not carrying it: ${
                      esc(r.lost_hosts.join(", "))}">⚠</span>` : "")
                + (r.forked ? " <span class='warn'>⚠FORK</span>" : "")));
    row.append(el("td", {}, `${r.committed}/${esc(r.target)}`));
    row.append(el("td", {}, r.hosts.map(h =>
        `<a href="${hostPath(h, folder)}">${esc(h)}</a>`).join(" + ")));
    row.append(el("td", {class: "k when",
                         title: r.t ? esc(new Date(r.t * 1000).toLocaleString())
                                    : "No host activity timestamp recorded"},
                  r.t ? esc(ago(r.t, serverNow)) : "unknown"));
    if (showStore)
      row.append(el("td", {class: "k", title: esc(r.store)},
                    esc(storeTail(r.store))));
    node.append(row);
  }
  return node;
}

// PARKED, IN ONE LINE (ADR 0008, F6): why it is waiting, since when, and the
// metal that would free it — regimes, devices, GB per device.
export function parkedWhy(parked) {
  const wants = (parked.wants || []).map(w =>
      `${(w.regimes || []).join("+")} on ${w.devices}x`
    + `${w.vram_gb === null || w.vram_gb === undefined ? "a whole device"
                                                       : w.vram_gb + " GB"}`);
  const since = parked.since
      ? ` since ${new Date(parked.since * 1000).toLocaleString()}` : "";
  return `parked${since}: ${parked.reason || "waiting for metal"}`
       + (wants.length ? ` — wants ${wants.join("; ")}` : "");
}
