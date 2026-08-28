// The index: every experiment this reader's roots hold, newest first, under
// the FOLDER it was born in.
//
// A folder is a store root chosen at birth and never moved (#58), so the tree
// here is not a filing system the UI maintains — it is where the runs already
// are. The name, tags and note are annotations read from each root's own
// annotations.jsonl; the observer renders them and writes nothing. The search
// is a case-insensitive substring over name + tags + note + run_id, and
// nothing fancier — it filters the rows already fetched, so typing costs no
// request and the poll cannot yank the box out from under the cursor.
"use strict";

import {el, esc, getJSON} from "./dom.js";
import {ctx, hostPath, legend, runPath} from "./nav.js";

let needle = "";
let latest = [];
const collapsed = new Set();

export async function drawIndex() {
  latest = await getJSON("/api/runs") || [];
  frame();
  render();
}

// ---- the frame: built ONCE, so a poll never rebuilds the search box -------

function frame() {
  if (document.getElementById("tree")) return;
  const holder = document.getElementById("page");
  holder.innerHTML = "";
  const find = el("div", {class: "find"});
  const box = el("input", {id: "q", type: "search", autocomplete: "off",
      placeholder: "search names, tags, notes, run ids"});
  box.value = needle;
  box.addEventListener("input", () => { needle = box.value; render(); });
  find.append(box);
  holder.append(find);
  holder.append(el("div", {id: "tree"}));
}

function matches(run, q) {
  if (!q) return true;
  const hay = [run.run_id, run.name || "", run.note || ""]
      .concat(run.tags || []).join(" ").toLowerCase();
  return hay.includes(q.toLowerCase());
}

// ---- the tree: one block per folder ---------------------------------------

function render() {
  const rows = latest.filter(r => matches(r, needle));
  const folders = [...new Set(latest.map(r => r.folder))].sort();
  const many = folders.length > 1 || (folders.length === 1 && folders[0]);
  ctx(`<span class="meta">${latest.length} experiment${latest.length === 1 ? "" : "s"}`
    + (needle ? ` · ${rows.length} matching` : "")
    + (many ? ` · ${folders.length} folder${folders.length === 1 ? "" : "s"}` : "")
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
    tree.append(many ? folderBlock(folder, mine) : table(mine, false));
  }
  legend("a folder IS a store root, chosen when the run was started and never"
    + " moved · name and tags are annotations (annotations.jsonl beside runs/)"
    + " — flavortext, never hashed, never read by an experiment · ⚠FORK = the"
    + " same run_id in more than one root · refreshes every 3s");
}

function folderBlock(folder, rows) {
  const block = el("details", collapsed.has(folder)
      ? {class: "group"} : {class: "group", open: "open"});
  block.append(el("summary", {},
      `<span class="key">${esc(folder || "(top)")}</span>`
    + `<span class="k">${rows.length} experiment${rows.length === 1 ? "" : "s"}</span>`));
  block.addEventListener("toggle", () => {
    if (block.open) collapsed.delete(folder); else collapsed.add(folder);
  });
  block.append(table(rows, true));
  return block;
}

function table(rows, inFolder) {
  // inside a folder block the store column is the folder said twice
  const node = el("table", {}, "<tr><th>experiment</th><th>tags</th>"
      + "<th>status</th><th>committed</th><th>host(s)</th>"
      + (inFolder ? "" : "<th>store</th>") + "</tr>");
  for (const r of rows.slice().reverse()) {
    const folder = inFolder ? r.folder : null;
    const row = el("tr", {});
    row.append(el("td", {},
        `<a href="${runPath(r.run_id, folder)}" title="${esc(r.note || "")}">`
      + `${esc(r.name || r.run_id)}</a>`
      + (r.name ? ` <span class="k">${esc(r.run_id)}</span>` : "")));
    row.append(el("td", {}, (r.tags || []).map(t =>
        `<span class="tag">${esc(t)}</span>`).join("") || "<span class='k'>—</span>"));
    row.append(el("td", {class: r.status === "running" ? "live" : ""},
                  esc(r.status) + (r.forked ? " <span class='warn'>⚠FORK</span>" : "")));
    row.append(el("td", {}, `${r.committed}/${esc(r.target)}`));
    row.append(el("td", {}, r.hosts.map(h =>
        `<a href="${hostPath(h, folder)}">${esc(h)}</a>`).join(" + ")));
    if (!inFolder) row.append(el("td", {class: "k"}, esc(r.store)));
    node.append(row);
  }
  return node;
}
