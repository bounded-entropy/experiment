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

import {ago, el, esc} from "./dom.js";
import {ctx, hostPath, legend, runPath, runsIndex} from "./nav.js";

let needle = "";
let latest = [];
let serverNow = null;
const collapsed = new Set();

export async function drawIndex() {
  const data = await runsIndex();
  latest = data.runs;
  serverNow = data.now;
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
      placeholder: "regex terms AND \u00b7 \u201c | \u201d for OR \u00b7 tag:lora exact \u00b7 name:/id:/note: scoped"});
  box.value = needle;
  box.addEventListener("input", () => { needle = box.value; render(); });
  find.append(box);
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
  const hay = [run.run_id, run.name || ""].concat(run.tags || []).join(" ");
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
    tree.append(many ? folderBlock(folder, mine) : dirBlocks(mine, false, ""));
  }
  legend("a folder IS a store root · <span class='key'>dir/</span> = the run's"
    + " filing subdir, chosen at submit and never moved · name and"
    + " tags are annotations — flavortext, never hashed · "
    + "<span class='stall'>stalled</span> = the journal says running but a host"
    + " of the run has no pulse (debug those first) · ⚠FORK = one run_id in two"
    + " roots · filter grammar: regex terms AND, | for OR · refreshes every 3s");
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
  block.append(dirBlocks(rows, true, folder));
  return block;
}

// ---- the filing inside one store: runs/<subdir>/<run_id> ------------------
// A SUBDIR is where the run's directory spawned at birth (submit's own
// indication) — organization inside one store, where the folder above is
// WHICH store. Top-level runs render bare; each subdir is its own block.

function dirBlocks(rows, inFolder, folderKey) {
  const bySub = new Map();
  for (const r of rows) {
    const sub = r.subdir || "";
    if (!bySub.has(sub)) bySub.set(sub, []);
    bySub.get(sub).push(r);
  }
  const frag = document.createDocumentFragment();
  const subs = [...bySub.keys()].sort();
  for (const sub of subs)
    if (!sub) frag.append(table(bySub.get(sub), inFolder));
  for (const sub of subs) {
    if (!sub) continue;
    const mine = bySub.get(sub);
    const key = folderKey + "//" + sub;
    const block = el("details", collapsed.has(key)
        ? {class: "group"} : {class: "group", open: "open"});
    block.append(el("summary", {},
        `<span class="key">${esc(sub)}/</span>`
      + `<span class="k">${mine.length} experiment${mine.length === 1 ? "" : "s"}</span>`));
    block.addEventListener("toggle", () => {
      if (block.open) collapsed.delete(key); else collapsed.add(key);
    });
    block.append(table(mine, inFolder));
    frag.append(block);
  }
  return frag;
}

function table(rows, inFolder) {
  // inside a folder block the store column is the folder said twice
  const node = el("table", {}, "<tr><th>experiment</th><th>tags</th>"
      + "<th>status</th><th>committed</th><th>host(s)</th><th>last</th>"
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
    const badge = r.status === "running" ? "live"
                : r.status === "stalled" ? "stall" : "";
    row.append(el("td", {class: badge},
                  esc(r.status)
                + (r.stalled_hosts ? ` <span class="k" title="no heartbeat from: ${
                      esc(r.stalled_hosts.join(", "))}">⚠</span>` : "")
                + (r.forked ? " <span class='warn'>⚠FORK</span>" : "")));
    row.append(el("td", {}, `${r.committed}/${esc(r.target)}`));
    row.append(el("td", {}, r.hosts.map(h =>
        `<a href="${hostPath(h, folder)}">${esc(h)}</a>`).join(" + ")));
    row.append(el("td", {class: "k", title: esc(new Date(r.t * 1000).toLocaleString())},
                  esc(ago(r.t, serverNow))));
    if (!inFolder) row.append(el("td", {class: "k"}, esc(r.store)));
    node.append(row);
  }
  return node;
}
