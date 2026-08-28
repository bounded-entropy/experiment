// The index: every experiment this reader's stores hold, newest first.
"use strict";

import {el, esc, getJSON} from "./dom.js";
import {ctx, hostPath, legend, runPath} from "./nav.js";

export async function drawIndex() {
  const runs = await getJSON("/api/runs") || [];
  const holder = document.getElementById("page");
  holder.innerHTML = "";
  ctx(`<span class="meta">${runs.length} experiment${runs.length === 1 ? "" : "s"}</span>`);
  if (!runs.length) {
    holder.append(el("p", {style: "padding:22px"},
        "no experiments journaled in this store yet"));
    return;
  }
  const table = el("table", {}, "<tr><th>run</th><th>status</th><th>committed</th>"
      + "<th>host(s)</th><th>store</th></tr>");
  for (const r of runs.slice().reverse()) {
    const row = el("tr", {});
    row.append(el("td", {}, `<a href="${runPath(r.run_id)}">${esc(r.run_id)}</a>`));
    row.append(el("td", {class: r.status === "running" ? "live" : ""},
                  esc(r.status) + (r.forked ? " <span class='warn'>⚠FORK</span>" : "")));
    row.append(el("td", {}, `${r.committed}/${esc(r.target)}`));
    row.append(el("td", {}, r.hosts.map(h =>
        `<a href="${hostPath(h)}">${esc(h)}</a>`).join(" + ")));
    row.append(el("td", {class: "k"}, esc(r.store)));
    table.append(row);
  }
  holder.append(table);
  legend("one experiment, one store (I10) · ⚠FORK = the same run_id in two stores"
    + " · hosts link to their journals · refreshes every 3s");
}
