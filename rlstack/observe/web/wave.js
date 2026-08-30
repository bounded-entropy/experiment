// One SEALED wave, read from the store (#57's charter amendment: sealed,
// content-addressed artifacts through peek verbs — never live state, never an
// attach, never a write). Two readings of the same bytes: the DISTRIBUTIONS
// the server computed once, and the trajectories themselves, grouped as they
// were trained and readable as chat.
//
// This page never polls: a committed wave is immutable, so there is nothing to
// poll for.
"use strict";

import {C, brief, el, esc, getAnswer, getJSON, note, raw, section} from "./dom.js";
import {histogramCard} from "./charts.js";
import {apiRun, ctx, drawAmbiguity, legend, route, runPath, syncSwitcher}
  from "./nav.js";

const FINISH_COLOR = {stop: C.rail, eos: C.feed, length: C.warn};

export async function drawWave() {
  const [answer, runs] = await Promise.all([
    getAnswer(apiRun(route.runId, "/wave/" + route.update, route.folder)),
    getJSON("/api/runs").then(d => (d && d.runs) || [])]);
  const wave = answer.data;
  const holder = document.getElementById("page");
  holder.innerHTML = "";
  if (wave && wave.ambiguous) { drawAmbiguity(route.runId, wave.ambiguous); return; }
  syncSwitcher(runs || []);
  if (!wave) {
    ctx(`<a href="${runPath(route.runId, route.folder)}" class="nav">`
      + `${esc(route.runId)}</a>`);
    // a sealed wave never polls — so a DOWN wire retries itself; only the
    // server's own "no" (a positive 404) stands as the answer
    holder.textContent = answer.missing ? "no such sealed wave"
                                        : "observer unreachable — retrying";
    if (!answer.missing) setTimeout(drawWave, 3000);
    return;
  }
  const s = wave.summary;
  ctx(`<a href="${runPath(route.runId, route.folder)}" style="color:#5fb2ff">${esc(route.runId)}</a>`
    + `<span>wave ${wave.update}</span>`
    + `<span class="meta">${s.trajectories} trajectories · ${s.groups} groups`
    + ` · ${s.tokens} generated tokens</span>`
    + (s.kl_mean !== null && s.kl_mean !== undefined
        ? `<span class="meta">mean KL ${brief(s.kl_mean)} nats</span>` : ""));

  drawDistributions(s);
  drawColumns(s, wave.ledger);
  drawTrajectories(wave);
  legend("this wave is sealed: runs/&lt;id&gt;/waves/" + String(wave.update).padStart(6, "0")
    + ".jsonl.gz plus its postdata, read through the observer's peek verbs · "
    + "hover a histogram bar for its edges and count · a bubble with a blue edge "
    + "was GENERATED (one Turn: one pinned bundle, one seed, one contiguous KV)");
}

// ---- the distributions ----------------------------------------------------

function drawDistributions(s) {
  const grid = section("distribution", "this wave's own bytes, binned server-side");
  for (const panel of s.panels) grid.append(histogramCard(panel));
  grid.append(finishCard(s.finish));
}

function finishCard(finish) {
  const node = el("div", {class: "card"});
  const total = Object.values(finish).reduce((a, b) => a + b, 0);
  node.append(el("div", {}, `<span class="name">finish reason</span>`
      + `<span class="who">per generated turn</span>`));
  const bar = el("div", {style: "display:flex; height:14px; border-radius:3px; "
      + "overflow:hidden; margin:10px 0 8px; background:#141a21"});
  for (const [reason, count] of Object.entries(finish)) {
    if (!count) continue;
    bar.append(el("div", {
      style: `flex:${count} 0 0; background:${FINISH_COLOR[reason] || C.dim}; `
             + "opacity:0.8",
      title: `${reason} ${count}`}));
  }
  node.append(bar);
  node.append(el("div", {class: "mix"}, Object.entries(finish).map(([reason, count]) =>
      `<span><span style="color:${FINISH_COLOR[reason] || C.dim}">■</span> `
      + `${esc(reason)} <b>${count}</b></span>`).join("")));
  node.append(el("div", {class: "now"}, total
      ? `${total} turn${total === 1 ? "" : "s"}` : "no turns"));
  return node;
}

// ---- the columns, as the ledger summarized them ---------------------------

function drawColumns(s, ledger) {
  const holder = document.getElementById("page");
  holder.append(el("h2", {}, "postdata <span>every column the pipeline wrote for "
      + "this wave</span>"));
  if (!s.columns.length) { note("this wave has no postdata columns"); return; }
  const table = el("table", {}, "<tr><th>column</th><th>kind</th><th>n</th>"
      + "<th>mean</th><th>min</th><th>max</th><th>ledger mean</th></tr>");
  const means = (ledger || {}).post || {};
  for (const c of s.columns) {
    const row = el("tr", {});
    row.append(el("td", {}, esc(c.name)));
    row.append(el("td", {class: "k"}, esc(c.kind)));
    row.append(el("td", {class: "n k"}, String(c.n)));
    row.append(el("td", {class: "n"}, raw(c.mean)));
    row.append(el("td", {class: "n k"}, raw(c.min)));
    row.append(el("td", {class: "n k"}, raw(c.max)));
    row.append(el("td", {class: "n k"}, c.name in means ? raw(means[c.name]) : "—"));
    table.append(row);
  }
  holder.append(table);
}

// ---- the trajectories, grouped as they were trained -----------------------

function drawTrajectories(wave) {
  const holder = document.getElementById("page");
  holder.append(el("h2", {}, "trajectories <span>by Group — the scope a "
      + "postprocessor saw</span>"));
  if (wave.truncated)
    note(`showing the first ${wave.summary.trajectories} of `
       + `${wave.summary.trajectories + wave.truncated} trajectories`);
  const holder2 = el("div", {class: "wave"});
  wave.groups.forEach((group, index) => holder2.append(groupBlock(group, index === 0)));
  holder.append(holder2);
}

function groupBlock(group, open) {
  const block = el("details", open ? {class: "group", open: "open"} : {class: "group"});
  const rewards = group.trajectories
      .map(t => (t.post.find(p => p.name === "reward") || {}).value)
      .filter(v => v !== undefined && v !== null);
  const summary = el("summary", {},
    `<span class="key">${esc(group.key)}</span>`
    + `<span class="k">${group.trajectories.length} trajector`
    + `${group.trajectories.length === 1 ? "y" : "ies"}</span>`
    + (rewards.length ? `<span class="k">reward mean ${brief(
        rewards.reduce((a, b) => a + b, 0) / rewards.length)}</span>` : "")
    + `<span class="k">${group.trajectories.reduce((a, t) => a + t.tokens, 0)} tok</span>`);
  block.append(summary);
  for (const traj of group.trajectories) block.append(trajectoryBlock(traj));
  return block;
}

function trajectoryBlock(traj) {
  const node = el("div", {class: "traj"});
  const facts = [`<span class="k">#${traj.index}</span> <b>${esc(traj.task.id)}</b>`];
  for (const fact of traj.post)
    facts.push(`${esc(fact.name)} <b>${brief(fact.value)}</b>`
      + (fact.kind === "token_level"
          ? ` <span class="k">mean/${fact.tokens} tok</span>` : ""));
  if (traj.kl !== null && traj.kl !== undefined)
    facts.push(`KL <b>${brief(traj.kl)}</b> <span class="k">nats</span>`);
  facts.push(`<b>${traj.tokens}</b> tok`);
  if (traj.finish) facts.push(`finish <b>${esc(traj.finish)}</b>`);
  if (traj.bundle_id) facts.push(`<span class="k">${esc(traj.bundle_id)}</span>`);
  const version = Object.entries(traj.policy_version)
      .map(([k, v]) => `${k}@${v}`).join(" ");
  if (version) facts.push(`<span class="k">${esc(version)}</span>`);
  node.append(el("div", {class: "facts"},
      facts.map(fact => `<span>${fact}</span>`).join("")));

  const chat = el("div", {class: "chat"});
  for (const message of traj.messages) chat.append(bubble(message, traj));
  node.append(chat);
  return node;
}

function bubble(message, traj) {
  const generated = message.turn !== null && message.turn !== undefined;
  const row = el("div", {class: generated ? "msg gen" : "msg"});
  row.append(el("div", {class: "role"}, esc(message.role)));
  const body = el("div", {class: "bubble"});
  if (generated) {
    const turn = traj.turns[message.turn] || {};
    body.append(el("span", {class: "tag"},
      `${message.tokens} tok · ${esc(message.finish)}`
      + (turn.seed !== undefined ? ` · seed ${turn.seed}` : "")
      + (turn.behavior_logprob_mean !== null && turn.behavior_logprob_mean !== undefined
          ? ` · lp ${brief(turn.behavior_logprob_mean)}` : "")));
  }
  const text = el("span", {});
  text.textContent = message.content;      // arbitrary sampled text: never HTML
  body.append(text);
  row.append(body);
  return row;
}
