// A compact view of existing filing paths. Nothing here changes a run's
// store, subdir or identity. A named series (sweep and sweep-controls, for
// example) shares a branch; an otherwise single-run leaf becomes a row.
"use strict";

export function runLabel(run) {
  return run.name || ((run.subdir || "").includes("/")
      ? run.subdir.split("/").pop() : run.run_id);
}

export function runTree(rows) {
  const top = branch("");
  for (const run of rows) {
    let at = top;
    for (const segment of (run.subdir || "").split("/").filter(Boolean)) {
      if (!at.children.has(segment)) at.children.set(segment, branch(segment));
      at = at.children.get(segment);
    }
    at.rows.push(run);
  }
  nestSeries(top);
  return top;
}

function branch(name) {
  return {name, rows: [], children: new Map()};
}

function nestSeries(node) {
  // Only an existing sibling can be a series parent. A coincidental shared
  // word alone never invents a project ("train-a" and "train-b" stay peers).
  const names = [...node.children.keys()].sort((a, b) => b.length - a.length);
  for (const name of names) {
    const parent = names.find(other => name.startsWith(other + "-"));
    if (!parent) continue;
    const child = node.children.get(name);
    const label = name.slice(parent.length + 1);
    // Both a/b and a-b may already exist. They are distinct filings; never
    // overwrite one simply because their compact labels would coincide.
    if (node.children.get(parent).children.has(label)) continue;
    child.name = label;
    node.children.get(parent).children.set(child.name, child);
    node.children.delete(name);
  }
  for (const child of node.children.values()) nestSeries(child);
}

export function branchRows(node) {
  return node.rows.concat(...[...node.children.values()].map(branchRows));
}

export function branchEntries(node) {
  return [...node.children.values()].sort((a, b) => a.name.localeCompare(b.name));
}
