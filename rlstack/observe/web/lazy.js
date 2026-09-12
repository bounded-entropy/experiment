// A disclosure fetches only after it opens. Failed reads stay retryable.
import {el} from "./dom.js";

export function lazyDetails(label, load, key, expanded) {
  const node = el("details", {class: "group"});
  node.append(el("summary", {}, label));
  const body = el("div", {style: "padding:12px"});
  node.append(body);
  let pending = false, loaded = false;
  async function reveal() {
    if (!node.open || pending || loaded) return;
    pending = true;
    body.textContent = "Loading…";
    try {
      await load(body);
      loaded = true;
    } catch (_) {
      body.textContent = "Could not load this section. ";
      const retry = el("button", {type: "button"}, "Retry");
      retry.addEventListener("click", reveal);
      body.append(retry);
    } finally { pending = false; }
  }
  node.addEventListener("toggle", () => {
    if (!node.isConnected) return;
    if (node.open) expanded.add(key); else expanded.delete(key);
    reveal();
  });
  if (expanded.has(key)) node.open = true;
  return node;
}
