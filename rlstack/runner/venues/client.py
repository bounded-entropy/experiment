"""Provider-independent spec submission and progress through a Desk and observer."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
import json
import tempfile
import time
from urllib.parse import quote, urlencode
from urllib.request import urlopen

from rlstack.data.stores.base import Store, run_reference
from rlstack.data.stores.local import LocalStore
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.remote import RemoteDesk, spec_from_json
from rlstack.spec.canonical import canonical_json
from rlstack.spec.specs import ExperimentSpec


class VenueClient:
    """A caller chooses endpoints; scientific frames and progress stay shared."""

    def __init__(self, desk: RemoteDesk, observer: str = "") -> None:
        self.desk = desk
        self.observer = observer

    async def wait_for_metal(self, name: str, timeout_s: float = 900.0) -> dict:
        """Readiness requires registration, a live lease and reachable capacity.

        Bound the polling window, not an individual Desk input: cancelling a
        provider RPC is not proof that its server-side work stopped.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            held = (await self.desk.status()).get("metal", {})
            if (name in held and held[name].get("plane") and held[name].get("live")
                    and held[name].get("residual") is not None):
                return held[name]
            await asyncio.sleep(10)
        raise SystemExit(f"{name} did not register within {timeout_s:.0f}s")

    async def guarded_release(self, names: Iterable[str], reason: str) -> dict:
        """Release only this caller's named metal, subject to the Desk's busy guard."""
        held = (await self.desk.status()).get("metal", {})
        out: dict[str, dict] = {}
        for name in sorted(names):
            if not held.get(name, {}).get("plane"):
                continue
            told = await self.desk.release(name, reason=reason)
            print(f"[release] {name}: {json.dumps(told)}", flush=True)
            out[name] = told
        return out

    async def mine_are_released(self, names: Iterable[str]) -> bool:
        """Another caller's live allocation does not invalidate this one's cleanup."""
        held = (await self.desk.status()).get("metal", {})
        standing = [name for name in sorted(names) if held.get(name, {}).get("plane")]
        print(f"[plane] mine still standing: {standing or 'none'}", flush=True)
        return not standing

    async def canonical_row(
        self, build: Callable[[Store], ExperimentSpec | dict[str, ExperimentSpec]],
        borrow: Iterable[str] = (),
    ) -> dict:
        """Build canonical specs locally and publish their plan bytes via Desk."""
        with tempfile.TemporaryDirectory() as root:
            store = LocalStore(root)
            for uri in borrow:
                store.cas_put(await self.desk.read_cas(uri))
            built = build(store)
            specs = built if isinstance(built, dict) else {"": built}
            rows = {name: json.loads(canonical_json(spec)) for name, spec in specs.items()}
            for key in store._list("cas"):
                await self.desk.put_plan(store._read(key))
        return rows if isinstance(built, dict) else rows[""]

    async def submit(self, row: dict, subdir: str, *, anchor: str | None = None,
                     solo: bool = False, resume: bool = False,
                     checkpointing: Checkpointing) -> dict:
        """Submit the same canonical row through either provider's transport,
        with the submission's own Checkpointing (ADR 0014) — no default."""
        return await self.desk.submit(spec_from_json(row), subdir=subdir,
                                      anchor=anchor, solo=solo, resume=resume,
                                      checkpointing=checkpointing)

    def progress(self, run_id: str, folder: str = "", tail: int = 3) -> dict:
        """Read committed progress from the existing observer API."""
        if not self.observer:
            raise SystemExit("RLSTACK_OBSERVER is unset: set the observer's base URL")
        subdir, _, name = run_id.rpartition("/")
        query = urlencode({"root": folder, "subdir": subdir})
        url = f"{self.observer.rstrip('/')}/api/run/{quote(name, safe='')}?{query}"
        with urlopen(url, timeout=60) as answer:
            told = json.loads(answer.read().decode("utf-8"))
        if "run_id" not in told:
            raise SystemExit(f"the observer does not know run {run_id}: {told}")
        updates = told["updates"]
        return {"extent": told["extent"], "completed": told["committed"],
                "planned": told["target"], "done": bool(told["done"]),
                "committed": int(updates[-1]["update"]) if updates else 0,
                "train": [dict(u.get("train", {})) for u in updates[-tail:]]}

    def ledgers(self, run_ids: Iterable[str], folder: str = "", tail: int = 8) -> dict:
        return {run_id: self.progress(run_id, folder, tail) for run_id in run_ids}

    async def follow(self, run_id: str, timeout_s: float, every_s: float = 60.0,
                     folder: str = "") -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            told = await asyncio.to_thread(self.progress, run_id, folder)
            print(f"[{run_id[:12]}] {told['completed']}/{told['planned']} "
                  f"{told['extent']} — {json.dumps(told['train'])}", flush=True)
            if told["done"]:
                return run_id
            await asyncio.sleep(every_s)
        raise SystemExit(f"{run_id} did not finish within {timeout_s:.0f}s")

    async def submit_and_follow(self, row: dict, subdir: str, timeout_s: float,
                                anchor: str | None = None, *,
                                checkpointing: Checkpointing) -> str:
        reply = await self.submit(row, subdir, anchor=anchor, checkpointing=checkpointing)
        if not reply.get("accepted"):
            raise SystemExit(f"not accepted: {reply}")
        return await self.follow(run_reference(reply["run_id"], subdir), timeout_s)
