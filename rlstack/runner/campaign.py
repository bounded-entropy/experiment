"""The campaign layer: where SPECS meet the fleet.

The desk is workload-blind (runner/desk.py) — its vocabulary is Demands and
frames. This module is the other side of that boundary, the only place that
turns an ExperimentSpec INTO the desk's vocabulary: `demands_of` reads the
topology as capability demands and marks the learner demand as the ANCHOR
(the learner is never remote, so the frame lands where the learner builds —
that rule lives here, with the spec, never at the desk), and `frame_for`
wraps the canonical row with the client's code claim.

`Campaigns` is the desk's spec-aware sidecar: it owns the passes that need
BOTH a store and spec knowledge — today `migrate`, the code-refresh warm-
fork — and serves them over the same Transport contract by wrapping a Desk,
so a venue wires ONE door and the desk core still never imports a spec
class. It writes the fleet journal through the desk's own store handle,
inside the desk's container: the single-writer property is the container's,
not the class's.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from rlstack.runner.desk import Demand, Desk, DeskError, demand_rows
from rlstack.spec.specs import ExperimentSpec, PoolMember


def demands_of(spec: ExperimentSpec) -> tuple[Demand, ...]:
    """The spec's topology as capability demands, learner marked ANCHOR.
    One HostSpec is one placement unit (its index is the demand's `group`),
    and `vram_gb` passes through AS DECLARED — total across shards, None for
    a whole device per shard. The spec speaks GB and so does the desk; the
    crossing to a partition's fraction happens once, at the metal's build,
    where the card is known (ADR 0001) — never here."""
    out: list[Demand] = []
    for hi, host in enumerate(spec.topology.hosts):
        for member in host.members:
            if isinstance(member, PoolMember):
                out.append(Demand(
                    pool=member.name, capability="inference",
                    base=member.base or spec.policy.base, shape=member.tp,
                    vram_gb=member.vram_gb, group=hi))
            else:
                out.append(Demand(
                    pool=None, capability="training", base=spec.policy.base,
                    shape=member.fsdp, vram_gb=member.vram_gb, group=hi,
                    anchor=True))
    return tuple(out)


def frame_for(spec: ExperimentSpec, subdir: str | None = None) -> dict:
    """The opaque frame a submission delivers: the canonical row, the
    client's code claim (the hashes the anchor host diffs against its own
    registry — the skew check's evidence), and the filing hint."""
    from rlstack.registry import code_hashes
    from rlstack.spec.canonical import canonical_json

    return {"spec": json.loads(canonical_json(spec)),
            "code": code_hashes(spec), "subdir": subdir}


class Campaigns:
    """The desk's spec-aware sidecar: passes that read stores AND specs,
    served over the desk's own Transport contract so a venue wires one door."""

    def __init__(self, desk: Desk) -> None:
        self.desk = desk
        self.store = desk.store

    async def submit(self, spec: ExperimentSpec,
                     subdir: str | None = None) -> dict:
        """A spec through the blind door: shaped here, delivered there."""
        return await self.desk.submit(demand_rows(demands_of(spec)),
                                      frame_for(spec, subdir))

    async def serve(self, verb: str, payload: dict) -> dict:
        if verb == "migrate":
            return await self.migrate(
                payload["run_ids"], optim=payload.get("optim", "load"),
                remaining_only=bool(payload.get("remaining_only", False)))
        return await self.desk.serve(verb, payload)

    def answer(self, verb: str, payload: dict) -> dict:
        return self.desk.answer(verb, payload)

    # ---- migrate: warm-fork running work onto the current code --------------

    async def migrate(self, run_ids: Sequence[str], *,
                      optim: str = "load",
                      remaining_only: bool = False) -> dict:
        """Warm-fork each named run into a NEW experiment continuing from its
        ledger tail — the code-refresh move: containers were replaced, the
        old runs' code hashes no longer exist anywhere, so each run's spec is
        resubmitted with init=WarmStart(parent@tail) and becomes a new
        run_id under the CURRENT code, moments included (optim="load" reads
        the tail, the one version retention keeps).

        `remaining_only` also slices the plans to the waves the parent never
        committed, so the child runs exactly what was left. That is only
        derivable for the standard on-policy pairing (train wave u = WaveRef
        "self://rollouts/u", 1:1); a run whose plans say anything fancier is
        REFUSED for slicing — migrate it full-plan instead, deliberately.

        Every child is journaled as a `migrate` event (parent, child,
        version) — lineage on the fleet plane, beside the manifest's own
        parent record. Per-run failures land in the report; one bad run never
        stops the pass.
        """
        import dataclasses
        import time

        from rlstack.runner.remote import spec_from_json
        from rlstack.spec.specs import WarmStart

        report: dict[str, dict] = {}
        for rid in run_ids:
            try:
                manifest = self.store.peek_manifest(rid)
                if manifest is None:
                    raise DeskError(f"no manifest for {rid!r} in this store")
                spec = spec_from_json(manifest["spec"])
                entries = self.store.peek_ledger(rid)
                if not entries:
                    raise DeskError(
                        f"{rid!r} committed nothing — resubmit it plainly; "
                        f"a warm start from nowhere is a fresh run wearing "
                        f"a parent it never had")
                tail = entries[-1]
                committed = int(tail["update"])
                version = max(int(v) for v in tail["versions"].values())
                child = dataclasses.replace(
                    spec, init=WarmStart(policy=f"store://{rid}@{version}",
                                         optim=optim))
                if remaining_only:
                    child = dataclasses.replace(
                        child, plans=self.sliced_plans(rid, spec, committed))
                reply = await self.submit(child)
                if reply.get("accepted"):
                    self.store.append_fleet_event({
                        "event": "migrate", "t": time.time(), "parent": rid,
                        "child": reply.get("run_id"), "version": version,
                        "remaining_only": remaining_only})
                report[rid] = {**reply, "parent_version": version}
            except Exception as refusal:
                report[rid] = {"accepted": False, "error": str(refusal)}
        return report

    def sliced_plans(self, rid: str, spec: ExperimentSpec, committed: int):
        """The parent's plans, minus everything its ledger already committed.

        Rollout waves past `committed` re-index from 1; the train plan is
        REBUILT as the standard pairing after PROVING the parent's was exactly
        that (every entry WaveRef "self://rollouts/u", 1:1) — self:// refs are
        index-coupled, so slicing anything fancier would silently retarget
        them. Refusals over guesses, everywhere. (Measurement has no plan to
        slice: it is not part of the run.)
        """
        from rlstack.data.plan import RunPlan, WaveRef, decode, encode
        from rlstack.spec.specs import Plans

        def plan_of(kind: str) -> RunPlan | None:
            data = self.store.peek_plan(rid, kind)
            return None if data is None else decode(data)

        train = plan_of("train")
        rollout = plan_of("rollout")
        if train is None or rollout is None:
            raise DeskError(
                f"{rid!r} carries no sliceable plans (pre-#59 run?) — "
                f"migrate it full-plan")
        expected = tuple(WaveRef(f"self://rollouts/{u}")
                         for u in range(1, len(train) + 1))
        if train.waves != expected or len(rollout) != len(train):
            raise DeskError(
                f"{rid!r}'s plans are not the standard on-policy pairing — "
                f"self:// refs are index-coupled, so slicing would retarget "
                f"them. Migrate it full-plan (remaining_only=False)")
        remaining = len(train) - committed
        if remaining <= 0:
            raise DeskError(f"{rid!r} committed its whole plan — nothing "
                             f"remaining to migrate")
        return Plans(
            train=self.store.cas_put(encode(RunPlan(tuple(
                WaveRef(f"self://rollouts/{u}")
                for u in range(1, remaining + 1))))),
            rollout=self.store.cas_put(
                encode(RunPlan(rollout.waves[committed:]))))
