"""Shared initial and per-update tensor publication; control messages only."""

from __future__ import annotations

import asyncio
from rlstack.policy.state import StateLayout
from rlstack.runner.signals import store_work


async def publish_tensors(learner, tenant, versions, servable, adapter_types,
                          pools, transfer, journal, update):
    """The caller holds learner admission through capture and all acks.

    A failed or uncertain transfer never replays an optimizer step. Requests
    keep pinning their immutable old versions; only a complete new bundle
    can be exposed by the caller's ledger commit.
    """
    from rlstack.policy.compile import compile_tensor_pin
    manifest = await asyncio.to_thread(learner.prepare_transfer,
        tenant, versions, servable, adapter_types)
    receipts = []
    prepared = []
    failed = None
    try:
        for pool in pools.values():
            targets = pool.live()
            if not targets:
                raise RuntimeError(f"no live destination for pool {pool.name!r}")
            for replica in targets:
                prepared.append(replica.engine)
                endpoint = await asyncio.to_thread(
                    replica.engine.prepare_transfer, manifest, transfer)
                if "complete" in endpoint:
                    receipt = endpoint["complete"]
                else:
                    sent = await asyncio.to_thread(learner.send_transfer,
                        manifest["token"], endpoint, transfer.timeout_s)
                    receipt = await asyncio.to_thread(replica.engine.finish_transfer,
                        manifest["source"], manifest["token"])
                    receipt = {**receipt, **sent, "channel": endpoint["channel"],
                               "host": endpoint["host"]}
                if receipt["bundle_id"] != manifest["bundle_id"]:
                    raise ValueError("inference acknowledged a different policy version")
                receipts.append({"pool": pool.name, "replica": replica.index, **receipt})
        pin = compile_tensor_pin(
            {name: StateLayout.from_row(row) for name, row in manifest["states"].items()},
            versions, adapter_types, manifest.get("code"))
        if journal is not None:
            await store_work(journal.append, {"event": "weight-transfer",
                "run_id": tenant, "update": update, "backend": "nccl",
                "bundle_id": pin.bundle_id, "versions": dict(versions),
                "source": manifest["source"], "token": manifest["token"],
                "materialize_s": manifest.get("materialize_s"),
                "capture_s": manifest.get("capture_s"),
                "hash_s": manifest["hash_s"], "destinations": receipts})
        return pin
    except BaseException as error:
        failed = error
        for engine in prepared:
            try:
                await asyncio.to_thread(engine.abort_transfer,
                    manifest["source"], manifest["token"])
            except BaseException as cleanup:
                error.add_note(f"publication cleanup also failed: {cleanup}")
        raise
    finally:
        try:
            await asyncio.to_thread(learner.release_transfer, manifest["token"])
        except BaseException as cleanup:
            if failed is None:
                raise
            failed.add_note(f"source snapshot release also failed: {cleanup}")
