"""A slow recovery reply must not prevent the new metal renewing its lease."""

import asyncio
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from venue_stub import modal_stubbed
from rlstack import LocalStore
from rlstack.runner.desk import Metal


class VenueReadinessTest(unittest.IsolatedAsyncioTestCase):
    async def test_heartbeats_continue_while_registration_recovers_runs(self):
        registration_started, heartbeat_seen = asyncio.Event(), asyncio.Event()
        release_registration = asyncio.Event()

        async def register(*args, **kwargs):
            registration_started.set()
            await release_registration.wait()
            return {}

        async def heartbeat(*args, **kwargs):
            heartbeat_seen.set()
            return {"heard": True}

        remote = SimpleNamespace(register_metal=register, heartbeat=heartbeat)
        source = Path(__file__).resolve().parents[1] / "deploy/modal_venue.py"
        with modal_stubbed(), tempfile.TemporaryDirectory() as temp:
            spec = importlib.util.spec_from_file_location("readiness_venue", source)
            venue = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = venue
            try:
                spec.loader.exec_module(venue)
                venue.HEARTBEAT_S = .005
                venue.a_store = lambda: LocalStore(temp)
                venue.store_volume = SimpleNamespace(commit=SimpleNamespace(aio=AsyncMock()))
                with patch("rlstack.runner.desk.MetalService.measure",
                           return_value=Metal("test", "L4", 1, 24)), \
                     patch("rlstack.runner.remote.RemoteDesk", return_value=remote), \
                     patch("rlstack.runner.remote.transport_for", return_value=None):
                    metal_type = venue.metal_class(venue.modal.App("test"), "test", "test",
                                                   "L4", None, module=spec.name)
                    metal = metal_type()
                    await metal.bring_up()
                    try:
                        await asyncio.wait_for(registration_started.wait(), 1)
                        await asyncio.wait_for(heartbeat_seen.wait(), 1)
                        self.assertFalse(release_registration.is_set())
                    finally:
                        metal.duties.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await metal.duties
            finally:
                sys.modules.pop(spec.name, None)
