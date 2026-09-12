"""Serve an existing runtime's service doors as a persistent process.

    python examples/http_daemons.py --factory my_venue:services --port 8000

The factory is an async context manager yielding Callable[[str], Service]. It
owns ordinary desk registration, finite idle release and resident teardown;
this wrapper owns only HTTP. There is no second scheduler or model loader here.
"""
from __future__ import annotations

import argparse
import asyncio
from importlib import import_module
import os
import signal

from rlstack.runner.transports.http import HttpServer


async def serve(factory, port: int) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    async with factory() as service_for:
        async with HttpServer(service_for, port=port,
                              token=os.environ.get("RLSTACK_HTTP_TOKEN", "")) as server:
            print(f"ready: {server.endpoint}", flush=True)
            await stop.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="module:async_context_manager")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    module, sep, name = args.factory.partition(":")
    if not sep or not module or not name:
        parser.error("--factory needs module:async_context_manager")
    factory = getattr(import_module(module), name)
    asyncio.run(serve(factory, args.port))


if __name__ == "__main__":
    main()
