"""rlstack CLI: thin argparse over the observer (rlstack/observe/).

    python -m rlstack hosts <store-locator> [<store-locator> ...]
    python -m rlstack runs  <store-locator> [<store-locator> ...]
    python -m rlstack gpu   <store-locator> [<store-locator> ...]

Locators resolve through observe.store_for — paths and file:// resolve
where mounted; modal:// tells you to run the reader beside the volume.
With no locators given, $RLSTACK_STORES (colon-separated) is used.
"""

from __future__ import annotations

import argparse
import os

from rlstack.observe import (
    render_gpu, render_hosts, render_runs, store_for,
)

VIEWS = {"hosts": render_hosts, "runs": render_runs, "gpu": render_gpu}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlstack")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in VIEWS:
        p = sub.add_parser(name)
        p.add_argument("store_locators", nargs="*",
                       help="store locators (default: $RLSTACK_STORES, "
                            "colon-separated)")
    args = parser.parse_args(argv)
    locators = args.store_locators or [
        r for r in os.environ.get("RLSTACK_STORES", "").split(":") if r]
    if not locators:
        parser.error("no store locators given and RLSTACK_STORES is unset")
    print(VIEWS[args.command]([store_for(loc) for loc in locators]), end="")


if __name__ == "__main__":
    main()
