"""THE document and its assets: the UI is real files, and this is their reader.

No build step, no CDN, no framework, no node toolchain — rlstack/observe/web/
holds the page exactly as it ships: index.html, style.css, and native ES modules
the browser loads itself. Every PAGE route returns index.html (the modules route
on location.pathname, so all six pages are the same document); /web/<file>
returns one asset.

The files are PACKAGE DATA read through importlib.resources, so every venue
serves the same bytes: `python -m rlstack ui <store>` off the source tree, and
any deployed image — whose build must ship web/ beside the .py sources
explicitly, because source-only packaging drops non-.py files (measured: the
page 200s with an empty body when web/ is missing; deploy/ui.py shows the
shape).
"""

from __future__ import annotations

from importlib.resources import files

WEB = files("rlstack.observe") / "web"
WEB_PREFIX = "/web/"
DOCUMENT = "index.html"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


def document() -> bytes:
    """index.html — the one document every page route returns. A missing file
    is a broken deployment, not a 404: it raises."""
    return (WEB / DOCUMENT).read_bytes()


def asset(name: str) -> tuple[bytes, str] | None:
    """One file under web/ as (bytes, content type), or None when the name is
    not one of ours.

    AN ASSET IS ONE FILE NAME: web/ is flat, so a name with a separator in it
    is not a file we serve — the reader never walks a path.
    """
    suffix = name[name.rfind("."):] if "." in name else ""
    if "/" in name or name.startswith(".") or suffix not in CONTENT_TYPES:
        return None
    try:
        return (WEB / name).read_bytes(), CONTENT_TYPES[suffix]
    except (FileNotFoundError, OSError):
        return None


def asset_names() -> list[str]:
    """Every file web/ ships — the deploy check ("did the assets reach the
    image?") is this list being the one the source tree has."""
    return sorted(entry.name for entry in WEB.iterdir()
                  if entry.is_file() and not entry.name.startswith("."))
