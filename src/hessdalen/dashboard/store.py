"""The name a built video is kept under.

The dashboard is used while the detector is being changed, so a stored
build has to stop matching once the code that decides what it holds has
been edited. A build names the modules that decide that, which keeps an
edit to one kind of build from throwing away the other kind.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

DETECTOR_PACKAGES = ("domain", "io", "processing")


def digest(payload: dict[str, Any], *, modules: tuple[str, ...]) -> str:
    """A name for a build, which changes when its settings or its code
    change."""
    described = json.dumps({"payload": payload, "code": _code_digest(modules)}, sort_keys=True)
    return hashlib.sha1(described.encode()).hexdigest()[:10]


@cache
def _code_digest(modules: tuple[str, ...]) -> str:
    package = Path(__file__).resolve().parents[1]
    sources = [source for name in DETECTOR_PACKAGES for source in sorted((package / name).rglob("*.py"))]

    hashed = hashlib.sha1()
    for source in [*sources, *(package / "dashboard" / name for name in modules)]:
        hashed.update(source.read_bytes())
    return hashed.hexdigest()[:10]
