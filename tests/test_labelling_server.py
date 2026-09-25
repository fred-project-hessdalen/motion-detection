"""The MCP server an agent labels the corpus through.

These need the analysis and agent groups, and are skipped where either
is not installed.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("mcp")

from hessdalen.dashboard.cluster_labels import read_labels  # noqa: E402
from hessdalen.labelling import server  # noqa: E402
from hessdalen.labelling.service import Settings  # noqa: E402
from hessdalen.labelling.sheets import SHEET, sheet_path  # noqa: E402
from labelling_corpus import BIRDS, corpus  # noqa: E402

SETTINGS = Settings(signature="map", flip_limit=2)


def _call(name: str, **arguments: Any) -> Any:
    """What the tool hands back, as the record or the list it returned."""
    result = asyncio.run(server.server.call_tool(name, arguments))
    held = result.structured_content
    return held["result"] if set(held) == {"result"} else held


def test_the_tools_are_the_ones_the_design_names() -> None:
    names = {tool.name for tool in asyncio.run(server.server.list_tools())}

    assert {"status", "clusters", "sample", "sheet", "verdict", "propagate", "judge", "fetch", "restore"} <= names


def test_the_status_comes_back_as_a_record(tmp_path: Path) -> None:
    server.start(corpus(tmp_path), settings=SETTINGS)

    status = _call("status")

    assert status["tracks"] == 18
    assert status["cached"] == 16


def test_a_verdict_names_the_track_through_the_sheet_it_was_read_on(tmp_path: Path) -> None:
    labelling = server.start(corpus(tmp_path), settings=SETTINGS)
    sheet = sheet_path(labelling.sheets_dir, keys=BIRDS[:2], captions=[], layout=SHEET, kind="sheet")
    sheet.parent.mkdir(parents=True)
    sheet.with_suffix(".json").write_text(json.dumps({"keys": BIRDS[:2], "layout": {}}))

    written = _call(
        "verdict",
        rows=[{"key": BIRDS[1], "name": "bird", "confidence": "sure", "note": "wings"}],
        sheet=str(sheet),
        round=1,
    )

    assert written == {"changed": [BIRDS[1]], "refused": []}
    assert read_labels(labelling.places.labels) == {"bird": [BIRDS[1]]}
    (line,) = _call("ledger", key=BIRDS[1])
    assert line["evidence"] == {"sheet": str(sheet), "row": 1}


def test_a_sheet_of_an_absent_recording_is_refused_with_its_name(tmp_path: Path) -> None:
    server.start(corpus(tmp_path), settings=SETTINGS)

    with pytest.raises(Exception, match="rec9.mkv"):
        _call("sheet", keys=["e/rec9/90"])
