"""The model calls of the labelling, under a backend of the tests' own.

These need the agent group, and are skipped where it is not
installed.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("anthropic")

from pydantic import BaseModel  # noqa: E402

from hessdalen.labelling.readers import (  # noqa: E402
    ModelReaders,
    OllamaBackend,
    RowContext,
    SheetReadings,
)
from hessdalen.labelling.service import NONE_OF_THESE  # noqa: E402

VOCABULARY = {"bird": {"definition": "a bird in flight", "examples": []}, "insect": {"definition": "", "examples": []}}


class ScriptedBackend:
    """A backend that answers what it was given, and keeps what it was
    asked."""

    def __init__(self, answer: BaseModel) -> None:
        self.answer = answer
        self.asked: list[dict[str, Any]] = []

    def ask[Answer: BaseModel](self, shape: type[Answer], *, system: str, images: Sequence[Path], text: str) -> Answer:
        self.asked.append({"shape": shape, "system": system, "images": list(images), "text": text})
        assert isinstance(self.answer, shape)
        return self.answer


def test_a_sheet_is_read_into_one_reading_per_row_in_the_rows_order(tmp_path: Path) -> None:
    backend = ScriptedBackend(
        SheetReadings.model_validate(
            {
                "rows": [
                    {"track": 8, "name": " Bird ", "confidence": "Sure", "note": "wings"},
                    {"track": 7, "name": "insect", "confidence": "maybe", "note": ""},
                ]
            }
        )
    )
    rows = [RowContext(key="a/1", neighbour_names=["bird"]), RowContext(key="a/2", neighbour_names=[])]

    readings = ModelReaders(backend).read(tmp_path / "sheet.png", rows=rows, first=7, vocabulary=VOCABULARY)

    assert [(reading.key, reading.name, reading.confidence) for reading in readings] == [
        ("a/1", "insect", "unsure"),
        ("a/2", "bird", "sure"),
    ]
    assert "- bird: a bird in flight" in backend.asked[0]["system"]
    assert "track 7: nearest seen tracks are named bird" in backend.asked[0]["text"]
    assert "a/1" not in backend.asked[0]["text"]


def test_a_row_the_model_left_out_is_read_as_none_of_these_and_unsure(tmp_path: Path) -> None:
    backend = ScriptedBackend(SheetReadings(rows=[]))

    (reading,) = ModelReaders(backend).read(
        tmp_path / "sheet.png", rows=[RowContext(key="a/1", neighbour_names=[])], first=1, vocabulary=VOCABULARY
    )

    assert reading.name == NONE_OF_THESE
    assert reading.confidence == "unsure"


def test_the_ollama_backend_posts_the_schema_and_the_images_and_parses_the_answer(tmp_path: Path) -> None:
    picture = tmp_path / "sheet.png"
    picture.write_bytes(b"png")
    posted: list[dict[str, Any]] = []

    def post(body: dict[str, Any]) -> dict[str, Any]:
        posted.append(body)
        return {
            "message": {
                "content": json.dumps({"rows": [{"track": 1, "name": "bird", "confidence": "sure", "note": ""}]})
            }
        }

    answered = OllamaBackend(model="qwen", post=post).ask(
        SheetReadings, system="the card", images=[picture], text="read"
    )

    assert answered.rows[0].name == "bird"
    assert posted[0]["model"] == "qwen"
    assert posted[0]["format"] == SheetReadings.model_json_schema()
    assert posted[0]["messages"][0] == {"role": "system", "content": "the card"}
    assert posted[0]["messages"][1]["images"] == ["cG5n"]
