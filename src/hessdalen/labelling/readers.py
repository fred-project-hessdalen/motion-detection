"""The model calls of the labelling: reading a sheet, judging a contested
track, and proposing a name.

Each call is one request with an image and a structured output, and
no call sees another's context. The ledger is the memory. What the
model is told is the vocabulary with its definitions, what a row of a
sheet shows, and per row the little the corpus knows that is not in
the picture.

The request goes to one of three backends, the Messages API, the
Responses API of OpenAI, or a model ollama serves on this machine.
The calls stand behind one interface so the harness can be run with a
reader of the tests' own.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from anthropic import Anthropic
from anthropic.types import ImageBlockParam, TextBlockParam
from openai import OpenAI
from openai.types.responses import EasyInputMessageParam, ResponseInputMessageContentListParam
from pydantic import BaseModel

from hessdalen.labelling.service import CONFIDENCES, NONE_OF_THESE, Reading

MODEL = "claude-fable-5-1"
MAX_TOKENS = 4096

SHEET_TEXT = """\
The picture is a contact sheet of tracks the movement detector found in
sky recordings from fixed cameras. Each row is one track. A row holds a
caption naming the track, then the track's path drawn on its own from
dark blue at its start to yellow at its end, then six crops of the
recording taken at even steps across the track. Each crop is cut one
box half-width around the detection, enlarged so its pixels stay
pixels, and stretched over the grey levels it holds, so a faint speck
comes out with a shape and the sky behind it comes out grainy.

Name what moved in each row, from the vocabulary, or answer
"none of these" when nothing in the vocabulary fits. Say how sure you
are: sure, likely or unsure, with a note of one short sentence. Read
the rows in order and answer every row once, by the track number its
caption starts with."""

VIEW_TEXT = """\
The picture is an isolation view of one track the movement detector
found in sky recordings from fixed cameras. The first row holds the
track's path drawn on its own from dark blue at its start to yellow at
its end, then twelve stretched close-up crops across the track. The
second row holds the whole frame at the same steps with the detection
boxed. The third row draws the blob's brightness and size over the
track's frames.

The track's name has changed more than once as tracks around it were
named. Settle what moved, from the vocabulary, or answer "review" when
you cannot, and say how sure you are: sure, likely or unsure."""

PROPOSAL_TEXT = """\
These sheets show tracks that were read as none of the names in the
vocabulary. Propose one name for the kind of thing they hold, in one
or two lowercase words, with a one-line definition, and say which rows
show it best by their keys."""

REVIEW = "review"


class RowReading(BaseModel):
    track: int
    name: str
    confidence: str
    note: str


class SheetReadings(BaseModel):
    rows: list[RowReading]


class Judgement(BaseModel):
    name: str
    confidence: str
    note: str


class Proposal(BaseModel):
    name: str
    definition: str
    examples: list[str]


@dataclass(frozen=True, slots=True)
class RowContext:
    """What a row's caption does not say: the names of the track's nearest
    seen tracks.

    The key is never shown to the model. A key carries the recording's
    name and the folder it was filed under, which name what the
    recording was kept for, and a reader told "bird" reads a bird.
    Rows are named to the model by their number alone.
    """

    key: str
    neighbour_names: Sequence[str]


class Readers(Protocol):
    def read(
        self, sheet: Path, *, rows: Sequence[RowContext], first: int, vocabulary: dict[str, Any]
    ) -> list[Reading]: ...

    def judge(self, view: Path, *, key: str, history: Sequence[str], vocabulary: dict[str, Any]) -> Reading: ...

    def propose(self, sheets: Sequence[Path], *, vocabulary: dict[str, Any]) -> Proposal: ...


class Backend(Protocol):
    """One request to a model: a system text, images, a user text, and the
    shape the answer has to come back in."""

    def ask[Answer: BaseModel](
        self, shape: type[Answer], *, system: str, images: Sequence[Path], text: str
    ) -> Answer: ...


class ModelReaders:
    """The three calls made to a model, through whichever backend serves
    it."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def read(self, sheet: Path, *, rows: Sequence[RowContext], first: int, vocabulary: dict[str, Any]) -> list[Reading]:
        """One reading per row, the rows numbered from first as their
        captions are."""
        context = "\n".join(
            f"- track {number}: nearest seen tracks are named {', '.join(row.neighbour_names) or 'nothing yet'}"
            for number, row in enumerate(rows, start=first)
        )
        text = f"{SHEET_TEXT}\n\nRows, in order:\n{context}"
        answered = self._ask(SheetReadings, images=[sheet], text=text, vocabulary=vocabulary)
        by_number = {row.track: row for row in answered.rows}
        readings = []
        for number, row in enumerate(rows, start=first):
            held = by_number.get(number)
            if held is None:
                readings.append(Reading(key=row.key, name=NONE_OF_THESE, confidence="unsure", note="no reading given"))
            else:
                readings.append(
                    Reading(key=row.key, name=_name(held.name), confidence=_confidence(held.confidence), note=held.note)
                )
        return readings

    def judge(self, view: Path, *, key: str, history: Sequence[str], vocabulary: dict[str, Any]) -> Reading:
        text = f"{VIEW_TEXT}\n\nWhat the ledger says of the track so far:\n" + "\n".join(
            f"- {line}" for line in history
        )
        answered = self._ask(Judgement, images=[view], text=text, vocabulary=vocabulary)
        name = REVIEW if _name(answered.name) == REVIEW else _name(answered.name)
        return Reading(key=key, name=name, confidence=_confidence(answered.confidence), note=answered.note)

    def propose(self, sheets: Sequence[Path], *, vocabulary: dict[str, Any]) -> Proposal:
        return self._ask(Proposal, images=list(sheets), text=PROPOSAL_TEXT, vocabulary=vocabulary)

    def _ask[Answer: BaseModel](
        self, shape: type[Answer], *, images: Sequence[Path], text: str, vocabulary: dict[str, Any]
    ) -> Answer:
        return self.backend.ask(shape, system=_vocabulary_card(vocabulary), images=images, text=text)


class AnthropicBackend:
    """A request to the Messages API, the answer parsed into the shape."""

    def __init__(self, client: Anthropic, *, model: str) -> None:
        self.client = client
        self.model = model

    def ask[Answer: BaseModel](self, shape: type[Answer], *, system: str, images: Sequence[Path], text: str) -> Answer:
        content: list[ImageBlockParam | TextBlockParam] = [_image_block(image) for image in images]
        content.append({"type": "text", "text": text})
        answered = self.client.messages.parse(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=shape,
        )
        if answered.parsed_output is None:
            raise ValueError("The model answered in no readable shape.")
        return answered.parsed_output


class OpenAIBackend:
    """A request to the Responses API, the answer parsed into the shape."""

    def __init__(self, client: OpenAI, *, model: str) -> None:
        self.client = client
        self.model = model

    def ask[Answer: BaseModel](self, shape: type[Answer], *, system: str, images: Sequence[Path], text: str) -> Answer:
        content: ResponseInputMessageContentListParam = [
            {"type": "input_image", "image_url": f"data:image/png;base64,{_encoded(image)}", "detail": "high"}
            for image in images
        ]
        content.append({"type": "input_text", "text": text})
        message: EasyInputMessageParam = {"role": "user", "content": content}
        answered = self.client.responses.parse(
            model=self.model,
            instructions=system,
            input=[message],
            text_format=shape,
        )
        if answered.output_parsed is None:
            raise ValueError("The model answered in no readable shape.")
        return answered.output_parsed


OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_SECONDS = 1800.0
"""How long one answer may take. A model that spills out of the card
into main memory takes minutes for a sheet."""


class OllamaBackend:
    """A request to a model ollama serves, with the answer held to the
    shape's JSON schema by ollama's own format argument.

    The request is posted through the callable given, which is what a
    test replaces.
    """

    def __init__(self, *, model: str, post: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.model = model
        self.post = post

    def ask[Answer: BaseModel](self, shape: type[Answer], *, system: str, images: Sequence[Path], text: str) -> Answer:
        body = {
            "model": self.model,
            "stream": False,
            "format": shape.model_json_schema(),
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text, "images": [_encoded(image) for image in images]},
            ],
        }
        answered = self.post(body)
        return shape.model_validate_json(str(answered["message"]["content"]))


def ollama_post(url: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """What posts a chat request to the ollama at this address and hands
    back its answer."""

    def post(body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{url.rstrip('/')}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            return dict(json.loads(response.read()))

    return post


def _vocabulary_card(vocabulary: dict[str, Any]) -> str:
    lines = [
        f"- {name}: {held.get('definition') or 'no definition written yet'}"
        for name, held in sorted(vocabulary.items())
    ]
    return (
        "You name what moved in pictures from sky cameras. The vocabulary, one name per line with its definition:\n"
        + "\n".join(lines)
        + f'\n\nAnswer with a name exactly as listed, or "{NONE_OF_THESE}".'
    )


def _image_block(path: Path) -> ImageBlockParam:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _encoded(path)}}


def _encoded(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def _name(name: str) -> str:
    return name.strip().lower()


def _confidence(confidence: str) -> str:
    held = confidence.strip().lower()
    return held if held in CONFIDENCES else "unsure"
