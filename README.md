# Movement Detection

Movement detection system prototype for Project Hessdalen.

## Examples

![Camera 1 Movement Detection](docs/birds.gif)

![Camera 2 Movement Detection](docs/meteor.gif)

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync --group core --group dev --group test
```


## Usage

```bash
uv run python scripts/dev/detect_movement.py path/to/video.mp4 --output output_directory/
```

Run with `--help` for all available parameters.


## Detection

Every pixel keeps a running mean and a running estimate of its own noise,
and a pixel counts towards a detection once it departs from that mean by
a given number of standard deviations. The estimate never falls below
`noise_floor` on `BackgroundSettings`, which is one grey level.

That floor is what most of the picture runs on. H.264 repeats a block
verbatim while nothing in it changes, so a pixel in a still part of the
scene has no noise left to measure, and over the example recordings the
estimate stays at the floor on 98.8 to 99.99 percent of the frame. With
the floor at one grey level the threshold there is `--detection-sigma`
grey levels. A dark sky, a moonlit slope and a snowy afternoon all read
on that one scale, so a single set of settings covers them.

The measured estimate takes over where the scene keeps moving. Over
foliage and a wind-blown horizon the noise climbs above the floor and
those pixels need a much larger departure before they report.

`--detection-sigma` sets how far a blob's brightest pixel has to depart
from the background to be reported, and it is the first parameter to
reach for. Values between 12 and 20 all work on the example recordings.

Distances are given as a ratio of the frame's larger dimension rather
than in pixels, so `--max-movement-ratio` and its siblings hold their
meaning at any `--target-height`.


## Graphics card

The per-pixel stage of the detector runs on an NVIDIA card when CuPy is
installed and a card answers, and on the host otherwise.

```bash
uv sync --group core --group gpu
```

Over the example recordings at a frame height of 1080 the card cuts
detection time from 74.0 to 21.1 seconds, and all ten recordings report
the same frame numbers, track ids and centroids either way.

Set `device` on `MovementSettings` to `"cpu"` to stay on the host, or to
`"cuda"` to fail when no card answers.


## Dashboard

A debugging dashboard lists the recordings under `data/examples`, runs the
detector over one or all of them, and plays the result with the tracks
drawn on it. The sidebar chooses the panels a run draws: the recording,
the deviation it was measured against, or both stacked. Each panel costs
its own pass over the recording, so one panel runs in about half the time
of both.

```bash
uv run --group dashboard --group gpu streamlit run src/hessdalen/dashboard/app.py
```

Pick a recording by clicking its row, set the parameters in the sidebar
and press Run. Each run is stored under `data/out/dashboard` under the
settings it used, so a recording that has been run with the settings
currently in the sidebar plays back without running again. Set
`HESSDALEN_EXAMPLES_DIR` to list recordings from somewhere other than
`data/examples`.

The Live switch plays a segment of the selected recording over and over
while the detector runs on it, and moving any setting starts the segment
again under the new value. At a frame height of 1080 a segment holds the
25 frames a second the cameras record at. The frames ahead of the
segment are measured as well and not drawn, so the tracks and the
background model stand where a run over the whole recording would leave
them, and a segment that starts a thousand frames in takes a few seconds
to reach. The clips start at the event, so they reach it at once.

Rendering the annotated video needs `ffmpeg` with `libx264` on PATH.


## Development

```bash
# Run tests
uv run pytest

# Linting and type checking
uv run ruff check .
uv run mypy src/

# Format code
uv run ruff format .
uv run docformatter . --recursive
```

## Data Management

This project uses DVC (Data Version Control) for managing example data and test datasets.

```bash
# Pull data from remote storage
dvc pull
```
