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
a given number of its own standard deviations. A dark sky, a moonlit
slope and a snowy afternoon all read on that one scale, so a single set
of settings covers them.

`--detection-sigma` sets how far a blob's brightest pixel has to depart
from the background to be reported, and it is the first parameter to
reach for. Values between 12 and 20 all work on the example recordings.

Distances are given as a ratio of the frame's larger dimension rather
than in pixels, so `--max-movement-ratio` and its siblings hold their
meaning at any `--target-height`.


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
