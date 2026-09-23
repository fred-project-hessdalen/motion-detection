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


## Configuration

`config/detector.toml` holds what a run starts from and what the
dashboard's sidebar opens on. Every value in it has a control of its own
on the page, and Save there writes the file, so tuning done on the
dashboard becomes a change to review and commit. The settings with no
control, such as the closing size and the noise floor, keep their values
with the settings class they belong to.

The tests read the same file, so the scene sweep measures whatever is
saved there against every scene it covers.


## Detection

Every pixel keeps a running mean, and a pixel counts towards a detection
once it departs from that mean by a given number of standard deviations.
The noise it is divided by is the larger of two estimates, with
`noise_floor` on `BackgroundSettings` under both at one grey level. One
is what the pixel's own residual has been over the frames before it. The
other is how far the residual spreads over the pixels around it,
measured on 16 by 16 blocks, with the block the pixel sits in and the
blocks touching it left out so an object does not raise the level it is
measured against.

The second estimate is there because of how the recordings are coded.
H.264 repeats a block verbatim while nothing in it changes, so a pixel in
a still part of the scene shows no noise to measure, and the estimate
over time falls to the floor on 98.8 to 99.99 percent of the frame. Every
keyframe then recodes the picture from scratch and moves those pixels
again, which a pixel measured against the frames before it reads as
movement. The pixels around it read it for what it is, because an object
covers a few blocks and a recoded picture covers all of them.

Either estimate on its own leaves a gap. The one over time is what holds
down foliage and a wind-blown horizon, where the same pixels move again
and again. The one over the neighbourhood is what holds down a recoded
picture, where everything moves at once.

Dividing by noise does not finish the keyframe, because a keyframe hands
a residual to pixels that had none rather than making the noise louder.
Between keyframes more than half the picture carries a residual of
exactly zero, which no threshold reaches, so the count of pixels over the
threshold jumps several times over however the deviation is scaled. Each
frame's deviation is therefore divided down a step at a time, by at most
three, until that count is back near the level the recording keeps
returning to. `foreground_budget` on `DetectionSettings` sets how far
above that level a frame may sit before its threshold moves, and on the
example recordings one frame in seven moves.

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
detection time from 112.7 to 28.0 seconds, and all ten recordings report
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

Under the settings, Save writes everything the sidebar sets to
`config/detector.toml`, Reset puts the detection, tracking and
background sliders back to the values that file holds, Export writes
those same settings to a file of your own, and Import reads one back.
Frame height, panels and device are saved but left out of Reset, Export
and Import. An exported file may name as few settings as it likes, and
the sliders it does not name stay where they stand.

The Live switch builds a segment of the selected recording into a short
clip and loops it in the page, and moving any setting builds the segment
again under the new value. The clip plays at the rate it was written at,
which a frame pushed at a time cannot do, because nothing on the far
side holds those frames to a rate. A clip already built for the settings
in the sidebar plays at once, so two values can be compared without
waiting for either again, and the clip built last keeps playing while
the next one is built.

Only the segment is detected, so the background model opens on its first
frame and takes its scene from the frames after it. At a frame height of
1080 a clip under `data/examples/clips` builds in 2 to 4 seconds, and an
eight-second segment 41 seconds into a recording takes 5 to 10 seconds.
Most of the latter is spent reaching the segment, because neither
decoder lands on the frame a linear decode calls by that number on these
files, so everything before it has to be decoded. Built clips are kept
under `data/out/dashboard/live`, which can be deleted at any time.

A stored run and a live clip are both written through `ffmpeg` with
`libx264`, which has to be on PATH.

### Track map

The Track map page shows every track of the corpus under `data/corpus`
as a point. Tracks with similar descriptors sit close together, and
Colour in the sidebar gives what the points are coloured by: the cluster
they fell into, the name that cluster has been given, the side they were
clustered on, the folder of the archive their recording was filed in, or
the camera that recorded them. A cluster label and a folder are
different things. The first is a name someone gave a cluster on this
page, and the second is where the recording sits in the archive, which
names the whole recording rather than the track.

Clicking a point draws that track at once from its stored path: where it
was in the frame, its shape fitted to a box of its own, and its
brightness and size frame by frame. Drawing it on its recording, from a
second before it starts to a second after it ends, runs in the
background, and four videos play once they are built. Panel chooses
between the recording and the deviation the detector measures it
against, and the tabs choose between the two views of the chosen panel.
In the frame holds the whole picture with the track's path and a box
drawn on it, and Close up holds a crop three box half-widths out from
the detection on every side, which follows it from frame to frame and
carries nothing drawn over it. The deviation is measured over the
stretch alone, so the background model opens on the stretch's first
frame and the second before the track starts is what it has to settle
in. The crop eases a quarter of the way towards the detection each
frame, so the hop of the matched pixel inside its blob hardly moves it,
and it jumps to the detection outright once that sits more than a box
half-width from the middle. The crop is black where it reaches past the
edge of the picture. All four come out of the one pass over the recording, because
reaching the track is most of what a build costs. A track deep in a
recording is reached by seeking to the keyframe before it, which needs
the recording's frame stamps read first. A recording whose stamps do
not say which frame is which has every frame ahead of the track passed
instead, which takes a minute on a 20-minute recording. A gallery below the map
draws 30 tracks at random from the selected track's cluster, or from any
cluster chosen in the sidebar, from their stored paths alone. Under that
gallery, Cluster label gives the cluster a name of your own and Save
keeps it against every one of the cluster's tracks in
`data/out/analysis/cluster-labels.json`, which holds each name with the
tracks under it and is what a training set is built from. A name is kept
in small letters with single spaces between its words and every word in
the singular, so `Street Lights` and `streetLight` come to the one name
`street light`, and a file holding two spellings of a name has the
tracks of both under it once it is read. A cluster named again moves to
the new name, and an empty name takes its tracks out of the one they
were under. A second
gallery draws the 30 tracks nearest the selected one on the map, from
any cluster and within the neighbour radius set in the sidebar, and
names the cluster of each with the name it is under. Cluster labels in
the sidebar holds the page to the tracks under the names picked there,
with every track of an unnamed cluster under `unlabelled`, and Folders
holds it to the tracks of recordings filed under the folders picked
there. Tracks whose movement from step to step is
uneven, which most clutter is, can be left out of the page with the
Rough tracks switch in the sidebar, and the Video cached switch there
draws the tracks whose recording is not on disk faintly. Search in the
sidebar selects a track by its number, by its recording or by both, in
the form the heading over a selected track gives them, such as `Track
7484 in Cam1_2025-06-03__12-40-00_noInsect`. The map marks
the selected track with a star and rings the tracks of each gallery in
the colour that gallery frames its panels in. The name a cluster has
been given stands over the middle of its points, and the Cluster labels
entry in the legend takes every name off the map. The map is drawn by
plotly.js, which the page loads from the jsDelivr CDN, so the page
needs internet access to show it.

The page runs no detection. A track whose video the sift did not keep
has its video fetched from the archive into `data/out/dashboard/videos`
before it is drawn. A one-minute cut is fetched on the click, and a
whole 20-minute recording waits for a press of Fetch video. No fetch is
made that would leave less than 12 GB free on the disk, which the
archive sift needs to keep running.

The map and the paths it draws from are written by a separate step,
which needs the analysis group:

```bash
uv run --group analysis python scripts/dev/map_tracks.py
```

The corpus keeps growing while the sift runs, and the map holds the
tracks that existed when this step ran, so run it again to take in new
ones. Built track clips are kept under `data/out/dashboard/tracks`.


## Development

```bash
# Run tests
uv run pytest

# Linting and type checking
uv run ruff check .
uv run mypy src scripts tests

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
