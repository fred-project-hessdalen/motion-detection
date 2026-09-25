# Labelling the corpus with an agent

Design notes for naming every track of the corpus with a model in the
loop. The track map groups the corpus into clusters, and the Track map
page lets a person name a cluster, tag a track and mark a track as
validated. Naming tens of thousands of tracks by hand does not scale,
so this design has a model read pictures of tracks and a program
spread its verdicts to the tracks around them.

Three parts: the labelling protocol, a Model Context Protocol (MCP)
server that exposes the corpus and the label files to an agent, and
the harness loop that runs the protocol with the model inside it.


## Corpus state

Measured on the map written on 2026-09-24:

- 58855 tracks, of which 51245 carry no name.
- 8036 tracks have their recording on disk, from 108 of the 1138
  recordings the corpus was sifted from.
- 188 clusters, and 7390 tracks the clustering left unassigned, of
  which 513 have a recording on disk. 176 clusters hold at least one
  track with a recording on disk.
- 101 clusters hold tracks under more than one name. Names are kept
  per track, and the map was rebuilt after the first naming pass, so
  the old cluster names lie scattered over the new clusters.
- 15 names are in use. 25 tracks are validated by a person.

The clusters are dirty. A cluster holds tracks of more than one kind,
so a name given to a whole cluster is wrong for part of it. The unit
of naming is therefore the track, and a name spreads from track to
track through neighbours in a signature space. The cluster is only a
way to spread the first pictures over the whole space.


## Sheets

A model reads images. The close-up video the page builds is too wide
and too faint to judge an object by, and a sheet of tight crops is
readable. A sheet holds one row per track. A row holds a caption with
the track id, the frame count, the folder label and the recording,
then the drawn path coloured from dark blue at its start to yellow at
its end, then six crops of the recording taken at even steps across
the track. Each crop is cut one box half-width around the detection,
enlarged by nearest neighbour so its pixels stay pixels, and stretched
over the grey levels it holds. A sheet of six rows builds in about
twelve seconds when the recordings are on disk.

An isolation view is a sheet of one track with twelve crops, the whole
frame beside the close-up at each step, and the track's light curve
and blob size drawn under it.


## Protocol

1. Setup. Snapshot the label files. Open a ledger with one line per
   decision. Fix the vocabulary to the names in use.
2. Seed. Take the clusters in order of how many unnamed tracks with a
   recording on disk each holds. Per cluster, read one sheet of six
   tracks spread over the cluster's core, its rim and distinct
   recordings. Every row gets its own verdict. Nothing is named beyond
   the rows seen.
3. Propagate. An unseen track takes a name only when its five nearest
   seen tracks in signature space agree four to one or better and the
   nearest of them lies within a distance cap. Every other unseen track
   stays unnamed.
4. Verify. From the tracks the propagation just named, sheet six with
   a recording on disk that sit nearest the cap or had the weakest
   vote. The vote each received is its prediction. Read the sheet and
   compare. The verdicts join the seen set. When fewer than five of
   six agree with their vote, the region is propagated again from the
   larger seen set.
5. Aim. The next sheet goes to tracks with a recording on disk whose
   nearest seen tracks disagree, or that have no seen track within the
   cap. Steps 3 to 5 repeat until no such track is left.
6. Fetch. Tracks whose recording is absent take the same propagation
   rule. The ones beyond the cap make the fetch list, ordered by which
   recording covers most of them. A fetched recording's tracks go
   through steps 3 to 5.
7. Isolate. A track whose name changed twice on different grounds is
   contested. Propagation stops on it. It gets an isolation view, its
   ledger history and the names of its five nearest seen tracks, and
   a final verdict or a "review" tag for the person. Several contested
   tracks in one cluster are sheeted together first, because a cluster
   boundary that runs through a kind leaves its contested tracks in
   one place.
8. Ground truth. Validated tracks are never written. A verdict that
   contradicts a validated track on the same sheet is logged and the
   sheet's verdicts are held back for the report.
9. Record. Every write carries its basis, one of seen, propagated or
   judged, and its evidence, the sheet path and row or the voting
   keys, in the ledger.

The signature space of step 3 is a setting. The first space is the 16
descriptors the clustering reads, as normal scores among the tracks of
each camera. A track carried into a path image scored better at
telling kinds apart in the comparison of 2026-09-24, so a second space
has to plug in behind the same setting.


## MCP server

One Python server over stdio in the package `hessdalen.labelling`,
started with

    uv run --group agent --group core --group dashboard \
        --group analysis python -m hessdalen.labelling.server

and registered in `.mcp.json`. It imports the dashboard's own modules
for reading the map, cutting crops, writing labels and fetching
recordings, so a verdict recorded through the server lands in the
files the Track map page reads. The server holds no opinions. It
reads, builds pictures, applies the vote rule and writes what it is
told to, with every write going into the ledger.

State loaded at start:

- The map table and the paths table, keyed as the page keys them.
- The signature space named by the `signature` setting, with a
  nearest-neighbour index over every track.
- The ledger at `data/out/analysis/labelling/ledger.jsonl`. The seen
  set, each track's flip count and the contested set are derived from
  it at start and kept in memory.
- The label files, reread by modification time on every call.

Read tools:

| Tool | Input | Output |
|---|---|---|
| `status` | vote rule | counts of tracks per name, seen, cached, contested, review-tagged, and how many cached tracks the vote leaves uncertain |
| `clusters` | count | per cluster the size, cached count, unnamed cached count, names held with counts, seen count and recording count, the most unnamed cached tracks first |
| `track` | keys | descriptor row, cluster, name, tags, validated flag, seen verdict, cached flag and flip count |
| `neighbours` | key, count, seen_only | the nearest tracks in signature space with distance, name and seen flag |
| `sample` | cluster, count | keys spread over core, rim and distinct recordings, cached, unseen and unvalidated |
| `uncertain` | count, vote rule | cached unseen tracks whose neighbour vote disagrees or that have no seen track within the cap, weakest first |
| `verify_candidates` | round, count | cached unseen tracks a propagation of the round named, the furthest from their voters and the weakest votes |
| `calibration` | count | cached tracks whose name is settled, validated ones first |
| `sheet` | keys | the sheet as image content, with a line naming its path and rows, at most eight rows |
| `track_sheet` | key | the isolation view as image content, with a line naming its path |
| `propagation_preview` | vote rule | what a propagation would name, counts per name and how many tracks it would leave uncertain, with nothing written |
| `ledger` | key or last n | ledger lines |
| `fetch_list` | count, vote rule | recordings not on disk ordered by how many uncertain tracks each covers, with sizes |
| `vocabulary` | none | every name a verdict may use, with its definition and examples |

The vote rule is three numbers, the votes taken, the agreement needed
and the cap, with defaults of 5, 4 and 1.0.

Write tools:

| Tool | Input | Effect |
|---|---|---|
| `predict` | rows of key and name, sheet path, round | ledger lines with basis "predicted", written before the sheet is read |
| `verdict` | rows of key, name, confidence, note, sheet path, round | a label written per key and a ledger line with basis "seen", the sheet path and the row as evidence. A validated key or a name outside the vocabulary is refused. An unsure confidence or "none of these" records the track as seen without a name |
| `judge` | one row, view path, round | the same as a verdict, from an isolation view, with basis "judged" |
| `propagate` | round, vote rule | the vote applied to every unseen, unvalidated, uncontested track, a ledger line per changed track with basis "propagated" and the voting keys, and the newly contested keys returned |
| `tag` | key, tags, round | the track's tags written, used for "review" |
| `define_name` | name, definition, example keys, round | the name added to the vocabulary after the near-name check, recorded in the ledger |
| `fetch` | recording | the recording fetched into the page's own folder within its disk floor, about a minute per recording, and the call returns when it has landed |
| `snapshot`, `restore` | name | copies of the label files under `labelling/snapshots` |

Every write takes the round it belongs to, and `status` says the
highest round written so far. The vocabulary is fixed at the first
call to the names then in the label file and grows only through
`define_name`.

A sheet is built once per set of keys and settings and stored under
`labelling/sheets/<digest>.png`, so the evidence path in the ledger
stays valid and a repeated call costs nothing. A track whose recording
is absent is refused with the recording named.

The page and the server both write a label file by reading it,
changing it and writing it whole, so a write by one between the
other's read and write is lost. The write functions in
`cluster_labels.py` therefore take a lock file beside the label file,
reread the file under the lock, apply the change and write, and the
page's modification-time cache picks the server's writes up on its
next run.


## Harness loop

The loop is a Python program, `hessdalen.labelling.harness`. Choosing
what to look at, propagating, verifying and stopping are rules, so
they live in code where they run the same way every round and leave a
ledger that can be replayed. The model is called for what needs eyes.
The harness calls the same service class the MCP tools wrap, in
process, so an interactive session with the MCP registered can inspect
or continue any state the harness left.

Model calls:

| Call | Input | Output |
|---|---|---|
| Reader | one sheet image, the vocabulary card, and per row the names of its five nearest seen tracks, with the caption carrying the track, its clip, its frame count and its camera | per row a name from the vocabulary or "none of these", a confidence of sure, likely or unsure, and a note |
| Judge | the isolation view of one contested track and its ledger history | a final name or "review" |
| Proposer | the last sheets holding rows the reader answered "none of these" | a proposed name with a definition and example keys, written to `labelling/proposals.jsonl`, with the example rows tagged "review" |

Each call is a fresh Messages API request with a structured output
schema, made through `hessdalen.labelling.readers`. No call sees a
previous call's context. The ledger is the memory. The calls stand
behind one interface, so the loop runs under a reader of the tests'
own.

The loop runs with

    uv run --group agent --group core --group dashboard \
        --group analysis python -m hessdalen.labelling.harness \
        --sheets 400 --fetches 0

with the API key in the environment. The sheet budget, the fetch
budget, the vote rule, the thresholds, the model and the signature
space are command line arguments.

Rounds:

0. Calibration. The reader reads two sheets of tracks whose name is
   settled, the validated ones first, blind. A reading that agrees is
   written as seen, so the settled tracks vote. The agreement rate and
   every disagreement go to the log. Under a threshold the harness
   stops, because propagating from a reader that misreads the ground
   truth spreads the misreading.
1. Seed, as protocol step 2. Rows answered "unsure" or "none of these"
   are recorded as seen without a name and do not vote.
2. Propagate, as protocol step 3. The vote is the prediction, so no
   model call predicts.
3. Verify, as protocol step 4. The agreement rate per round goes to
   the log. Once a sheet's worth of verified readings exist, the cap
   moves to the largest distance up to which the readings so far
   agree with the vote at the threshold.
4. Aim, as protocol step 5. Rounds 2 to 4 repeat until no cached
   track is uncertain or the sheet budget is spent.
5. Fetch, as protocol step 6, within a fetch budget.
6. Isolate, as protocol step 7, through the judge.
7. Report. The calibration rate, the agreement rate per verification,
   sheets read, recordings fetched, tracks judged, names proposed, and
   the status of the corpus.

Settings: rows per sheet 6, votes 5, agreement 4 of 5, the cap 1.0
until verification moves it, flip limit 2, and a sheet budget and a
fetch budget given on the command line.

Safety:

- The harness resumes from the ledger, so a stopped run continues
  where it was.
- A snapshot of the label files is taken at the start of every run,
  under the round the run starts in.
- New names never enter the vocabulary from the loop. The proposer's
  output waits in the proposals file, and `define_name` is the person's
  to call.
- The label file lock is held only for the moment of a write, so the
  page stays usable while the harness runs.

Cost, from the earlier pass: a sheet builds in about twelve seconds
and a reader call carries one image of about 1200 by 860 pixels. Two
hundred seed sheets and a few hundred verification sheets come to a
few hours of building and a few hundred model calls.
