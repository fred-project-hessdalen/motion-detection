# Trajectory classification and anomaly flagging

Design notes for the stage after tracking. The detector finds moving
objects and follows them across frames. The open question is how to sort
the resulting trajectories into recurring classes and flag the ones no
class explains.

The per-track record, the writer that stores it and the fixed-length
descriptor are built. The recurrence prior, the clustering and both
anomaly rankings are still a design.


## Corpus

Hundreds of hours of recordings sit in cloud storage and have not been
pulled. None of it is labelled. The five labelled events in
`data/examples/metadata.csv` are the whole ground truth.

The corpus is built by running the analytical detector over the archive
and storing every track it produces. Measured on the card at a frame
height of 1080 from 4K sources, a single decode-and-detect process runs
6801 frames in 21.71 seconds, which is about 12 times realtime for a
25 fps recording. Three hundred hours is therefore roughly one machine
day single-process, and decode is the wall, so it parallelises across
cores. Pulling and extracting is cheap relative to everything
downstream.


## Per-track record

Decide the fields before extraction. Re-running over hundreds of hours
to add one is the expensive mistake.

`hessdalen.io.tracks` writes one row per frame of every track, with the
centroid, the peak deviation, the blob's pixel count, its summed grey
levels and the major and minor axis of the ellipse with its second
moments. `scripts/dev/extract_tracks.py` writes one file per recording.

Position alone is not enough for either route below. The photometric
channel is likely the stronger discriminator: a meteor shows a single
rise and decay, a bird carries a wingbeat flicker, an aircraft strobes
periodically, a satellite is near-flat with slow variation.


## Corpus versioning

Every trajectory is a product of the analytical detector and the
settings it ran under, which live in `config/detector.toml`. Any model
trained on the corpus learns the detector as much as it learns the sky.

A track file carries the frame height and the whole settings tree in its
metadata, including the values no dashboard control offers, so two runs
that differ anywhere can be told apart and a model can be trained on
either population alone.


## Clutter and recurrence

The extracted population is dominated by detector clutter. Candidate
tracks opened per example recording run from 5 to 977 against 1 to 16
confirmed tracks. An anomaly score defined against that population ranks
unusual clutter highly, and more data does not change what "normal"
means.

Scale hands over one handle that small data cannot. Clutter is
site-bound and repeats: the same branch, the same reflection, the same
insect corridor, at the same image coordinates on the same camera, night
after night. A track class that recurs at a fixed location on a fixed
camera is mundane by construction, and mining that needs no labels.

Build the per-camera recurrence prior before any model. Both routes
below need it.


## Classical trajectory comparison

Two families, both standard in the trajectory data mining literature.

### Fixed-length descriptors

Map each trajectory to a vector of physically meaningful scalars and
cluster in that space. Comparability is solved by construction and every
axis is readable, which matters when the output has to be defended to a
domain expert.

`hessdalen.analysis.descriptors` builds 27 of them per track, and
`scripts/dev/describe_tracks.py` writes one row per track for a
directory of track files. Grouped by what they separate:

- Kinematics: frames held, frames missed, path length, net
  displacement, straightness, and the mean, maximum and spread of
  speed. All as ratios of the frame's larger side, as the tracker's
  distances already are.
- Shape: mean and largest heading change, the residual of a
  straight-line fit, and the residuals of a steady-heading and a
  steady-acceleration fit.
- Frame geometry: how far the first and last point sit from the nearest
  frame edge, and the elevation band. A track that crosses the frame
  and a track that appears and vanishes inside it are different
  populations.
- Photometry: peak deviation, area and its swing, elongation, summed
  brightness and its swing, and the strength and rate of any rhythm in
  the brightness.

Standardise per camera, or the first principal axis is the site.

Two of these needed care, and both were found by running the
descriptors over the example recordings. A meteor's single rise and
decay is one narrow lobe at the low end of the spectrum and holds over
half the power in its strongest bin, so a rhythm is only counted when
it completes at least three cycles within the track. A heading measured
across half a pixel says nothing, so heading changes are weighed by how
far the track moved to make them.

Still missing from the list: a correlated random walk in the model
library, and the rise and fall asymmetry of the light curve.

### Elastic distance measures

Keep the raw sequences, define a pairwise distance that tolerates
different lengths, and cluster with a method that needs only a distance
matrix: agglomerative, k-medoids, spectral, DBSCAN, OPTICS, HDBSCAN. The
density-based ones leave points unassigned, and that unassigned set is
the candidate anomaly set.

- Dynamic Time Warping (DTW) is the default. Warping buys invariance to
  speed, which may be unwanted here, since angular speed separates a
  satellite from an insect.
- Discrete Frechet and Hausdorff distance compare geometry with timing
  discarded.
- Longest Common Subsequence (LCSS) and Edit Distance on Real sequence
  (EDR) tolerate dropped and spurious points, which matters given
  `max_missed_frames` on `TrackingSettings`.
- Symmetrised Segment-Path Distance (SSPD), and TRACLUS where
  sub-trajectory structure carries more than whole-track shape.

Normalise before any of these: translate to the start point, rotate to
the principal axis, scale by frame dimension. Without that the
clustering reports where in the frame things happen.

Library support: `tslearn`, `dtaidistance`, `similaritymeasures` for the
distances, `hdbscan` and scikit-learn for the clustering.


## Learned sequence encoder

The intended route is self-supervised: extract trajectories with the
analytical detector, train a sequence model on them without labels, and
take the encoder as a fixed-length representation that makes
variable-length trajectories comparable. With hundreds of hours
available, the corpus is large enough to train on.

### Pretext task choice

The pretext task determines what the encoder knows, and next-step
prediction on positions has two problems.

Constant-velocity extrapolation achieves most of the achievable loss on
short tracks, so the model converges toward a learned Kalman filter and
its residuals approach the constant-velocity residual already listed
among the classical descriptors above. The training budget buys
something the descriptor route gets for free.

An autoregressive objective also optimises the hidden state to be
sufficient for the next step, which makes the representation local in
time. The final hidden state is a weak whole-trajectory summary.

Three choices address both:

- Masked infilling or sequence-to-sequence reconstruction as the
  objective. Both force a whole-sequence summary through the bottleneck,
  which is the quantity to be clustered.
- A probabilistic head predicting a distribution per step. Calibrated
  surprise is a better anomaly score than squared error, and it is
  localised in time, so it points at the frame where the object did
  something the model could not predict. That signal may be worth more
  than the encoder.
- Every channel from the per-track record as model input. The
  degenerate solution lives in the kinematics. A light curve does not
  extrapolate linearly and forces the model to carry something about the
  object.

Trajectories here are short. A meteor covers 1 to 3 seconds, so 25 to 75
points. Sequence models earn their keep on longer sequences with
compositional structure, which is an argument for keeping the classical
descriptor as the baseline the encoder has to beat.


## Evaluation with few labels

Volume does not create labels, so the ranking still needs validation.

The predictor itself validates on held-out next-step or infill
likelihood, which says nothing about whether the anomaly ranking is
meaningful.

For the ranking: score every track in a recording, then record where the
labelled events land in the ranking. Hand-label a few hundred mundane
tracks drawn from the top of the ranking and the same procedure yields a
precision curve. Reviewing only the head of the ranking keeps the
labelling cost bounded.

The output is a ranked review queue with a human at the end. Anomalous
signatures are rare and the corpus is clutter-heavy, so any score
returns its top few percent as unusual whatever the method.
