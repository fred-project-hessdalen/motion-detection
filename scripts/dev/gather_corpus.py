"""Gather a sifted corpus of track files into one table of descriptions.

The sift writes one track file per clip and fills an event folder at a
time, so this can be run while it is still going and re-run as more
folders land.
"""

import argparse
import collections
from pathlib import Path

from hessdalen.analysis.corpus import read_corpus, write_corpus


def main(args: argparse.Namespace) -> None:
    corpus = read_corpus(args.tracks)
    if not corpus.clips:
        raise SystemExit(f"No track files under {args.tracks}.")

    events = {(clip.label, clip.event) for clip in corpus.clips}
    by_label = collections.Counter(label for label, _event in events)
    print(f"{len(corpus.clips)} clips in {len(events)} event folders")
    for label, count in sorted(by_label.items()):
        print(f"  {label:12s} {count} folders")

    print(f"{write_corpus(args.output, corpus)} tracks written to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gather a sifted corpus into one table")
    parser.add_argument("tracks", type=Path, help="Corpus root holding class and event folders")
    parser.add_argument("output", type=Path, help="File to write the gathered descriptions to")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
