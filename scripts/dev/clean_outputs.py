"""Report what the working folders hold, and delete what can be built again.

The dashboard renders a video per detection run, a segment per live view
and a clip per track it is asked about, the sift stages a download per
recording it is about to read, and none of that is tracked. It reaches
tens of gigabytes over a few days of work, which is what runs the disk
down under the floor the sift refuses to fetch below.

The ledgers under data/out/sift are the exception and are never listed
here. They hold what the detector found over a terabyte of video, and a
rescore reads them rather than running any of it again.
"""

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIFT_SCRIPT = "sift_drive.py"
CLEANUP_SCRIPT = "clean_outputs"


@dataclass(frozen=True, slots=True)
class Group:
    """A set of files that one part of the project writes and can write
    again."""

    name: str
    directory: Path
    patterns: tuple[str, ...]
    rebuilt_by: str
    """What it costs to have these files back."""

    staged: bool = False
    """Whether a running sift is using this directory as it goes."""

    def files(self) -> list[Path]:
        found = {path for pattern in self.patterns for path in self.directory.glob(pattern) if path.is_file()}
        return sorted(found)


def main(args: argparse.Namespace) -> None:
    groups = groups_under(args.root)
    for group in groups:
        print(report(group))

    print(f"\n{free_gigabytes(args.root):.1f} GB free")
    if not args.remove:
        return

    taken = [group for group in groups if group.name in args.remove]
    refuse_while_sifting(taken)
    for group in taken:
        print(f"removed {delete(group)} of {group.name}")

    print(f"{free_gigabytes(args.root):.1f} GB free")


def groups_under(root: Path) -> list[Group]:
    """Every set of rebuildable files, largest first in the order they are
    worth deleting."""
    dashboard = root / "data" / "out" / "dashboard"
    return [
        Group(
            name="live",
            directory=dashboard / "live",
            patterns=("**/*",),
            rebuilt_by="the live view, when it next plays that stretch",
        ),
        Group(
            name="runs",
            directory=dashboard,
            patterns=("*.mp4", "*.json"),
            rebuilt_by="the recordings page, by running the detector again",
        ),
        Group(
            name="track-clips",
            directory=dashboard / "tracks",
            patterns=("**/*",),
            rebuilt_by="the track map, when a track is picked again",
        ),
        Group(
            name="fetched",
            directory=dashboard / "videos",
            patterns=("**/*",),
            rebuilt_by="the track map, by fetching the recording again",
        ),
        Group(
            name="tables",
            directory=root / "data" / "out" / "sift",
            patterns=("*.csv",),
            rebuilt_by="a sift run with --table, out of the ledger",
        ),
        Group(
            name="staging",
            directory=root / "data" / "out" / "sift" / "clips",
            patterns=("*.mkv", "*.mp4", "*.part"),
            rebuilt_by="the next sift run, by fetching those recordings again",
            staged=True,
        ),
        Group(
            name="corpus-videos",
            directory=root / "data" / "corpus" / "videos",
            patterns=("*.mkv", "*.mp4"),
            rebuilt_by="a sift collect, by fetching every kept recording again",
            staged=True,
        ),
    ]


def report(group: Group) -> str:
    files = group.files()
    return f"{group.name:14s} {describe(files):>12s}  {group.directory}\n{'':14s} back from {group.rebuilt_by}"


def refuse_while_sifting(groups: list[Group]) -> None:
    """Stop before deleting what a running sift is working on.

    A sift deletes each staged recording once it has read it and fetches
    the kept ones into the corpus, so pulling those files out from under
    a run loses the recordings it is holding.
    """
    if any(group.staged for group in groups) and sifting():
        raise SystemExit("A sift run is going. Stop it before removing what it is using, or leave those groups out.")


def sifting() -> bool:
    """Whether a sift run is on this machine, read off the process table.

    The shell that started this cleanup carries the whole command line,
    so a search for the script's name finds this process as well unless
    the cleanup's own name is excluded.
    """
    own = str(os.getpid())
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or entry.name == own:
            continue
        command = _command_of(entry)
        if SIFT_SCRIPT in command and CLEANUP_SCRIPT not in command:
            return True
    return False


def _command_of(process: Path) -> str:
    try:
        return (process / "cmdline").read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


def delete(group: Group) -> str:
    """Remove the group's files, and say what was in it."""
    files = group.files()
    held = describe(files)

    for path in files:
        path.unlink(missing_ok=True)
    return held


def describe(files: list[Path]) -> str:
    total = sum(path.stat().st_size for path in files if path.exists())
    return f"{len(files)} files, {total / 1e9:.1f} GB"


def free_gigabytes(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def parse_args() -> argparse.Namespace:
    names = [group.name for group in groups_under(REPO_ROOT)]
    parser = argparse.ArgumentParser(description="Report the rebuildable working files, and delete the named ones")
    parser.add_argument("--remove", nargs="+", choices=names, default=[], metavar="GROUP", help=f"One of {names}")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository to work in")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
