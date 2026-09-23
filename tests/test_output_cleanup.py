"""What the cleanup offers to delete, and what it leaves where it is."""

import argparse
import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "dev" / "clean_outputs.py"


@pytest.fixture(scope="module")
def cleanup():
    spec = importlib.util.spec_from_file_location("clean_outputs", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def root(tmp_path) -> Path:
    _written(tmp_path / "data/out/dashboard/Cam1_2025-02-20__11-40-00__abc123.mp4")
    _written(tmp_path / "data/out/dashboard/live/segment.mp4")
    _written(tmp_path / "data/out/dashboard/tracks/track_7.mp4")
    _written(tmp_path / "data/out/sift/clips/Cam2_2025-12-22__00-00-01_UTC_p.mkv")
    _written(tmp_path / "data/out/sift/ledger.jsonl")
    _written(tmp_path / "data/out/sift/bulk-ledger.jsonl")
    _written(tmp_path / "data/corpus/tracks/birds/morning/a.parquet")
    _written(tmp_path / "data/cuts/clips/a_clip_0.000_5.000.mkv")
    return tmp_path


def test_a_group_holds_what_its_part_of_the_project_wrote(cleanup, root) -> None:
    held = {group.name: [path.name for path in group.files()] for group in cleanup.groups_under(root)}

    assert held["runs"] == ["Cam1_2025-02-20__11-40-00__abc123.mp4"]
    assert held["live"] == ["segment.mp4"]
    assert held["staging"] == ["Cam2_2025-12-22__00-00-01_UTC_p.mkv"]


def test_no_group_offers_a_ledger_or_a_tracked_file(cleanup, root) -> None:
    offered = {path for group in cleanup.groups_under(root) for path in group.files()}

    assert root / "data/out/sift/ledger.jsonl" not in offered
    assert root / "data/out/sift/bulk-ledger.jsonl" not in offered
    assert root / "data/corpus/tracks/birds/morning/a.parquet" not in offered
    assert root / "data/cuts/clips/a_clip_0.000_5.000.mkv" not in offered


def test_removing_one_group_leaves_the_others(cleanup, root) -> None:
    cleanup.main(argparse.Namespace(root=root, remove=["live"]))

    assert not (root / "data/out/dashboard/live/segment.mp4").exists()
    assert (root / "data/out/dashboard/tracks/track_7.mp4").exists()
    assert (root / "data/out/sift/ledger.jsonl").exists()


def test_what_a_sift_is_using_is_left_alone_while_it_runs(cleanup, root, monkeypatch) -> None:
    monkeypatch.setattr(cleanup, "sifting", lambda: True)

    with pytest.raises(SystemExit, match="sift run is going"):
        cleanup.main(argparse.Namespace(root=root, remove=["staging"]))

    assert (root / "data/out/sift/clips/Cam2_2025-12-22__00-00-01_UTC_p.mkv").exists()


def test_a_report_says_what_a_group_costs_to_have_back(cleanup, root) -> None:
    [live] = [group for group in cleanup.groups_under(root) if group.name == "live"]

    assert "1 files" in cleanup.report(live)
    assert "back from the live view" in cleanup.report(live)


def _written(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 16)
