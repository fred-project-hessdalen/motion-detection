"""Test the ffmpeg call a built video is written through."""

from pathlib import Path

from hessdalen.dashboard.encoder import FALLBACK_FPS, LIVE, STORED, encode, encoder_command, even, playback_rate


def command(encoding=STORED) -> list[str]:
    return encoder_command(
        Path("out.mp4"),
        width=1920,
        height=2160,
        frames_per_second=25.0,
        encoding=encoding,
    )


def test_the_call_carries_the_preset_and_quality_it_was_given():
    stored = command(STORED)
    live = command(LIVE)

    assert stored[stored.index("-preset") + 1] == "veryfast"
    assert live[live.index("-preset") + 1] == "ultrafast"
    assert stored[stored.index("-crf") + 1] == live[live.index("-crf") + 1] == "26"


def test_the_call_declares_the_frames_it_is_fed():
    given = command()

    assert given[given.index("-s") + 1] == "1920x2160"
    assert given[given.index("-pix_fmt") + 1] == "yuv420p"
    assert given[given.index("-r") + 1].startswith("25.0")


def test_a_side_that_cannot_be_halved_loses_its_last_line():
    assert even(1081) == 1080
    assert even(1080) == 1080


def test_a_container_that_reports_no_rate_falls_back_to_one():
    assert playback_rate(0.0) == FALLBACK_FPS
    assert playback_rate(25.0) == 25.0


def test_a_source_holding_no_frames_writes_nothing(tmp_path):
    """The encoder is never started, so no file is left for a later read to
    take for a whole one."""
    output = tmp_path / "empty.mp4"

    written = encode(
        iter(()),
        output=output,
        width=64,
        height=64,
        frames_per_second=25.0,
        encoding=LIVE,
    )

    assert written == 0
    assert not output.exists()
