"""How a stream's reading thread hands frames, and failures, to its reader."""

import pytest

from hessdalen.io.video import prefetched


def test_frames_arrive_in_order() -> None:
    assert list(prefetched(iter(range(10)))) == list(range(10))


def test_a_failure_while_reading_is_raised_to_the_reader() -> None:
    """A reader left waiting for a frame the thread will never send would
    wait forever."""

    def frames():
        yield 1
        yield 2
        raise OSError("the recording ends early")

    read = []
    with pytest.raises(OSError, match="ends early"):
        for frame in prefetched(frames()):
            read.append(frame)

    assert read == [1, 2]
