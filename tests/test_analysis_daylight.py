"""When a recording was made, and whether the sun was up over Hessdalen."""

from datetime import datetime, timezone

import pytest

from hessdalen.analysis.daylight import (
    DAY,
    LATITUDE,
    LONGITUDE,
    NIGHT,
    TWILIGHT,
    UNKNOWN,
    daylight,
    recording_time,
    sun_altitude,
)


def test_a_clock_name_is_norwegian_local_time() -> None:
    assert recording_time("Cam1_2025-08-20__12-58-51_bird_017.mkv") == datetime(
        2025, 8, 20, 10, 58, 51, tzinfo=timezone.utc
    )


def test_a_clock_name_marked_utc_is_utc() -> None:
    assert recording_time("Cam5_2026-07-25__11-40-00_UTC.mkv") == datetime(2026, 7, 25, 11, 40, 0, tzinfo=timezone.utc)
    assert recording_time("Cam2_2025-12-24__12-20-01_UTC_p.mkv") == datetime(
        2025, 12, 24, 12, 20, 1, tzinfo=timezone.utc
    )


def test_an_epoch_name_holds_the_moment_itself() -> None:
    assert recording_time("Cam1-20250820-125851-1755687531339-7_005.mp4") == datetime(
        2025, 8, 20, 10, 58, 51, 339000, tzinfo=timezone.utc
    )


def test_a_name_with_no_time_is_unknown() -> None:
    assert recording_time("meteor_clip.mkv") is None
    assert daylight("meteor_clip.mkv") == UNKNOWN


def test_the_midsummer_noon_sun_stands_where_the_latitude_puts_it() -> None:
    noon = datetime(2025, 6, 21, 11, 15, tzinfo=timezone.utc)

    assert sun_altitude(noon, latitude=LATITUDE, longitude=LONGITUDE) == pytest.approx(90 - LATITUDE + 23.44, abs=0.5)


def test_hessdalen_has_twilight_at_midsummer_midnight_and_night_at_midwinter() -> None:
    assert daylight("Cam1_2025-06-21__01-00-00.mkv") == TWILIGHT
    assert daylight("Cam1_2025-12-21__01-00-00.mkv") == NIGHT
    assert daylight("Cam1_2025-06-21__13-00-00.mkv") == DAY
