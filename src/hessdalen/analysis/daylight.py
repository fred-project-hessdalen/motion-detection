"""When a recording was made, and whether the sun was up.

A recording's name holds its start time. Names of the form
Cam1_2025-08-20__12-58-51 give Norwegian local time unless they say
UTC, and names of the form Cam1-20250820-125851-1755687531339-7 hold
the same moment as milliseconds since the epoch. The cameras stand in
the Hessdalen valley, and the sun's altitude there at that moment
follows from the standard solar position formulas, which are good to
a fraction of a degree, more than enough to say day, twilight or
night.

The position is the valley's, from the Hessdalen lights article of
the English Wikipedia, at 62.7933 north and 11.1883 east. The cameras
stand within a few kilometres of it, which moves the sun's altitude
by under a tenth of a degree.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

LATITUDE = 62.7933
LONGITUDE = 11.1883
LOCAL_TIME = ZoneInfo("Europe/Oslo")

DAY = "day"
TWILIGHT = "twilight"
NIGHT = "night"
UNKNOWN = "unknown"
"""What a recording is under when its name holds no time."""

DAYLIGHTS = (DAY, TWILIGHT, NIGHT)

TWILIGHT_BELOW = 0.0
NIGHT_BELOW = -6.0
"""Sun altitudes in degrees under which it is twilight and under which
it is night. Six degrees under the horizon is the end of civil
twilight, below which the sky no longer lights the ground."""

CLOCK_NAME = re.compile(r"^Cam\d+_(\d{4})-(\d{2})-(\d{2})__(\d{2})-(\d{2})-(\d{2})(?P<rest>.*)$")
EPOCH_NAME = re.compile(r"^Cam\d+-\d{8}-\d{6}-(?P<epoch>\d{13})-")


def recording_time(name: str) -> datetime | None:
    """The moment a recording started, in UTC, or None when its name does
    not say."""
    stem = name.rsplit(".", 1)[0]
    epoch = EPOCH_NAME.match(stem)
    if epoch:
        return datetime.fromtimestamp(int(epoch["epoch"]) / 1000.0, tz=timezone.utc)
    clock = CLOCK_NAME.match(stem)
    if clock is None:
        return None
    year, month, day, hour, minute, second = (int(clock.group(place)) for place in range(1, 7))
    zone = timezone.utc if "_UTC" in clock["rest"] else LOCAL_TIME
    return datetime(year, month, day, hour, minute, second, tzinfo=zone).astimezone(timezone.utc)


def sun_altitude(when: datetime, *, latitude: float, longitude: float) -> float:
    """How high the sun stood over the horizon at that moment and place, in
    degrees, negative below the horizon."""
    moment = when.astimezone(timezone.utc)
    day_of_year = moment.timetuple().tm_yday
    hours = moment.hour + moment.minute / 60.0 + moment.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hours - 12.0) / 24.0)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    true_solar_minutes = hours * 60.0 + equation_of_time + 4.0 * longitude
    hour_angle = math.radians(true_solar_minutes / 4.0 - 180.0)
    lat = math.radians(latitude)
    cos_zenith = math.sin(lat) * math.sin(declination) + math.cos(lat) * math.cos(declination) * math.cos(hour_angle)
    return 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith))))


def daylight_of(altitude: float) -> str:
    """Day, twilight or night for a sun altitude in degrees."""
    if altitude > TWILIGHT_BELOW:
        return DAY
    if altitude > NIGHT_BELOW:
        return TWILIGHT
    return NIGHT


def daylight(name: str) -> str:
    """Whether the sun was up over Hessdalen when the recording started."""
    when = recording_time(name)
    if when is None:
        return UNKNOWN
    return daylight_of(sun_altitude(when, latitude=LATITUDE, longitude=LONGITUDE))
