"""Test the file the detector settings are written to and read back from."""

from hessdalen.dashboard.settings import SETTINGS, as_file, defaults, from_file


def test_every_setting_opens_inside_the_range_its_slider_offers():
    for key, setting in SETTINGS.items():
        assert setting.lowest <= setting.default <= setting.highest, key


def test_the_settings_survive_being_written_and_read_back():
    written = as_file(defaults())

    reading = from_file(written)

    assert reading.problem == ""
    assert reading.settings == defaults()
    assert reading.held == 0


def test_a_setting_the_file_leaves_out_is_left_out():
    reading = from_file(b'{"detection_sigma": 20.0}')

    assert reading.settings == {"detection_sigma": 20.0}


def test_a_value_past_the_end_of_its_slider_is_brought_back_to_it():
    reading = from_file(b'{"detection_sigma": 999.0, "min_pixels": -4}')

    assert reading.settings == {"detection_sigma": SETTINGS["detection_sigma"].highest, "min_pixels": 1}
    assert reading.held == 2


def test_a_whole_setting_stays_whole():
    reading = from_file(b'{"min_pixels": 7.0}')

    assert reading.settings == {"min_pixels": 7}
    assert isinstance(reading.settings["min_pixels"], int)


def test_a_name_the_detector_does_not_know_is_passed_over():
    reading = from_file(b'{"detection_sigma": 20.0, "colour": 3}')

    assert reading.settings == {"detection_sigma": 20.0}


def test_a_file_that_is_not_json_is_refused():
    reading = from_file(b"not json at all")

    assert reading.settings == {}
    assert reading.problem


def test_a_file_naming_no_setting_is_refused():
    reading = from_file(b'{"colour": 3}')

    assert reading.settings == {}
    assert reading.problem


def test_a_file_holding_something_other_than_a_set_of_settings_is_refused():
    reading = from_file(b"[1, 2, 3]")

    assert reading.settings == {}
    assert reading.problem


def test_a_setting_written_as_true_is_not_a_number():
    reading = from_file(b'{"detection_sigma": true}')

    assert reading.settings == {}
