"""Test the file the detector settings are read from and saved to."""

from dataclasses import replace

import pytest

from hessdalen.config import CONFIG_PATH, Config, config, read_config, write_config


def test_the_config_that_ships_is_readable():
    held = read_config(CONFIG_PATH)

    assert held.frame_height > 0
    assert held.panels in {"recording", "deviation", "both"}
    assert held.settings.device in {"auto", "cpu", "cuda"}


def test_the_settings_survive_being_saved_and_read_back(tmp_path):
    path = tmp_path / "detector.toml"
    held = read_config(CONFIG_PATH)
    changed = replace(
        held,
        frame_height=540,
        panels="deviation",
        settings=replace(held.settings, detection=replace(held.settings.detection, detection_sigma=22.5)),
    )

    write_config(changed, path)

    assert read_config(path) == changed


def test_saving_lets_go_of_the_reading_held_from_before(tmp_path):
    before = config()

    write_config(replace(before, frame_height=540), tmp_path / "detector.toml")

    assert config() == read_config(CONFIG_PATH)


def test_a_setting_the_config_leaves_out_is_refused(tmp_path):
    path = tmp_path / "detector.toml"
    path.write_text('frame_height = 1080\npanels = "both"\ndevice = "auto"\n[detection]\n[background]\n[tracking]\n')

    with pytest.raises(KeyError):
        read_config(path)


def test_a_table_the_config_leaves_out_is_refused(tmp_path):
    path = tmp_path / "detector.toml"
    path.write_text('frame_height = 1080\npanels = "both"\ndevice = "auto"\n')

    with pytest.raises(ValueError, match="detection"):
        read_config(path)


def test_a_frame_height_nothing_offers_is_refused(tmp_path):
    path = tmp_path / "detector.toml"
    held = read_config(CONFIG_PATH)
    write_config(replace(held, frame_height=137), path)

    with pytest.raises(ValueError, match="frame_height"):
        read_config(path)


def test_a_panel_choice_the_dashboard_does_not_draw_is_refused(tmp_path):
    path = tmp_path / "detector.toml"
    write_config(read_config(CONFIG_PATH), path)
    path.write_text(path.read_text().replace('panels = "both"', 'panels = "sideways"'))

    with pytest.raises(ValueError, match="panels"):
        read_config(path)


def test_a_device_the_detector_cannot_use_is_refused(tmp_path):
    path = tmp_path / "detector.toml"
    write_config(read_config(CONFIG_PATH), path)
    path.write_text(path.read_text().replace('device = "auto"', 'device = "quantum"'))

    with pytest.raises(ValueError, match="device"):
        read_config(path)


def test_saving_writes_a_config_into_a_directory_that_is_not_there_yet(tmp_path):
    path = tmp_path / "nested" / "detector.toml"

    write_config(read_config(CONFIG_PATH), path)

    assert isinstance(read_config(path), Config)


def test_saving_keeps_the_heading_that_says_what_the_file_is(tmp_path):
    """Writing the tables carries no comment across, so a save would otherwise
    leave the file saying nothing about itself."""
    path = tmp_path / "detector.toml"

    write_config(read_config(CONFIG_PATH), path)

    assert path.read_text().startswith("# What the detector starts from")
    assert read_config(path) == read_config(CONFIG_PATH)
