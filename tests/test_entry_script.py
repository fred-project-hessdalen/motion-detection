"""Running the entry script without flags has to mean the library defaults.

The settings are slotted dataclasses, so reading a default off the class
gives a slot descriptor and not the number, which argparse accepts and
passes on.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from hessdalen.config import config

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "dev" / "detect_movement.py"


@pytest.fixture(scope="module")
def entry_script():
    spec = importlib.util.spec_from_file_location("detect_movement", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bare_invocation_builds_the_settings_the_config_holds(entry_script, monkeypatch):
    monkeypatch.setattr(sys, "argv", [SCRIPT_PATH.name, "recording.mkv"])

    settings = entry_script.settings_from_args(entry_script.parse_args())

    assert settings == config().settings


def test_a_flag_reaches_the_settings(entry_script, monkeypatch):
    monkeypatch.setattr(sys, "argv", [SCRIPT_PATH.name, "recording.mkv", "--detection-sigma", "9.5"])

    settings = entry_script.settings_from_args(entry_script.parse_args())

    assert settings.detection.detection_sigma == 9.5
