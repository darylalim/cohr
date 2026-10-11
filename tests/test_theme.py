"""PLAN §7 theme: light/dark parity, WCAG contrast, offline fonts, no telemetry."""

import tomllib
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
MODES = ("light", "dark")
SEMANTIC = ("blueColor", "greenColor", "orangeColor", "violetColor", "redColor", "grayColor")


@pytest.fixture(scope="module")
def config() -> dict:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def luminance(hex_color: str) -> float:
    """WCAG 2.x relative luminance of #RRGGBB."""
    rgb = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_contrast_helper_matches_known_values() -> None:
    assert contrast("#FFFFFF", "#000000") == pytest.approx(21.0)
    assert contrast("#777777", "#FFFFFF") == pytest.approx(4.48, abs=0.01)


def test_light_and_dark_set_the_same_keys(config: dict) -> None:
    light, dark = config["theme"]["light"], config["theme"]["dark"]
    assert set(light) == set(dark)
    assert set(light["sidebar"]) == set(dark["sidebar"])
    assert len(light["chartCategoricalColors"]) == len(dark["chartCategoricalColors"])


@pytest.mark.parametrize("mode", MODES)
def test_contrast(config: dict, mode: str) -> None:
    t = config["theme"][mode]
    bg = t["backgroundColor"]
    assert contrast(t["textColor"], bg) >= 7.0
    assert contrast(t["linkColor"], bg) >= 4.5
    for key in SEMANTIC:
        assert contrast(t[key], bg) >= 4.5, key
    assert contrast("#FFFFFF", t["primaryColor"]) >= 4.5
    assert contrast(t["primaryColor"], bg) >= 3.0


@pytest.mark.parametrize("mode", MODES)
def test_sidebar_is_recessed(config: dict, mode: str) -> None:
    """Both modes: the sidebar is darker than the main area (§7 layering)."""
    t = config["theme"][mode]
    assert luminance(t["sidebar"]["backgroundColor"]) < luminance(t["backgroundColor"])


def test_offline_and_no_telemetry(config: dict) -> None:
    assert "fonts.googleapis.com" not in CONFIG.read_text(encoding="utf-8")
    assert config["browser"]["gatherUsageStats"] is False
    assert config["server"]["address"] == "localhost"
    assert config["server"]["maxUploadSize"] == 4096
    assert config["server"]["fileWatcherType"] == "none"  # Phase 5a: no torch import attempts
    assert config["theme"]["font"] == "sans-serif"
