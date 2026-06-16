from __future__ import annotations

import tomllib
from pathlib import Path

import bonemechreg


def test_console_script_is_declared_in_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert scripts["mechanoregulation"] == "bonemechreg.cli:main"


def test_package_exports_post_timelapse_surface_only() -> None:
    assert hasattr(bonemechreg, "mechanoregulation")
