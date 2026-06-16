from __future__ import annotations

from pathlib import Path

from bonemechreg.cli import main


def test_cli_run_dry_run_reports_discovered_cases(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "bonemechreg.cli.run_post_timelapse_mechanoregulation",
        lambda **kwargs: {"discovered": 2, "processed": 0, "skipped": 0, "failed": 0, "dry_run": True},
    )

    exit_code = main(["run", str(tmp_path), "--profile", "XtremeCTII", "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "discovered=2" in captured.out


def test_cli_run_invokes_workflow_with_expected_flags(tmp_path: Path, monkeypatch, capsys) -> None:
    recorded: dict[str, object] = {}

    def fake_run(**kwargs):
        recorded.update(kwargs)
        return {"discovered": 1, "processed": 1, "skipped": 0, "failed": 0, "dry_run": False}

    monkeypatch.setattr("bonemechreg.cli.run_post_timelapse_mechanoregulation", fake_run)

    exit_code = main(["run", str(tmp_path), "--profile", "XtremeCTII", "--overwrite"])
    capsys.readouterr()

    assert exit_code == 0
    assert recorded["profile"] == "XtremeCTII"
    assert recorded["overwrite"] is True


def test_cli_analyze_runs_standalone_folder(tmp_path: Path, monkeypatch, capsys) -> None:
    recorded: dict[str, object] = {}
    csv_path = tmp_path / "mechanoregulation" / "summary.csv"
    png_path = tmp_path / "mechanoregulation" / "curves.png"

    def fake_analyze(input_dir):
        recorded["input_dir"] = input_dir
        return {"csv": csv_path, "png": png_path, "json": tmp_path / "summary.json"}

    monkeypatch.setattr("bonemechreg.cli.run_standalone_analysis", fake_analyze)

    exit_code = main(["analyze", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert recorded["input_dir"] == tmp_path
    assert f"csv={csv_path}" in captured.out
    assert f"png={png_path}" in captured.out
