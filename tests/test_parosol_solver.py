from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import SimpleITK as sitk

from bonemechreg.parosol import _run_parosol_profile, solve_sed_to_file


def _install_fake_parosol_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sed: np.ndarray,
    exit_code: int = 0,
    stdout: str = "ok\n",
    stderr: str = "",
) -> dict[str, object]:
    calls: dict[str, object] = {}

    class FakeProcess:
        def __init__(self, argv, stdout=None, stderr=None, text=True):
            calls["argv"] = list(argv)
            stdout.write(stdout_text)
            stderr.write(stderr_text)
            output_dir = Path(argv[argv.index("--output") + 1])
            if exit_code == 0:
                (output_dir / "fields").mkdir(parents=True)
                sitk.WriteImage(sitk.GetImageFromArray(sed), str(output_dir / "fields" / "sed.nii.gz"))

        def wait(self):
            return exit_code

    stdout_text = stdout
    stderr_text = stderr
    monkeypatch.setattr("bonemechreg.parosol.subprocess.Popen", FakeProcess)
    return calls


def test_solve_case_sed_passes_profile_and_writes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    material = tmp_path / "baseline_material.nii.gz"
    image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8))
    sitk.WriteImage(image, str(material))
    calls = _install_fake_parosol_subprocess(monkeypatch, sed=np.full((2, 2, 2), 0.5, dtype=np.float32))

    out = solve_sed_to_file(
        material_image_path=material,
        output_path=tmp_path / "case_sed.nii.gz",
        profile="XtremeCTII",
    )

    argv = calls["argv"]
    assert isinstance(argv, list)
    assert argv[:3] == [sys.executable, "-m", "parosol_py.cli"]
    assert argv[3:6] == [str(material), "--profile", "XtremeCTII"]
    assert argv[6] == "--output"
    assert Path(argv[7]).name == "case"
    assert out.exists()


def test_solve_case_sed_preserves_parosol_wrapper_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    material = tmp_path / "baseline_material.nii.gz"
    image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8))
    sitk.WriteImage(image, str(material))
    _install_fake_parosol_subprocess(
        monkeypatch,
        sed=np.full((2, 2, 2), 0.5, dtype=np.float32),
        stdout="solver stdout\n",
        stderr="solver stderr\n",
    )
    debug_dir = tmp_path / "debug_solve"

    solve_sed_to_file(
        material_image_path=material,
        output_path=tmp_path / "case_sed.nii.gz",
        profile="XtremeCTII",
        debug_dir=debug_dir,
    )

    assert (debug_dir / "bonemechreg_parosol_command.txt").read_text(encoding="utf-8").startswith(
        f"{sys.executable} -m parosol_py.cli {material} --profile XtremeCTII --output {debug_dir}"
    )
    assert (debug_dir / "bonemechreg_parosol_stdout.log").read_text(encoding="utf-8") == "solver stdout\n"
    assert (debug_dir / "bonemechreg_parosol_stderr.log").read_text(encoding="utf-8") == "solver stderr\n"
    assert (debug_dir / "parosol_live_stdout.log").read_text(encoding="utf-8") == "solver stdout\n"
    assert (debug_dir / "parosol_live_stderr.log").read_text(encoding="utf-8") == "solver stderr\n"
    assert (debug_dir / "bonemechreg_parosol_exit_code.txt").read_text(encoding="utf-8") == "0\n"


def test_parosol_wrapper_logs_failed_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    material = tmp_path / "baseline_material.nii.gz"
    image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8))
    sitk.WriteImage(image, str(material))
    _install_fake_parosol_subprocess(
        monkeypatch,
        sed=np.full((2, 2, 2), 0.5, dtype=np.float32),
        exit_code=2,
        stderr="native solver was not reached\n",
    )
    debug_dir = tmp_path / "debug_solve"

    with pytest.raises(RuntimeError, match="failed with exit code 2"):
        solve_sed_to_file(
            material_image_path=material,
            output_path=tmp_path / "case_sed.nii.gz",
            profile="XtremeCTII",
            debug_dir=debug_dir,
        )

    assert (debug_dir / "bonemechreg_parosol_command.txt").exists()
    assert "native solver was not reached" in (debug_dir / "bonemechreg_parosol_stderr.log").read_text(
        encoding="utf-8"
    )
    assert (debug_dir / "bonemechreg_parosol_exit_code.txt").read_text(encoding="utf-8") == "2\n"


def test_solver_import_error_is_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = tmp_path / "baseline.nii.gz"
    image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.float32))
    sitk.WriteImage(image, str(baseline))

    def fake_run(*, image_path: Path, profile: str, debug_dir=None) -> np.ndarray:
        raise ModuleNotFoundError("No module named 'parosol_py'")

    monkeypatch.setattr("bonemechreg.parosol._run_parosol_profile", fake_run)

    with pytest.raises(RuntimeError, match="parosol-py is required"):
        solve_sed_to_file(
            material_image_path=baseline,
            output_path=tmp_path / "case_sed.nii.gz",
            profile="XtremeCTII",
        )


def test_parosol_profile_reads_exported_sed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = np.full((2, 3, 4), 0.25, dtype=np.float32)
    _install_fake_parosol_subprocess(monkeypatch, sed=expected)

    sed = _run_parosol_profile(image_path=tmp_path / "baseline.nii.gz", profile="XtremeCTII")

    assert np.allclose(sed, expected)
