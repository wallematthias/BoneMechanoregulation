from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import SimpleITK as sitk

from bonemechreg.parosol import _run_parosol_profile, solve_sed_to_file


def _install_fake_parosol_cli(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sed: np.ndarray,
    exit_code: int = 0,
) -> dict[str, object]:
    calls: dict[str, object] = {}

    def fake_main(argv):
        calls["argv"] = list(argv)
        output_dir = Path(argv[argv.index("--output") + 1])
        if exit_code == 0:
            (output_dir / "fields").mkdir(parents=True)
            sitk.WriteImage(sitk.GetImageFromArray(sed), str(output_dir / "fields" / "sed.nii.gz"))
        return exit_code

    fake_cli = types.ModuleType("parosol_py.cli")
    fake_cli.main = fake_main
    monkeypatch.setitem(sys.modules, "parosol_py", types.ModuleType("parosol_py"))
    monkeypatch.setitem(sys.modules, "parosol_py.cli", fake_cli)
    return calls


def test_solve_case_sed_passes_profile_and_writes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    material = tmp_path / "baseline_material.nii.gz"
    image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8))
    sitk.WriteImage(image, str(material))
    calls = _install_fake_parosol_cli(monkeypatch, sed=np.full((2, 2, 2), 0.5, dtype=np.float32))

    out = solve_sed_to_file(
        material_image_path=material,
        output_path=tmp_path / "case_sed.nii.gz",
        profile="XtremeCTII",
    )

    argv = calls["argv"]
    assert isinstance(argv, list)
    assert argv[:4] == [str(material), "--profile", "XtremeCTII", "--output"]
    assert Path(argv[4]).name == "case"
    assert out.exists()


def test_solver_import_error_is_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = tmp_path / "baseline.nii.gz"
    image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.float32))
    sitk.WriteImage(image, str(baseline))

    def fake_run(*, image_path: Path, profile: str) -> np.ndarray:
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
    _install_fake_parosol_cli(monkeypatch, sed=expected)

    sed = _run_parosol_profile(image_path=tmp_path / "baseline.nii.gz", profile="XtremeCTII")

    np.testing.assert_allclose(sed, expected)
