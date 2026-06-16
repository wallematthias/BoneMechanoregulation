from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from bonemechreg.timelapse import case_outputs, discover_timelapse_cases
from bonemechreg.post_timelapse import run_post_timelapse_mechanoregulation


def _write_image(path: Path, value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.full((2, 2, 2), value, dtype=np.uint8))
    sitk.WriteImage(image, str(path))


def _write_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(values.astype(np.uint8)), str(path))


def _make_case_fixture(tmp_path: Path):
    root = tmp_path / "dataset"
    case_dir = root / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "analysis" / "pairwise_t0" / "ses-C1_ses-C2"
    remodelling = case_dir / "sub-001_ses-C1_ses-C2_remodelling.nii.gz"
    baseline = case_dir / "sub-001_ses-C1_pairwise_t0_image.nii.gz"
    _write_image(remodelling, value=2)
    _write_image(baseline, value=1)
    return root, discover_timelapse_cases(root)[0]


def test_run_cases_reuses_existing_sed_when_summary_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, case = _make_case_fixture(tmp_path)
    outputs = case_outputs(case)
    outputs["sed"].parent.mkdir(parents=True, exist_ok=True)
    _write_image(outputs["sed"], value=3)
    called = {"solve": 0, "analyze": 0}

    def fake_solve(**kwargs):
        called["solve"] += 1
        return outputs["sed"]

    class FakeResult:
        orf = 2.0
        orr = 0.5
        orf_ci = (1.5, 2.5)
        orr_ci = (0.25, 0.75)
        orr_increasing_strain = 0.5
        orr_decreasing_strain = 2.0
        orr_increasing_strain_ci = (0.25, 0.75)
        orr_decreasing_strain_ci = (1.3333333333333333, 4.0)
        pvalue_form = 0.01
        pvalue_res = 0.02
        conditional_curves = {"F": {"mean": [0.1, 0.2]}, "R": {"mean": [0.2, 0.1]}, "Q": {"mean": [0.7, 0.7]}, "support": [0.1, 0.2]}
        binned_odds_diagnostics = {}
        sample_counts = {"n_sampled_voxels": 10}
        settings = {"profile": "XtremeCTII"}
        plot_paths = {"conditional_curves": outputs["curves"]}

    def fake_mechreg(**kwargs):
        called["analyze"] += 1
        outputs["curves"].write_bytes(b"plot")
        return FakeResult()

    monkeypatch.setattr("bonemechreg.post_timelapse.solve_sed_to_file", fake_solve)
    monkeypatch.setattr("bonemechreg.post_timelapse.mechanoregulation", fake_mechreg)

    summary = run_post_timelapse_mechanoregulation(dataset_root=root, profile="XtremeCTII")

    assert called["solve"] == 0
    assert called["analyze"] == 1
    assert summary["processed"] == 1
    assert outputs["summary"].exists()
    assert outputs["csv"].exists()


def test_run_cases_builds_material_labels_from_native_baseline_segmentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    site = root / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "site-tibia"
    remodelling = site / "analysis" / "visualize" / "sub-001_site-tibia_comp-full_t0-T1_t1-T2_remodelling.nii.gz"
    baseline = site / "transformed_images" / "ses-T1" / "sub-001_site-tibia_ses-T1_image_fused.nii.gz"
    stack = site / "ses-T1" / "stacks"
    segmentation = stack / "sub-001_site-tibia_ses-T1_stack-01_seg.nii.gz"
    trab = stack / "sub-001_site-tibia_ses-T1_stack-01_mask-trab.nii.gz"
    cort = stack / "sub-001_site-tibia_ses-T1_stack-01_mask-cort.nii.gz"
    full = stack / "sub-001_site-tibia_ses-T1_stack-01_mask-full.nii.gz"

    remodelling_values = np.full((2, 2, 2), 2, dtype=np.uint8)
    remodelling_values[1, 1, 1] = 3
    _write_array(remodelling, remodelling_values)
    _write_image(baseline, value=1)

    seg_values = np.zeros((2, 2, 2), dtype=np.uint8)
    seg_values[0, 0, 0] = 1
    seg_values[0, 0, 1] = 1
    _write_array(segmentation, seg_values)
    trab_values = np.zeros((2, 2, 2), dtype=np.uint8)
    trab_values[0, 0, 0] = 1
    _write_array(trab, trab_values)
    cort_values = np.zeros((2, 2, 2), dtype=np.uint8)
    cort_values[0, 0, 1] = 1
    _write_array(cort, cort_values)
    _write_array(full, np.ones((2, 2, 2), dtype=np.uint8))

    case = discover_timelapse_cases(root)[0]
    outputs = case_outputs(case)

    def fake_solve(**kwargs):
        material = sitk.GetArrayFromImage(sitk.ReadImage(str(kwargs["material_image_path"])))
        assert material[0, 0, 0] == 100
        assert material[0, 0, 1] == 127
        assert material[1, 1, 1] == 0
        _write_image(outputs["sed"], value=1)
        return outputs["sed"]

    class FakeResult:
        orf = 2.0
        orr = 0.5
        orf_ci = (1.5, 2.5)
        orr_ci = (0.25, 0.75)
        orr_increasing_strain = 0.5
        orr_decreasing_strain = 2.0
        orr_increasing_strain_ci = (0.25, 0.75)
        orr_decreasing_strain_ci = (1.3333333333333333, 4.0)
        pvalue_form = 0.01
        pvalue_res = 0.02
        conditional_curves = {"F": {"mean": [0.1]}, "R": {"mean": [0.2]}, "Q": {"mean": [0.7]}, "support": [0.1]}
        binned_odds_diagnostics = {}
        sample_counts = {"n_sampled_voxels": 10}
        settings = {"profile": "XtremeCTII"}
        plot_paths = {"conditional_curves": outputs["curves"]}

    def fake_mechreg(**kwargs):
        assert kwargs["analysis_mask"] is not None
        outputs["curves"].write_bytes(b"plot")
        return FakeResult()

    monkeypatch.setattr("bonemechreg.post_timelapse.solve_sed_to_file", fake_solve)
    monkeypatch.setattr("bonemechreg.post_timelapse.mechanoregulation", fake_mechreg)

    summary = run_post_timelapse_mechanoregulation(dataset_root=root, profile="XtremeCTII")

    assert summary["processed"] == 1
    assert outputs["material"].exists()


def test_run_cases_dry_run_reports_discovered_cases(tmp_path: Path) -> None:
    root, _case = _make_case_fixture(tmp_path)

    summary = run_post_timelapse_mechanoregulation(
        dataset_root=root,
        profile="XtremeCTII",
        dry_run=True,
    )

    assert summary["discovered"] == 1
    assert summary["processed"] == 0
    assert summary["dry_run"] is True


def test_run_cases_skips_complete_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, case = _make_case_fixture(tmp_path)
    outputs = case_outputs(case)
    outputs["sed"].parent.mkdir(parents=True, exist_ok=True)
    _write_image(outputs["sed"], value=3)
    outputs["summary"].write_text(json.dumps({"ok": True}), encoding="utf-8")
    outputs["csv"].write_text("ok\n", encoding="utf-8")
    outputs["curves"].write_bytes(b"plot")

    def fail_solve(**kwargs):
        raise AssertionError("solve should not run")

    monkeypatch.setattr("bonemechreg.post_timelapse.solve_sed_to_file", fail_solve)

    summary = run_post_timelapse_mechanoregulation(dataset_root=root, profile="XtremeCTII")

    assert summary["skipped"] == 1
    assert summary["processed"] == 0
