from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from bonemechreg.timelapse import case_outputs, discover_timelapse_cases
from bonemechreg.timelapse import TimelapseCase
from bonemechreg.post_timelapse import (
    _read_analysis_mask,
    run_post_timelapse_case,
    run_post_timelapse_mechanoregulation,
)


def _write_image(path: Path, value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.full((2, 2, 2), value, dtype=np.uint8))
    sitk.WriteImage(image, str(path))


def _write_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(values.astype(np.uint8)), str(path))


def _write_float_image(path: Path, values: np.ndarray, *, spacing=(1.0, 1.0, 1.0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(values.astype(np.float32))
    image.SetSpacing(tuple(float(value) for value in spacing))
    sitk.WriteImage(image, str(path))


def _make_case_fixture(tmp_path: Path):
    root = tmp_path / "dataset"
    case_dir = root / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "analysis" / "pairwise_t0" / "ses-C1_ses-C2"
    remodelling = case_dir / "sub-001_ses-C1_ses-C2_remodelling.nii.gz"
    baseline = case_dir / "sub-001_ses-C1_pairwise_t0_image.nii.gz"
    _write_image(remodelling, value=2)
    _write_image(baseline, value=1)
    return root, discover_timelapse_cases(root)[0]


def test_read_analysis_mask_resamples_to_remodelling_grid(tmp_path: Path) -> None:
    remodelling = sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.uint8))
    remodelling.SetSpacing((0.08, 0.08, 0.08))
    mask = sitk.GetImageFromArray(np.ones((4, 4, 4), dtype=np.uint8))
    mask.SetSpacing((0.08, 0.08, 0.08))
    mask_path = tmp_path / "full_mask.nii.gz"
    sitk.WriteImage(mask, str(mask_path))

    aligned = _read_analysis_mask(mask_path, remodelling, roi="full")

    assert aligned is not None
    assert aligned.GetSize() == remodelling.GetSize()
    assert aligned.GetSpacing() == remodelling.GetSpacing()
    assert aligned.GetOrigin() == remodelling.GetOrigin()
    assert aligned.GetDirection() == remodelling.GetDirection()
    assert np.all(sitk.GetArrayFromImage(aligned) == 1)


def test_run_case_resamples_matched_sed_to_remodelling_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remodelling = tmp_path / "remodelling.nii.gz"
    sed = tmp_path / "sed.nii.gz"
    _write_array(remodelling, np.full((2, 2, 2), 2, dtype=np.uint8))
    _write_float_image(sed, np.ones((4, 4, 4), dtype=np.float32), spacing=(0.5, 0.5, 0.5))
    case = TimelapseCase(
        subject_id="001",
        case_id="scene-row-01",
        baseline_image_path=remodelling,
        remodelling_image_path=remodelling,
        output_dir=tmp_path / "out",
        baseline_sed_path=sed,
        full_mask_path=None,
    )
    outputs = case_outputs(case)

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
        assert kwargs["baseline_strain"].GetSize() == sitk.ReadImage(str(remodelling)).GetSize()
        assert kwargs["baseline_strain"].GetSpacing() == sitk.ReadImage(str(remodelling)).GetSpacing()
        outputs["curves"].write_bytes(b"plot")
        return FakeResult()

    monkeypatch.setattr("bonemechreg.post_timelapse.solve_sed_to_file", lambda **kwargs: sed)
    monkeypatch.setattr("bonemechreg.post_timelapse.mechanoregulation", fake_mechreg)

    summary = run_post_timelapse_case(case, profile="XtremeCTII", verbose=True)

    assert summary["processed"] == 1


def test_run_case_preserves_index_aligned_sed_when_physical_metadata_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remodelling = tmp_path / "remodelling.nii.gz"
    sed = tmp_path / "sed.nii.gz"
    remodelling_image = sitk.GetImageFromArray(np.full((3, 3, 3), 2, dtype=np.uint8))
    remodelling_image.SetSpacing((0.082, 0.082, 0.082))
    remodelling_image.SetOrigin((46.0, 38.0, 0.0))
    sitk.WriteImage(remodelling_image, str(remodelling))
    sed_image = sitk.GetImageFromArray(np.ones((3, 3, 3), dtype=np.float32))
    sed_image.SetSpacing((0.082, 0.082, 0.082))
    sed_image.SetOrigin((-566.0, -465.0, 0.0))
    sed_image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(sed_image, str(sed))
    case = TimelapseCase(
        subject_id="001",
        case_id="scene-row-01",
        baseline_image_path=remodelling,
        remodelling_image_path=remodelling,
        output_dir=tmp_path / "out",
        baseline_sed_path=sed,
        full_mask_path=None,
    )
    outputs = case_outputs(case, roi="full")
    sed_outputs = case_outputs(case)

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
        assert kwargs["baseline_strain"].GetSize() == remodelling_image.GetSize()
        assert kwargs["baseline_strain"].GetOrigin() == remodelling_image.GetOrigin()
        assert kwargs["baseline_strain"].GetDirection() == remodelling_image.GetDirection()
        assert np.count_nonzero(sitk.GetArrayFromImage(kwargs["baseline_strain"])) == 27
        outputs["curves"].write_bytes(b"plot")
        return FakeResult()

    monkeypatch.setattr("bonemechreg.post_timelapse.solve_sed_to_file", lambda **kwargs: sed)
    monkeypatch.setattr("bonemechreg.post_timelapse.mechanoregulation", fake_mechreg)

    summary = run_post_timelapse_case(case, profile="XtremeCTII", verbose=True)

    assert summary["processed"] == 1


def test_run_cases_reuses_existing_sed_when_summary_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, case = _make_case_fixture(tmp_path)
    outputs = case_outputs(case, roi="full")
    sed_outputs = case_outputs(case)
    sed_outputs["sed"].parent.mkdir(parents=True, exist_ok=True)
    _write_image(sed_outputs["sed"], value=3)
    called = {"solve": 0, "analyze": 0}

    def fake_solve(**kwargs):
        called["solve"] += 1
        return sed_outputs["sed"]

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
        assert kwargs["n_boot"] == 100
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


def test_verbose_run_reports_existing_sed_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, case = _make_case_fixture(tmp_path)
    outputs = case_outputs(case, roi="full")
    sed_outputs = case_outputs(case)
    sed_outputs["sed"].parent.mkdir(parents=True, exist_ok=True)
    _write_image(sed_outputs["sed"], value=3)

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

    def fail_solve(**kwargs):
        raise AssertionError("existing SED should be reused")

    def fake_mechreg(**kwargs):
        assert kwargs["n_boot"] == 100
        outputs["curves"].write_bytes(b"plot")
        return FakeResult()

    monkeypatch.setattr("bonemechreg.post_timelapse.solve_sed_to_file", fail_solve)
    monkeypatch.setattr("bonemechreg.post_timelapse.mechanoregulation", fake_mechreg)

    run_post_timelapse_mechanoregulation(dataset_root=root, profile="XtremeCTII", verbose=True)

    assert "reusing existing baseline SED" in capsys.readouterr().out


def test_run_cases_reuses_matched_fea_sed_without_copying_or_solving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, case = _make_case_fixture(tmp_path)
    external_sed = root / "derivatives" / "FEA" / "sub-001" / "ses-C1" / "xct" / "maps" / "sed.nii.gz"
    _write_image(external_sed, value=4)
    case = type(case)(
        **{
            **case.__dict__,
            "baseline_sed_path": external_sed,
        }
    )
    outputs = case_outputs(case, roi="full")
    sed_outputs = case_outputs(case)

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

    def fail_solve(**kwargs):
        raise AssertionError("matched FEA SED should be reused")

    def fake_mechreg(**kwargs):
        sed_values = sitk.GetArrayFromImage(kwargs["baseline_strain"])
        assert np.all(sed_values == 4)
        outputs["curves"].write_bytes(b"plot")
        return FakeResult()

    monkeypatch.setattr("bonemechreg.post_timelapse.solve_sed_to_file", fail_solve)
    monkeypatch.setattr("bonemechreg.post_timelapse.mechanoregulation", fake_mechreg)

    summary = run_post_timelapse_case(case, profile="XtremeCTII", verbose=True)

    assert summary["processed"] == 1
    assert "reusing matched FEA SED" in capsys.readouterr().out
    assert not sed_outputs["sed"].exists()
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
        assert kwargs["n_boot"] == 100
        outputs["curves"].write_bytes(b"plot")
        return FakeResult()

    monkeypatch.setattr("bonemechreg.post_timelapse.solve_sed_to_file", fake_solve)
    monkeypatch.setattr("bonemechreg.post_timelapse.mechanoregulation", fake_mechreg)

    summary = run_post_timelapse_mechanoregulation(dataset_root=root, profile="XtremeCTII")

    assert summary["processed"] == 1
    assert outputs["material"].exists()


def test_run_cases_analyzes_available_timelapse_rois_without_resolving_sed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    site = root / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "site-radius"
    remodelling = site / "analysis" / "visualize" / "sub-001_site-radius_comp-full_t0-T1_t1-T2_remodelling.nii.gz"
    baseline = site / "transformed_images" / "ses-T1" / "sub-001_site-radius_ses-T1_image_fused.nii.gz"
    stack = site / "ses-T1" / "stacks"
    segmentation = stack / "sub-001_site-radius_ses-T1_stack-01_seg.nii.gz"
    full = stack / "sub-001_site-radius_ses-T1_stack-01_mask-full.nii.gz"
    trab = stack / "sub-001_site-radius_ses-T1_stack-01_mask-trab.nii.gz"
    cort = stack / "sub-001_site-radius_ses-T1_stack-01_mask-cort.nii.gz"

    _write_array(remodelling, np.full((2, 2, 2), 2, dtype=np.uint8))
    _write_image(baseline, value=1)
    _write_array(segmentation, np.ones((2, 2, 2), dtype=np.uint8))
    _write_array(full, np.ones((2, 2, 2), dtype=np.uint8))
    trab_values = np.zeros((2, 2, 2), dtype=np.uint8)
    trab_values[:, :, 0] = 1
    cort_values = np.zeros((2, 2, 2), dtype=np.uint8)
    cort_values[:, :, 1] = 1
    _write_array(trab, trab_values)
    _write_array(cort, cort_values)

    case = discover_timelapse_cases(root)[0]
    outputs = case_outputs(case, roi="full")
    sed_outputs = case_outputs(case)
    sed_outputs["sed"].parent.mkdir(parents=True, exist_ok=True)
    _write_image(sed_outputs["sed"], value=3)
    analyzed_rois = []

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
        settings = {}
        plot_paths = {}

    def fail_solve(**kwargs):
        raise AssertionError("existing SED should be reused once for all ROIs")

    def fake_mechreg(**kwargs):
        roi_name = kwargs["run_name"].split("_roi-")[-1]
        analyzed_rois.append(roi_name)
        assert kwargs["surface_event_mapping"] == "surface_dilation_resorption_wins"
        assert kwargs["odds_model"] == "normalized_percent"
        return FakeResult()

    monkeypatch.setattr("bonemechreg.post_timelapse.solve_sed_to_file", fail_solve)
    monkeypatch.setattr("bonemechreg.post_timelapse.mechanoregulation", fake_mechreg)

    summary = run_post_timelapse_mechanoregulation(dataset_root=root, profile="XtremeCTI", reanalyze=True)

    assert analyzed_rois == ["full", "trab", "cort"]
    assert summary["processed"] == 1
    for roi in analyzed_rois:
        roi_outputs = case_outputs(case, roi=roi)
        assert roi_outputs["summary"].exists()
        assert roi_outputs["csv"].exists()
        payload = json.loads(roi_outputs["summary"].read_text(encoding="utf-8"))
        assert payload["roi"] == roi


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


def test_run_cases_filters_by_case_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, case = _make_case_fixture(tmp_path)
    called = {"case": None}

    def fake_run_case(case_arg, profile, overwrite, *, reanalyze=False, n_boot=100, bootstrap_sampling_perc=100.0, verbose=False):
        called["case"] = case_arg

    monkeypatch.setattr("bonemechreg.post_timelapse._run_case", fake_run_case)

    summary = run_post_timelapse_mechanoregulation(
        dataset_root=root,
        profile="XtremeCTII",
        case_id=case.case_id,
    )

    assert called["case"] == case
    assert summary["discovered"] == 1
    assert summary["processed"] == 1


def test_run_cases_missing_case_id_runs_no_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _case = _make_case_fixture(tmp_path)

    def fail_run_case(*args, **kwargs):
        raise AssertionError("no case should be run")

    monkeypatch.setattr("bonemechreg.post_timelapse._run_case", fail_run_case)

    summary = run_post_timelapse_mechanoregulation(
        dataset_root=root,
        profile="XtremeCTII",
        case_id="missing-case",
    )

    assert summary["discovered"] == 0
    assert summary["processed"] == 0


def test_run_post_timelapse_case_returns_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _root, case = _make_case_fixture(tmp_path)
    called = {"case": None}

    def fake_run_case(case_arg, profile, overwrite, *, reanalyze=False, n_boot=100, bootstrap_sampling_perc=100.0, verbose=False):
        called["case"] = case_arg
        assert profile == "XtremeCTII"
        assert overwrite is False
        assert reanalyze is False
        assert n_boot == 100
        assert bootstrap_sampling_perc == 100.0
        assert verbose is False

    monkeypatch.setattr("bonemechreg.post_timelapse._run_case", fake_run_case)

    summary = run_post_timelapse_case(case, profile="XtremeCTII")

    assert called["case"] == case
    assert summary["discovered"] == 1
    assert summary["processed"] == 1
    assert summary["skipped"] == 0
    assert summary["failed"] == 0
    assert summary["dry_run"] is False
    assert summary["case_id"] == case.case_id
    assert summary["output_dir"] == str(case.output_dir)


def test_run_post_timelapse_case_passes_configured_bootstrap_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, case = _make_case_fixture(tmp_path)

    def fake_run_case(case_arg, profile, overwrite, *, reanalyze=False, n_boot=100, bootstrap_sampling_perc=100.0, verbose=False):
        assert case_arg == case
        assert profile == "XtremeCTII"
        assert overwrite is False
        assert reanalyze is False
        assert n_boot == 37
        assert bootstrap_sampling_perc == 5.0
        assert verbose is False

    monkeypatch.setattr("bonemechreg.post_timelapse._run_case", fake_run_case)

    summary = run_post_timelapse_case(
        case,
        profile="XtremeCTII",
        n_boot=37,
        bootstrap_sampling_perc=5.0,
    )

    assert summary["processed"] == 1


def test_run_cases_skips_complete_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, case = _make_case_fixture(tmp_path)
    outputs = case_outputs(case, roi="full")
    sed_outputs = case_outputs(case)
    sed_outputs["sed"].parent.mkdir(parents=True, exist_ok=True)
    _write_image(sed_outputs["sed"], value=3)
    outputs["summary"].write_text(json.dumps({"ok": True}), encoding="utf-8")
    outputs["csv"].write_text("ok\n", encoding="utf-8")
    outputs["curves"].write_bytes(b"plot")
    outputs["schulte_curves"].write_bytes(b"plot")

    def fail_solve(**kwargs):
        raise AssertionError("solve should not run")

    monkeypatch.setattr("bonemechreg.post_timelapse.solve_sed_to_file", fail_solve)

    summary = run_post_timelapse_mechanoregulation(dataset_root=root, profile="XtremeCTII")

    assert summary["skipped"] == 1
    assert summary["processed"] == 0


def test_reanalyze_reruns_complete_case_without_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, case = _make_case_fixture(tmp_path)
    outputs = case_outputs(case, roi="full")
    sed_outputs = case_outputs(case)
    sed_outputs["sed"].parent.mkdir(parents=True, exist_ok=True)
    _write_image(sed_outputs["sed"], value=3)
    outputs["summary"].write_text(json.dumps({"ok": True}), encoding="utf-8")
    outputs["csv"].write_text("ok\n", encoding="utf-8")
    outputs["curves"].write_bytes(b"plot")
    outputs["schulte_curves"].write_bytes(b"plot")
    called = {"reanalyze": None}

    def fake_run_case(case_arg, profile, overwrite, *, reanalyze=False, n_boot=100, bootstrap_sampling_perc=100.0, verbose=False):
        assert case_arg == case
        assert overwrite is False
        called["reanalyze"] = reanalyze

    monkeypatch.setattr("bonemechreg.post_timelapse._run_case", fake_run_case)

    summary = run_post_timelapse_mechanoregulation(
        dataset_root=root,
        profile="XtremeCTII",
        reanalyze=True,
    )

    assert called["reanalyze"] is True
    assert summary["processed"] == 1
    assert summary["skipped"] == 0
