from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from bonemechreg.timelapse import available_case_rois, case_outputs, discover_timelapse_cases


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.float32))
    sitk.WriteImage(image, str(path))


def test_discover_cases_returns_pairwise_t0_case(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    remodelling_path = (
        root
        / "derivatives"
        / "TimelapsedHRpQCT"
        / "sub-001"
        / "analysis"
        / "pairwise_t0"
        / "ses-C1_ses-C2"
        / "sub-001_ses-C1_ses-C2_remodelling.nii.gz"
    )
    baseline_path = (
        root
        / "derivatives"
        / "TimelapsedHRpQCT"
        / "sub-001"
        / "analysis"
        / "pairwise_t0"
        / "ses-C1_ses-C2"
        / "sub-001_ses-C1_pairwise_t0_image.nii.gz"
    )
    _write_image(remodelling_path)
    _write_image(baseline_path)

    cases = discover_timelapse_cases(root)

    assert len(cases) == 1
    assert cases[0].subject_id == "001"
    assert cases[0].baseline_image_path == baseline_path
    assert cases[0].remodelling_image_path == remodelling_path


def test_discover_cases_accepts_direct_timelapsed_derivative_root(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    derivative_root = root / "derivatives" / "TimelapsedHRpQCT"
    remodelling_path = (
        derivative_root
        / "sub-001"
        / "analysis"
        / "pairwise_t0"
        / "ses-C1_ses-C2"
        / "sub-001_ses-C1_ses-C2_remodelling.nii.gz"
    )
    baseline_path = remodelling_path.with_name("sub-001_ses-C1_pairwise_t0_image.nii.gz")
    _write_image(remodelling_path)
    _write_image(baseline_path)

    cases = discover_timelapse_cases(derivative_root)

    assert len(cases) == 1
    assert cases[0].subject_id == "001"
    assert cases[0].remodelling_image_path == remodelling_path


def test_case_outputs_use_expected_suffixes(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    remodelling_path = root / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "analysis" / "pairwise_t0" / "ses-C1_ses-C2" / "sub-001_ses-C1_ses-C2_remodelling.nii.gz"
    baseline_path = remodelling_path.with_name("sub-001_ses-C1_pairwise_t0_image.nii.gz")
    _write_image(remodelling_path)
    _write_image(baseline_path)

    case = discover_timelapse_cases(root)[0]
    outputs = case_outputs(case)

    assert outputs["sed"].name.endswith("_sed.nii.gz")
    assert outputs["summary"].name.endswith("_mechanoregulation_summary.json")
    assert outputs["csv"].name.endswith("_mechanoregulation_summary.csv")
    assert outputs["curves"].name.endswith("_conditional_curves.png")


def test_case_outputs_can_be_roi_scoped_without_changing_solver_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    remodelling_path = root / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "analysis" / "pairwise_t0" / "ses-C1_ses-C2" / "sub-001_ses-C1_ses-C2_remodelling.nii.gz"
    baseline_path = remodelling_path.with_name("sub-001_ses-C1_pairwise_t0_image.nii.gz")
    _write_image(remodelling_path)
    _write_image(baseline_path)

    case = discover_timelapse_cases(root)[0]
    base_outputs = case_outputs(case)
    trab_outputs = case_outputs(case, roi="trab")

    assert available_case_rois(case) == {"full": None}
    assert trab_outputs["sed"] == base_outputs["sed"]
    assert trab_outputs["material"] == base_outputs["material"]
    assert trab_outputs["summary"].name.endswith("_roi-trab_mechanoregulation_summary.json")
    assert trab_outputs["csv"].name.endswith("_roi-trab_mechanoregulation_summary.csv")
    assert trab_outputs["curves"].name.endswith("_roi-trab_conditional_curves.png")
    assert trab_outputs["schulte_curves"].name.endswith("_roi-trab_schulte_binned_curves.png")


def test_discover_cases_supports_current_timelapsed_visualize_layout(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    remodelling_path = (
        root
        / "derivatives"
        / "TimelapsedHRpQCT"
        / "sub-SAMPLE355"
        / "site-tibia"
        / "analysis"
        / "visualize"
        / "sub-SAMPLE355_site-tibia_comp-full_t0-T1_t1-T2_thr-225p0_cluster-12_remodelling.nii.gz"
    )
    baseline_path = (
        root
        / "derivatives"
        / "TimelapsedHRpQCT"
        / "sub-SAMPLE355"
        / "site-tibia"
        / "transformed_images"
        / "ses-T1"
        / "sub-SAMPLE355_site-tibia_ses-T1_image_fused.nii.gz"
    )
    _write_image(remodelling_path)
    _write_image(baseline_path)

    cases = discover_timelapse_cases(root)

    assert len(cases) == 1
    assert cases[0].subject_id == "SAMPLE355"
    assert cases[0].baseline_image_path == baseline_path
    assert cases[0].remodelling_image_path == remodelling_path
    assert cases[0].output_dir == remodelling_path.parents[2] / "mechanoregulation"


def test_discover_cases_accepts_selected_current_subject_root(tmp_path: Path) -> None:
    subject_root = tmp_path / "TimelapsedHRpQCT" / "sub-SAMPLE355"
    site = subject_root / "site-tibia"
    remodelling_path = (
        site
        / "analysis"
        / "visualize"
        / "sub-SAMPLE355_site-tibia_comp-full_t0-T1_t1-T2_thr-225p0_cluster-12_remodelling.nii.gz"
    )
    baseline_path = site / "transformed_images" / "ses-T1" / "sub-SAMPLE355_site-tibia_ses-T1_image_fused.nii.gz"
    _write_image(remodelling_path)
    _write_image(baseline_path)

    cases = discover_timelapse_cases(subject_root)

    assert len(cases) == 1
    assert cases[0].subject_id == "SAMPLE355"
    assert cases[0].baseline_image_path == baseline_path
    assert cases[0].remodelling_image_path == remodelling_path


def test_discover_current_timelapse_prefers_common_region_roi_masks(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    remodelling_path = (
        root
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "xct"
        / "analysis"
        / "visualize"
        / "sub-001_voi-radiusleft_desc-roi_union_t0-001_t1-002_thr-225p0_cluster-12_remodelling.nii.gz"
    )
    baseline_path = (
        root
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "transformed"
        / "sub-001_ses-001_voi-radiusleft_image-fused.nii.gz"
    )
    transformed_full = baseline_path.with_name("sub-001_ses-001_voi-radiusleft_desc-full_mask-fused.nii.gz")
    transformed_trab = baseline_path.with_name("sub-001_ses-001_voi-radiusleft_desc-trab_mask-fused.nii.gz")
    transformed_cort = baseline_path.with_name("sub-001_ses-001_voi-radiusleft_desc-cort_mask-fused.nii.gz")
    common_full = (
        root
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "xct"
        / "analysis"
        / "common_regions"
        / "sub-001_voi-radiusleft_desc-full_common-alltimepoints.nii.gz"
    )
    common_trab = common_full.with_name("sub-001_voi-radiusleft_desc-trab_common-alltimepoints.nii.gz")
    common_cort = common_full.with_name("sub-001_voi-radiusleft_desc-cort_common-alltimepoints.nii.gz")
    for path in (
        remodelling_path,
        baseline_path,
        transformed_full,
        transformed_trab,
        transformed_cort,
        common_full,
        common_trab,
        common_cort,
    ):
        _write_image(path)

    case = discover_timelapse_cases(root)[0]

    assert case.full_mask_path == common_full
    assert case.trab_mask_path == common_trab
    assert case.cort_mask_path == common_cort
    assert available_case_rois(case) == {
        "full": common_full,
        "trab": common_trab,
        "cort": common_cort,
    }


def test_discover_cases_accepts_selected_current_site_root(tmp_path: Path) -> None:
    site = tmp_path / "TimelapsedHRpQCT" / "sub-SAMPLE355" / "site-tibia"
    remodelling_path = (
        site
        / "analysis"
        / "visualize"
        / "sub-SAMPLE355_site-tibia_comp-full_t0-T1_t1-T2_thr-225p0_cluster-12_remodelling.nii.gz"
    )
    baseline_path = site / "transformed_images" / "ses-T1" / "sub-SAMPLE355_site-tibia_ses-T1_image_fused.nii.gz"
    _write_image(remodelling_path)
    _write_image(baseline_path)

    cases = discover_timelapse_cases(site)

    assert len(cases) == 1
    assert cases[0].subject_id == "SAMPLE355"
    assert cases[0].baseline_image_path == baseline_path
    assert cases[0].remodelling_image_path == remodelling_path
