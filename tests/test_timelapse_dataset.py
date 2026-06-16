from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from bonemechreg.timelapse import case_outputs, discover_timelapse_cases


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
    assert cases[0].subject_id == "sub-001"
    assert cases[0].baseline_image_path == baseline_path
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
    assert cases[0].subject_id == "sub-SAMPLE355"
    assert cases[0].baseline_image_path == baseline_path
    assert cases[0].remodelling_image_path == remodelling_path
    assert cases[0].output_dir == remodelling_path.parents[2] / "mechanoregulation"
