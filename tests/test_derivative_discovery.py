from __future__ import annotations

from pathlib import Path

from bone_imaging_derivatives import DerivativeManifest, DerivativeRecord, write_manifest

from bonemechreg.timelapse import discover_timelapse_cases


def test_discover_cases_prefers_timelapsed_manifest_records(tmp_path: Path) -> None:
    """Ignoring a Timelapsed manifest would force new datasets back into the legacy folder parser."""
    root = tmp_path / "dataset"
    remodelling = root / "source" / "remodelling.nii.gz"
    baseline = root / "source" / "baseline.nii.gz"
    records = (
        DerivativeRecord(
            "Timelapse",
            "remodelling_pairwise_table",
            "001",
            "tibia",
            "C1",
            None,
            "native",
            remodelling,
            "generated",
            inputs=("baseline-image",),
        ),
        DerivativeRecord(
            "Timelapse",
            "transformed_image",
            "001",
            "tibia",
            "C1",
            None,
            "native",
            baseline,
            "generated",
            record_id="baseline-image",
        ),
    )
    manifest = DerivativeManifest.create(
        "Timelapse",
        root,
        {"name": "test", "version": "1"},
        records,
    )
    write_manifest(manifest, root / "derivatives" / "Timelapse" / "manifest.json")

    cases = discover_timelapse_cases(root)

    assert len(cases) == 1
    assert cases[0].baseline_image_path == baseline
    assert cases[0].remodelling_image_path == remodelling
    assert cases[0].output_dir == (
        root / "derivatives" / "Mechanoregulation" / "sub-001" / "xct" / "runs" / "remodelling"
    )


def test_discover_current_timelapse_remodelling_matches_profile_specific_fea_sed(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    remodelling = (
        root
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "xct"
        / "analysis"
        / "visualize"
        / "sub-001_voi-radiusleft_desc-roi_union_t0-001_t1-002_thr-225p0_cluster-5_remodelling.nii.gz"
    )
    baseline = (
        root
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "transformed"
        / "sub-001_ses-001_voi-radiusleft_image-fused.nii.gz"
    )
    sed = (
        root
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "maps"
        / "sub-001_ses-001_voi-radiusleft_desc-XtremeCTII_map-sed.nii.gz"
    )
    for path in (remodelling, baseline, sed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    cases = discover_timelapse_cases(root, sed_profile="XtremeCTII")

    assert len(cases) == 1
    assert cases[0].subject_id == "001"
    assert cases[0].site == "radiusleft"
    assert cases[0].baseline_session_id == "001"
    assert cases[0].followup_session_id == "002"
    assert cases[0].baseline_image_path == baseline
    assert cases[0].baseline_sed_path == sed
    assert cases[0].output_dir == (
        root / "derivatives" / "Mechanoregulation" / "sub-001" / "xct" / "runs" / cases[0].case_id
    )


def test_discover_current_timelapse_does_not_cross_match_fea_sed_profiles(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    remodelling = (
        root
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "xct"
        / "analysis"
        / "visualize"
        / "sub-001_voi-radiusleft_desc-roi_union_t0-001_t1-002_thr-225p0_cluster-5_remodelling.nii.gz"
    )
    baseline = (
        root
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "transformed"
        / "sub-001_ses-001_voi-radiusleft_image-fused.nii.gz"
    )
    xtreme_i_sed = (
        root
        / "derivatives"
        / "FEA"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "maps"
        / "sub-001_ses-001_voi-radiusleft_desc-XtremeCTI_map-sed.nii.gz"
    )
    for path in (remodelling, baseline, xtreme_i_sed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    cases = discover_timelapse_cases(root, sed_profile="XtremeCTII")

    assert len(cases) == 1
    assert cases[0].baseline_sed_path is None
