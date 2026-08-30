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
            "Timelapsed",
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
            "Timelapsed",
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
        "Timelapsed",
        root,
        {"name": "test", "version": "1"},
        records,
    )
    write_manifest(manifest, root / "derivatives" / "Timelapsed" / "manifest.json")

    cases = discover_timelapse_cases(root)

    assert len(cases) == 1
    assert cases[0].baseline_image_path == baseline
    assert cases[0].remodelling_image_path == remodelling
    assert cases[0].output_dir == (
        root / "derivatives" / "Mechanoregulation" / "sub-001" / "site-tibia" / "runs" / "remodelling"
    )
