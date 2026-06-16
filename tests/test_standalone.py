from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from bonemechreg.mechreg import MechanoregulationResult
from bonemechreg.standalone import (
    _load_or_reconstruct_strain,
    discover_standalone_inputs,
    run_standalone_analysis,
    write_summary_csv,
)


def _write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(np.asarray(array)), str(path))


def test_discover_standalone_inputs_from_advection_like_folder(tmp_path: Path) -> None:
    _write_image(tmp_path / "Sample_T0.mha", np.ones((3, 3, 3), dtype=np.float32))
    _write_image(tmp_path / "Sample_T1.mha", np.ones((3, 3, 3), dtype=np.float32))
    _write_image(tmp_path / "Sample_T0_to_T1_REMODELLING.mha", np.ones((3, 3, 3), dtype=np.uint8))
    _write_image(tmp_path / "Sample_sed.mha", np.ones((3, 3, 3), dtype=np.float32))

    inputs = discover_standalone_inputs(tmp_path)

    assert inputs.baseline_density.name == "Sample_T0.mha"
    assert inputs.followup_density is not None
    assert inputs.remodelling_image is not None
    assert inputs.baseline_strain is not None
    assert inputs.output_dir == tmp_path / "mechanoregulation"


def test_discover_standalone_inputs_prefers_nested_saved_baseline_sed(tmp_path: Path) -> None:
    _write_image(tmp_path / "Sample_T0.mha", np.ones((3, 3, 3), dtype=np.float32))
    _write_image(tmp_path / "Sample_T1.mha", np.ones((3, 3, 3), dtype=np.float32))
    _write_image(tmp_path / "Sample_T0_to_T1_REMODELLING.mha", np.ones((3, 3, 3), dtype=np.uint8))
    _write_image(tmp_path / "mechanoreg_model_event_output" / "baseline_sed_raw.nii.gz", np.ones((3, 3, 3), dtype=np.float32))

    inputs = discover_standalone_inputs(tmp_path)

    assert inputs.baseline_strain is not None
    assert inputs.baseline_strain.name == "baseline_sed_raw.nii.gz"


def test_standalone_sed_discovery_uses_exported_parosol_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "case"
    sibling = tmp_path / "case_parosol"
    shape = (3, 4, 5)
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    _write_image(run_dir / "Sample_T0.mha", np.ones(shape, dtype=np.float32))
    _write_image(run_dir / "Sample_T1.mha", np.ones(shape, dtype=np.float32))
    _write_image(run_dir / "Sample_T0_to_T1_REMODELLING.mha", np.ones(shape, dtype=np.uint8))
    _write_image(sibling / "interval_001_step_001" / "iteration_001" / "fields" / "sed.nii.gz", expected)

    inputs = discover_standalone_inputs(run_dir)
    strain = _load_or_reconstruct_strain(inputs, reference_shape=shape)

    np.testing.assert_allclose(strain, expected)


def test_write_summary_csv_contains_requested_metrics(tmp_path: Path) -> None:
    result = MechanoregulationResult(
        orf=1.5,
        orr=1.7,
        orf_ci=(1.2, 1.9),
        orr_ci=(1.1, 2.1),
        orr_increasing_strain=0.6,
        orr_decreasing_strain=1.7,
        orr_increasing_strain_ci=(0.4, 0.8),
        orr_decreasing_strain_ci=(1.1, 2.1),
        pvalue_form=0.01,
        pvalue_res=0.02,
        conditional_curves={"ccr": {"max": 0.73, "threshold_values": [20.0, 80.0]}},
        binned_odds_diagnostics={},
        sample_counts={
            "n_surface_voxels": 10,
            "n_sampled_voxels": 9,
            "n_formation": 2,
            "n_resorption": 3,
            "n_quiescence": 4,
        },
        settings={},
    )

    path = write_summary_csv(result, tmp_path / "summary.csv")

    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["CCR"] == "0.73"
    assert row["CCR_low_threshold"] == "20.0"
    assert row["CCR_high_threshold"] == "80.0"
    assert row["binned_lazy_zone_low"] == "20.0"
    assert row["binned_lazy_zone_high"] == "80.0"
    assert row["OR_F"] == "50.0"
    assert row["OR_R"] == "70.0"
    assert row["OR_F_ratio"] == "1.5"
    assert row["OR_R_ratio"] == "1.7"
    assert row["formation_odds_increase_percent_per_sed_percent"] == "50.0"
    assert row["resorption_odds_increase_percent_per_sed_percent_decrease"] == "70.0"


def test_run_standalone_analysis_writes_csv_png_json(tmp_path: Path, monkeypatch) -> None:
    shape = (4, 5, 6)
    _write_image(tmp_path / "Sample_T0.mha", np.ones(shape, dtype=np.float32))
    _write_image(tmp_path / "Sample_T1.mha", np.ones(shape, dtype=np.float32))
    _write_image(tmp_path / "Sample_T0_to_T1_REMODELLING.mha", np.ones(shape, dtype=np.uint8) * 2)
    _write_image(tmp_path / "Sample_sed.mha", np.ones(shape, dtype=np.float32))

    def fake_mechanoregulation(**kwargs):
        assert kwargs["cap_percentile"] == 99.0
        assert kwargs["legacy_clip_to_unit"] is False
        return MechanoregulationResult(
            orf=1.5,
            orr=1.7,
            orf_ci=(1.2, 1.9),
            orr_ci=(1.1, 2.1),
            orr_increasing_strain=0.6,
            orr_decreasing_strain=1.7,
            orr_increasing_strain_ci=(0.4, 0.8),
            orr_decreasing_strain_ci=(1.1, 2.1),
            pvalue_form=0.01,
            pvalue_res=0.02,
            conditional_curves={
                "strain": [0.5, 1.5, 2.5],
                "ccr": {"max": 0.73, "threshold_values": [0.5, 2.5]},
                "F": {"mean": [0.1, 0.2, 0.3]},
                "R": {"mean": [0.3, 0.2, 0.1]},
                "Q": {"mean": [0.6, 0.6, 0.6]},
                "schulte": {
                    "strain": [0.5, 1.5, 2.5],
                    "F": {"mean": [0.1, 0.2, 0.3]},
                    "R": {"mean": [0.3, 0.2, 0.1]},
                    "Q": {"mean": [0.6, 0.6, 0.6]},
                },
            },
            binned_odds_diagnostics={},
            sample_counts={"n_sampled_voxels": 9},
            settings={},
        )

    monkeypatch.setattr("bonemechreg.standalone.mechanoregulation", fake_mechanoregulation)
    outputs = run_standalone_analysis(tmp_path)

    assert outputs["csv"].exists()
    assert outputs["png"].exists()
    assert outputs["json"].exists()
