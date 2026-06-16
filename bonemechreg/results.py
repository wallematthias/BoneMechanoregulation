"""Summary serialization helpers.

The workflow writes both rich JSON, for reproducibility, and compact CSV, for
downstream tables. This module owns JSON payload construction for batch
TimelapsedHRpQCT runs and standalone analyses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bonemechreg.mechreg import MechanoregulationResult
from bonemechreg.timelapse import TimelapseCase


def _odds_increase_percent(odds_ratio: float) -> float:
    """Convert an odds ratio to percent odds increase."""
    return 100.0 * (float(odds_ratio) - 1.0)


def _finite_or_nan(value: Any) -> float:
    """Convert a scalar to float, returning NaN for invalid or non-finite input."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if out == out and out not in (float("inf"), float("-inf")) else float("nan")


def _summary_row(result: MechanoregulationResult) -> dict[str, Any]:
    """Build the compact tabular row shared by standalone and batch CSV output."""
    ccr = result.conditional_curves.get("ccr", {})
    thresholds = ccr.get("threshold_values", []) if isinstance(ccr, dict) else []
    logistic_lazy_zone = result.conditional_curves.get("logistic_lazy_zone", {})
    logistic_low = logistic_lazy_zone.get("low", float("nan")) if isinstance(logistic_lazy_zone, dict) else float("nan")
    logistic_high = logistic_lazy_zone.get("high", float("nan")) if isinstance(logistic_lazy_zone, dict) else float("nan")
    or_f_percent = _odds_increase_percent(result.orf)
    or_r_percent = _odds_increase_percent(result.orr)
    or_f_ci_percent = (
        _odds_increase_percent(result.orf_ci[0]),
        _odds_increase_percent(result.orf_ci[1]),
    )
    or_r_ci_percent = (
        _odds_increase_percent(result.orr_ci[0]),
        _odds_increase_percent(result.orr_ci[1]),
    )
    return {
        "CCR": _finite_or_nan(ccr.get("max", float("nan")) if isinstance(ccr, dict) else float("nan")),
        "CCR_low_threshold": _finite_or_nan(thresholds[0] if len(thresholds) > 0 else float("nan")),
        "CCR_high_threshold": _finite_or_nan(thresholds[1] if len(thresholds) > 1 else float("nan")),
        "binned_lazy_zone_low": _finite_or_nan(thresholds[0] if len(thresholds) > 0 else float("nan")),
        "binned_lazy_zone_high": _finite_or_nan(thresholds[1] if len(thresholds) > 1 else float("nan")),
        "logistic_lazy_zone_low": _finite_or_nan(logistic_low),
        "logistic_lazy_zone_high": _finite_or_nan(logistic_high),
        "OR_F": _finite_or_nan(or_f_percent),
        "OR_R": _finite_or_nan(or_r_percent),
        "OR_F_CI_low": _finite_or_nan(or_f_ci_percent[0]),
        "OR_F_CI_high": _finite_or_nan(or_f_ci_percent[1]),
        "OR_R_CI_low": _finite_or_nan(or_r_ci_percent[0]),
        "OR_R_CI_high": _finite_or_nan(or_r_ci_percent[1]),
        "OR_F_ratio": _finite_or_nan(result.orf),
        "OR_R_ratio": _finite_or_nan(result.orr),
        "OR_F_ratio_CI_low": _finite_or_nan(result.orf_ci[0]),
        "OR_F_ratio_CI_high": _finite_or_nan(result.orf_ci[1]),
        "OR_R_ratio_CI_low": _finite_or_nan(result.orr_ci[0]),
        "OR_R_ratio_CI_high": _finite_or_nan(result.orr_ci[1]),
        "formation_odds_increase_percent_per_sed_percent": or_f_percent,
        "formation_odds_increase_percent_CI_low": or_f_ci_percent[0],
        "formation_odds_increase_percent_CI_high": or_f_ci_percent[1],
        "resorption_odds_increase_percent_per_sed_percent_decrease": or_r_percent,
        "resorption_odds_increase_percent_CI_low": or_r_ci_percent[0],
        "resorption_odds_increase_percent_CI_high": or_r_ci_percent[1],
        "n_surface_voxels": int(result.sample_counts.get("n_surface_voxels", 0)),
        "n_sampled_voxels": int(result.sample_counts.get("n_sampled_voxels", 0)),
        "n_formation": int(result.sample_counts.get("n_formation", 0)),
        "n_resorption": int(result.sample_counts.get("n_resorption", 0)),
        "n_quiescence": int(result.sample_counts.get("n_quiescence", 0)),
        "n_cancelled_overlap": int(result.sample_counts.get("n_cancelled_overlap", 0)),
    }


def write_mechanoregulation_summary(
    *,
    case: TimelapseCase,
    profile: str,
    result: MechanoregulationResult,
    output_path: str | Path,
) -> Path:
    """Write a JSON summary for one TimelapsedHRpQCT pairwise case."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "subject_id": case.subject_id,
        "case_id": case.case_id,
        "profile": profile,
        "OR_F": _odds_increase_percent(result.orf),
        "OR_R": _odds_increase_percent(result.orr),
        "OR_F_CI": [
            _odds_increase_percent(result.orf_ci[0]),
            _odds_increase_percent(result.orf_ci[1]),
        ],
        "OR_R_CI": [
            _odds_increase_percent(result.orr_ci[0]),
            _odds_increase_percent(result.orr_ci[1]),
        ],
        "OR_F_ratio": float(result.orf),
        "OR_R_ratio": float(result.orr),
        "OR_R_increasing_strain": float(result.orr_increasing_strain),
        "OR_R_decreasing_strain": float(result.orr_decreasing_strain),
        "formation_odds_increase_percent_per_sed_percent": _odds_increase_percent(result.orf),
        "resorption_odds_increase_percent_per_sed_percent_decrease": _odds_increase_percent(result.orr),
        "orf_ci": [float(result.orf_ci[0]), float(result.orf_ci[1])],
        "orr_ci": [float(result.orr_ci[0]), float(result.orr_ci[1])],
        "formation_odds_increase_percent_ci": [
            _odds_increase_percent(result.orf_ci[0]),
            _odds_increase_percent(result.orf_ci[1]),
        ],
        "resorption_odds_increase_percent_ci": [
            _odds_increase_percent(result.orr_ci[0]),
            _odds_increase_percent(result.orr_ci[1]),
        ],
        "orr_increasing_strain_ci": [
            float(result.orr_increasing_strain_ci[0]),
            float(result.orr_increasing_strain_ci[1]),
        ],
        "orr_decreasing_strain_ci": [
            float(result.orr_decreasing_strain_ci[0]),
            float(result.orr_decreasing_strain_ci[1]),
        ],
        "pvalue_form": float(result.pvalue_form),
        "pvalue_res": float(result.pvalue_res),
        "conditional_curves": result.conditional_curves,
        "binned_odds_diagnostics": result.binned_odds_diagnostics,
        "sample_counts": result.sample_counts,
        "settings": result.settings,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_mechanoregulation_summary_payload(
    *,
    result: MechanoregulationResult,
    output_path: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a JSON summary when no TimelapsedHRpQCT case object is available."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "OR_F": _odds_increase_percent(result.orf),
        "OR_R": _odds_increase_percent(result.orr),
        "OR_F_CI": [
            _odds_increase_percent(result.orf_ci[0]),
            _odds_increase_percent(result.orf_ci[1]),
        ],
        "OR_R_CI": [
            _odds_increase_percent(result.orr_ci[0]),
            _odds_increase_percent(result.orr_ci[1]),
        ],
        "OR_F_ratio": float(result.orf),
        "OR_R_ratio": float(result.orr),
        "OR_R_increasing_strain": float(result.orr_increasing_strain),
        "OR_R_decreasing_strain": float(result.orr_decreasing_strain),
        "formation_odds_increase_percent_per_sed_percent": _odds_increase_percent(result.orf),
        "resorption_odds_increase_percent_per_sed_percent_decrease": _odds_increase_percent(result.orr),
        "orf_ci": [float(result.orf_ci[0]), float(result.orf_ci[1])],
        "orr_ci": [float(result.orr_ci[0]), float(result.orr_ci[1])],
        "formation_odds_increase_percent_ci": [
            _odds_increase_percent(result.orf_ci[0]),
            _odds_increase_percent(result.orf_ci[1]),
        ],
        "resorption_odds_increase_percent_ci": [
            _odds_increase_percent(result.orr_ci[0]),
            _odds_increase_percent(result.orr_ci[1]),
        ],
        "orr_increasing_strain_ci": [
            float(result.orr_increasing_strain_ci[0]),
            float(result.orr_increasing_strain_ci[1]),
        ],
        "orr_decreasing_strain_ci": [
            float(result.orr_decreasing_strain_ci[0]),
            float(result.orr_decreasing_strain_ci[1]),
        ],
        "pvalue_form": float(result.pvalue_form),
        "pvalue_res": float(result.pvalue_res),
        "conditional_curves": result.conditional_curves,
        "binned_odds_diagnostics": result.binned_odds_diagnostics,
        "sample_counts": result.sample_counts,
        "settings": result.settings,
    }
    if extra:
        payload.update(extra)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_mechanoregulation_summary_csv(result: MechanoregulationResult, output_path: str | Path) -> Path:
    """Write the compact one-row mechanoregulation CSV summary."""
    import csv

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = _summary_row(result)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return output_path


def format_workflow_summary(summary: dict[str, Any]) -> str:
    """Format batch workflow counts for concise CLI output."""
    keys = ("discovered", "processed", "skipped", "failed", "dry_run")
    return " ".join(f"{key}={summary.get(key)}" for key in keys)
