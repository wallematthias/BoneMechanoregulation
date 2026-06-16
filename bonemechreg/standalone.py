"""Standalone mechanoregulation analysis for an already-produced output folder.

The ``mechanoregulation analyze`` command uses this module. It discovers a
baseline image, a follow-up or remodelling label image, an optional mask, and a
baseline SED field from a single folder, then writes a compact CSV/JSON/PNG
summary. This path is useful for synthetic advection folders where the
timelapse derivative tree is not present.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from bonemechreg.mechreg import MechanoregulationResult, mechanoregulation
from bonemechreg.mechreg import (
    FORMATION_COLOR,
    LAZY_ZONE_COLOR,
    QUIESCENCE_COLOR,
    RESORPTION_COLOR,
)
from bonemechreg.results import write_mechanoregulation_summary_payload


@dataclass(frozen=True)
class StandaloneInputs:
    """Resolved files needed for one standalone analysis."""

    input_dir: Path
    baseline_density: Path
    followup_density: Path | None
    remodelling_image: Path | None
    baseline_strain: Path | None
    analysis_mask: Path | None
    output_dir: Path
    run_name: str


def discover_standalone_inputs(input_dir: str | Path) -> StandaloneInputs:
    """Discover baseline, remodelling, mask, and SED inputs in a loose folder.

    The discovery rules intentionally match the filenames produced by the
    advection simulations and by TimelapsedHRpQCT. The function does not load
    image data; it only resolves paths and raises actionable errors when the
    minimum required inputs cannot be found.
    """
    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    remodelling = _first(root, ("*REMODELLING.AIM", "*REMODELLING.mha", "*remodelling*.nii.gz", "*remodelling*.mha"))
    baseline = _first(root, ("*T0.AIM", "*_T0.nii.gz", "*_T0.mha", "*pairwise_t0*image*.nii.gz"))
    followup = _first(root, ("*T1.AIM", "*_T1.nii.gz", "*_T1.mha"))
    strain = _first(root, ("*sed*.nii.gz", "*sed*.mha", "*SED*.nii.gz", "*SED*.mha"))
    if strain is None:
        strain = _first_recursive(
            root,
            (
                "**/baseline_sed_scaled_balanced_thresholds.nii.gz",
                "**/baseline_sed_scaled*.nii.gz",
                "**/baseline_sed_raw.nii.gz",
                "**/baseline_sed*.nii.gz",
                "**/mechanoreg_model_event_parosol/fields/sed.nii.gz",
            ),
        )
    mask = _first(root, ("*TRAB*MASK*.AIM", "*COMPARTMENT_MASK*.AIM", "*mask*.nii.gz", "*MASK*.AIM"))

    if baseline is None:
        raise FileNotFoundError(f"Could not find a baseline image in {root}")
    if remodelling is None and followup is None:
        raise FileNotFoundError(f"Could not find a remodelling label image or follow-up density image in {root}")

    run_name = _run_name_from_path(remodelling or followup or baseline)
    return StandaloneInputs(
        input_dir=root,
        baseline_density=baseline,
        followup_density=followup,
        remodelling_image=remodelling,
        baseline_strain=strain,
        analysis_mask=mask,
        output_dir=root / "mechanoregulation",
        run_name=run_name,
    )


def run_standalone_analysis(input_dir: str | Path) -> dict[str, Path]:
    """Run a full standalone analysis and return paths to CSV, PNG, and JSON."""
    inputs = discover_standalone_inputs(input_dir)
    inputs.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_density = _read_scalar_xyz(inputs.baseline_density, density=True)
    followup_density = _read_scalar_xyz(inputs.followup_density, density=True) if inputs.followup_density is not None else None
    remodelling = _read_scalar_xyz(inputs.remodelling_image, density=False) if inputs.remodelling_image is not None else None
    mask = _read_scalar_xyz(inputs.analysis_mask, density=False) if inputs.analysis_mask is not None else None
    baseline_strain = _load_or_reconstruct_strain(inputs, reference_shape=baseline_density.shape)

    result = mechanoregulation(
        remodelling_image=remodelling,
        baseline_density=baseline_density if remodelling is None else None,
        followup_density=followup_density if remodelling is None else None,
        baseline_strain=baseline_strain,
        analysis_mask=mask,
        run_name=inputs.run_name,
        work_dir=inputs.output_dir,
        n_boot=1000,
        seed=0,
        cap_percentile=99.0,
        resorption_or_definition="decreasing_strain",
        legacy_clip_to_unit=False,
        plot=False,
        return_full=True,
    )
    assert isinstance(result, MechanoregulationResult)

    csv_path = inputs.output_dir / f"{inputs.run_name}_mechanoregulation_summary.csv"
    png_path = inputs.output_dir / f"{inputs.run_name}_mechanoregulation_curves.png"
    json_path = inputs.output_dir / f"{inputs.run_name}_mechanoregulation_summary.json"
    write_summary_csv(result, csv_path)
    plot_mechanoregulation_summary(result, png_path)
    write_mechanoregulation_summary_payload(
        result=result,
        output_path=json_path,
        extra={
            "input_dir": str(inputs.input_dir),
            "baseline_density": str(inputs.baseline_density),
            "followup_density": None if inputs.followup_density is None else str(inputs.followup_density),
            "remodelling_image": None if inputs.remodelling_image is None else str(inputs.remodelling_image),
            "baseline_strain": None if inputs.baseline_strain is None else str(inputs.baseline_strain),
            "analysis_mask": None if inputs.analysis_mask is None else str(inputs.analysis_mask),
        },
    )
    return {"csv": csv_path, "png": png_path, "json": json_path}


def write_summary_csv(result: MechanoregulationResult, output_path: str | Path) -> Path:
    """Write the release-facing one-row summary CSV.

    The CSV contains the values users typically copy into downstream tables:
    CCR and lazy-zone thresholds, OR_F/OR_R with confidence intervals, percent
    odds increases, and the final F/R/Q sample counts after surface projection.
    """
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ccr = result.conditional_curves.get("ccr", {})
    thresholds = ccr.get("threshold_values", []) if isinstance(ccr, dict) else []
    logistic_lazy_zone = _lazy_zone_thresholds(result)
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
    row = {
        "CCR": _finite_or_nan(ccr.get("max", np.nan) if isinstance(ccr, dict) else np.nan),
        "CCR_low_threshold": _finite_or_nan(thresholds[0] if len(thresholds) > 0 else np.nan),
        "CCR_high_threshold": _finite_or_nan(thresholds[1] if len(thresholds) > 1 else np.nan),
        "binned_lazy_zone_low": _finite_or_nan(thresholds[0] if len(thresholds) > 0 else np.nan),
        "binned_lazy_zone_high": _finite_or_nan(thresholds[1] if len(thresholds) > 1 else np.nan),
        "logistic_lazy_zone_low": _finite_or_nan(logistic_lazy_zone[0] if len(logistic_lazy_zone) > 0 else np.nan),
        "logistic_lazy_zone_high": _finite_or_nan(logistic_lazy_zone[1] if len(logistic_lazy_zone) > 1 else np.nan),
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
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return output


def plot_mechanoregulation_summary(result: MechanoregulationResult, output_path: str | Path) -> Path:
    """Write the two-panel mechanoregulation summary figure.

    Left: jagged Schulte-style conditional probability curves with a strongly
    smoothed overlay. Right: bootstrap-median logistic probability curves for
    R/Q/F. Both panels show the logistic lazy zone as a gray band.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "font.family": "Arial",
        }
    )

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    schulte = result.conditional_curves.get("schulte", {})
    x = np.asarray(schulte.get("strain", []), dtype=np.float64)
    if x.size == 0:
        x = np.asarray(result.conditional_curves.get("strain", []), dtype=np.float64)
        schulte = result.conditional_curves
    ccr = result.conditional_curves.get("ccr", {})
    thresholds = ccr.get("threshold_values", []) if isinstance(ccr, dict) else []
    logistic_lazy_zone = _lazy_zone_thresholds(result)
    ccr_value = _finite_or_nan(ccr.get("max", np.nan) if isinstance(ccr, dict) else np.nan)

    fig, (ax_prob, ax_or) = plt.subplots(
        1,
        2,
        figsize=(8.2, 3.4),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
        constrained_layout=True,
    )
    _plot_class_curve(ax_prob, x, schulte, "R", RESORPTION_COLOR, "Resorption", smooth_sigma=3.0)
    _plot_class_curve(ax_prob, x, schulte, "Q", QUIESCENCE_COLOR, "Quiescence", smooth_sigma=3.0, alpha=0.55)
    _plot_class_curve(ax_prob, x, schulte, "F", FORMATION_COLOR, "Formation", smooth_sigma=3.0)
    _draw_lazy_zone(ax_prob, list(thresholds))
    ax_prob.set_xlabel("Normalized SED (%)")
    ax_prob.set_ylabel("Conditional probability")
    if np.isfinite(ccr_value):
        ax_prob.set_title(f"Schulte binned curves (CCR={ccr_value:.3g})", fontsize=9)
    ax_prob.set_xlim(0, 100)
    ax_prob.set_ylim(0, 1)
    ax_prob.legend(frameon=False, fontsize=7, loc="best")

    support = np.asarray(result.conditional_curves.get("strain", []), dtype=np.float64)
    pf_curve = result.conditional_curves.get("F", {})
    pr_curve = result.conditional_curves.get("R", {})
    _plot_logistic_probability(ax_or, support, pr_curve, RESORPTION_COLOR, "P(R)")
    _plot_remaining_probability(ax_or, support, pr_curve, pf_curve)
    _plot_logistic_probability(ax_or, support, pf_curve, FORMATION_COLOR, "P(F)")
    _draw_lazy_zone(ax_or, logistic_lazy_zone)
    ax_or.set_xlabel("Normalized SED (%)")
    ax_or.set_ylabel("Predicted probability")
    ax_or.set_title("Parametric logistic diagnostic", fontsize=9)
    ax_or.set_xlim(0, 100)
    ax_or.set_ylim(0, 1)
    ax_or.legend(frameon=False, fontsize=7, loc="best")

    or_f_percent = _odds_increase_percent(result.orf)
    or_r_percent = _odds_increase_percent(result.orr)
    title = f"OR_F={or_f_percent:.3g}%, OR_R={or_r_percent:.3g}%"
    fig.suptitle(title, fontsize=9)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def _plot_logistic_probability(
    ax: Any,
    x: np.ndarray,
    curve: dict[str, Any],
    color: str,
    label: str,
) -> None:
    """Plot one logistic probability curve with optional bootstrap band."""
    values = np.asarray(curve.get("mean", []), dtype=np.float64)
    if x.size != values.size:
        return
    median = np.asarray(curve.get("median", []), dtype=np.float64)
    low = np.asarray(curve.get("low", []), dtype=np.float64)
    high = np.asarray(curve.get("high", []), dtype=np.float64)
    if low.size == x.size and high.size == x.size:
        ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
    if median.size == x.size:
        ax.plot(x, median, color=color, linewidth=2.0, label=label)
    else:
        ax.plot(x, values, color=color, linewidth=2.0, label=label)


def _plot_remaining_probability(
    ax: Any,
    x: np.ndarray,
    resorption_curve: dict[str, Any],
    formation_curve: dict[str, Any],
) -> None:
    """Plot quiescence as the remaining probability between R and F."""
    pr = np.asarray(resorption_curve.get("median", resorption_curve.get("mean", [])), dtype=np.float64)
    pf = np.asarray(formation_curve.get("median", formation_curve.get("mean", [])), dtype=np.float64)
    if x.size != pr.size or x.size != pf.size:
        return
    pq = np.clip(1.0 - pr - pf, 0.0, 1.0)
    ax.plot(x, pq, color=QUIESCENCE_COLOR, linewidth=1.8, alpha=0.8, label="P(Q)")
    ax.fill_between(x, pr, pr + pq, color=QUIESCENCE_COLOR, alpha=0.08, linewidth=0)


def _lazy_zone_thresholds(result: MechanoregulationResult) -> list[float]:
    """Estimate logistic lazy-zone bounds from R/Q and F/Q curve crossings."""
    x = np.asarray(result.conditional_curves.get("strain", []), dtype=np.float64)
    if x.size == 0:
        return []
    pr_curve = result.conditional_curves.get("R", {})
    pf_curve = result.conditional_curves.get("F", {})
    pr = np.asarray(pr_curve.get("median", pr_curve.get("mean", [])), dtype=np.float64)
    pf = np.asarray(pf_curve.get("median", pf_curve.get("mean", [])), dtype=np.float64)
    if pr.size != x.size or pf.size != x.size:
        return []
    pq = np.clip(1.0 - pr - pf, 0.0, 1.0)
    low = _first_crossing(x, pr - pq, prefer="falling")
    high = _first_crossing(x, pf - pq, prefer="rising")
    out: list[float] = []
    if np.isfinite(low):
        out.append(float(low))
    if np.isfinite(high):
        out.append(float(high))
    return out


def _first_crossing(x: np.ndarray, y: np.ndarray, *, prefer: str) -> float:
    """Return the first linear-interpolated zero crossing in a curve."""
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return float("nan")
    for i in range(x.size - 1):
        y0 = float(y[i])
        y1 = float(y[i + 1])
        if prefer == "falling" and not (y0 >= 0.0 and y1 <= 0.0):
            continue
        if prefer == "rising" and not (y0 <= 0.0 and y1 >= 0.0):
            continue
        if y1 == y0:
            return float(x[i])
        frac = -y0 / (y1 - y0)
        return float(x[i] + frac * (x[i + 1] - x[i]))
    return float("nan")


def _plot_class_curve(
    ax: Any,
    x: np.ndarray,
    curves: dict[str, Any],
    key: str,
    color: str,
    label: str,
    *,
    smooth_sigma: float,
    alpha: float = 0.85,
) -> None:
    """Plot the raw binned Schulte curve and a smoothed visual guide."""
    from scipy.ndimage import gaussian_filter1d

    values = np.asarray(curves.get(key, {}).get("mean", []), dtype=np.float64)
    if x.size != values.size:
        return
    ax.plot(x, values, color=color, linewidth=0.7, alpha=0.35 * alpha)
    smooth = gaussian_filter1d(values, sigma=float(smooth_sigma), mode="nearest")
    ax.plot(x, smooth, color=color, linewidth=2.0, alpha=alpha, label=label)


def _draw_lazy_zone(ax: Any, thresholds: list[float]) -> None:
    """Draw lazy-zone shading and threshold lines on a Matplotlib axis."""
    if len(thresholds) >= 2:
        low = float(thresholds[0])
        high = float(thresholds[1])
        ax.axvspan(low, high, color=LAZY_ZONE_COLOR, alpha=0.28, linewidth=0, label="Lazy zone")
    labels = ("Lazy-zone low", "Lazy-zone high")
    colors = (RESORPTION_COLOR, FORMATION_COLOR)
    for idx, threshold in enumerate(thresholds[:2]):
        ax.axvline(float(threshold), color=colors[idx], linestyle="--", linewidth=0.9, alpha=0.65, label=labels[idx])


def _load_or_reconstruct_strain(inputs: StandaloneInputs, *, reference_shape: tuple[int, ...]) -> np.ndarray:
    """Load baseline SED from file or from an exported ParOSol field folder."""
    if inputs.baseline_strain is not None:
        strain = _read_scalar_xyz(inputs.baseline_strain, density=True)
        if strain.shape != reference_shape:
            raise ValueError("baseline strain shape does not match baseline image")
        return strain.astype(np.float32, copy=False)

    sed_path = _find_parosol_sed_field(inputs.input_dir)
    if sed_path is None:
        raise FileNotFoundError(
            "No baseline SED image found and no exported ParOSol fields/sed.nii.gz could be discovered."
        )
    strain = _read_scalar_xyz(sed_path, density=True)
    if strain.shape != reference_shape:
        raise ValueError(f"exported SED shape {strain.shape} does not match baseline image shape {reference_shape}")
    return strain.astype(np.float32, copy=False)


def _find_parosol_sed_field(input_dir: Path) -> Path | None:
    """Find a ParOSol ``fields/sed.nii.gz`` near a synthetic run folder."""
    sibling = input_dir.with_name(input_dir.name + "_parosol")
    roots = [sibling, input_dir]
    for root in roots:
        if not root.exists():
            continue
        for candidate in sorted(root.glob("interval_001_step_001/**/fields/sed.nii.gz")):
            return candidate
        for candidate in sorted(root.glob("**/fields/sed.nii.gz")):
            return candidate
    return None


def _read_scalar_xyz(path: Path | None, *, density: bool) -> np.ndarray:
    """Read AIM or SimpleITK-compatible scalar image data as a NumPy array."""
    if path is None:
        raise ValueError("path is required")
    path = Path(path).expanduser().resolve()
    if _is_aim(path):
        try:
            import py_aimio
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("AIM input requires py_aimio") from exc
        array, _meta = py_aimio.read_aim(str(path), density=bool(density))
        return np.asarray(array)
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)


def _first(root: Path, patterns: tuple[str, ...]) -> Path | None:
    """Return the first direct child file matching any glob pattern."""
    for pattern in patterns:
        matches = [p for p in sorted(root.glob(pattern)) if p.is_file()]
        if matches:
            return matches[0]
    return None


def _first_recursive(root: Path, patterns: tuple[str, ...]) -> Path | None:
    """Return the first recursively matched file for any glob pattern."""
    for pattern in patterns:
        matches = [p for p in sorted(root.glob(pattern)) if p.is_file()]
        if matches:
            return matches[0]
    return None


def _is_aim(path: Path) -> bool:
    """Return true for native Scanco AIM names, including ``.AIM;1`` files."""
    name = path.name.lower()
    return name.endswith(".aim") or ".aim;" in name


def _run_name_from_path(path: Path) -> str:
    """Create a stable output stem by stripping common image suffixes."""
    name = path.name
    for suffix in (".nii.gz", ".mha", ".AIM", ".aim"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _finite_or_nan(value: Any) -> float:
    """Convert a scalar to float, returning NaN for invalid or non-finite input."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _odds_increase_percent(odds_ratio: Any) -> float:
    """Convert an odds ratio into percent odds increase."""
    value = _finite_or_nan(odds_ratio)
    if not np.isfinite(value):
        return float("nan")
    return 100.0 * (value - 1.0)
