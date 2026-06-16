"""Core mechanoregulation analysis.

This module contains the actual Schulte-style mechanoregulation math used by
the package. The public entry point is :func:`mechanoregulation`; the helper
functions are intentionally kept in this file so the full analysis can be read
from top to bottom:

1. derive or load a remodelling label image;
2. project formation and resorption events symmetrically onto baseline surface;
3. normalize baseline SED into 0-100 percent strain bins;
4. compute conditional probability curves, CCR, and logistic odds ratios.

Arrays inside this module use ``xyz`` order. SimpleITK images are converted
from their native ``zyx`` NumPy representation at the input boundary.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

RESORPTION_COLOR = "#6a3d9a"
FORMATION_COLOR = "#f28e2b"
QUIESCENCE_COLOR = "#8c8c8c"
LAZY_ZONE_COLOR = "#d0d0d0"


@dataclass
class MechanoregulationResult:
    """All numerical and plotting outputs from one mechanoregulation run.

    Attributes:
        orf: Formation odds ratio per one normalized SED percentage point.
        orr: Resorption odds ratio. By default this is expressed per one
            normalized SED percentage point decrease, so larger values mean
            resorption is enriched in lower-strain regions.
        orf_ci: Bootstrap confidence interval for ``orf``.
        orr_ci: Bootstrap confidence interval for ``orr``.
        orr_increasing_strain: Resorption odds ratio if interpreted with
            increasing strain. This is retained for transparency but is not the
            default reported OR_R.
        orr_decreasing_strain: Resorption odds ratio with decreasing strain,
            matching the interpretation used in the methods text.
        pvalue_form: Mean bootstrap p-value for formation slope.
        pvalue_res: Mean bootstrap p-value for resorption slope.
        conditional_curves: Logistic and binned conditional probability curves.
        binned_odds_diagnostics: Bin-wise class enrichment diagnostics.
        sample_counts: Counts after symmetric surface projection.
        settings: Analysis parameters written to JSON output.
        plot_paths: Optional paths written by the internal plotting helper.
    """

    orf: float
    orr: float
    orf_ci: tuple[float, float]
    orr_ci: tuple[float, float]
    orr_increasing_strain: float
    orr_decreasing_strain: float
    orr_increasing_strain_ci: tuple[float, float]
    orr_decreasing_strain_ci: tuple[float, float]
    pvalue_form: float
    pvalue_res: float
    conditional_curves: dict[str, Any]
    binned_odds_diagnostics: dict[str, Any]
    sample_counts: dict[str, int]
    settings: dict[str, Any]
    plot_paths: dict[str, Path] | None = None


def _as_numpy_xyz(image: Any, array_order: str) -> np.ndarray:
    """Convert a 3D image-like object into the package's internal ``xyz`` order.

    SimpleITK returns arrays in ``zyx`` order, whereas most algorithmic code in
    this package is easier to reason about in ``xyz`` order. This helper is the
    single boundary where that convention is enforced.
    """
    if isinstance(image, sitk.Image):
        return np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"input image must be 3D, got shape {arr.shape}")
    order = str(array_order).strip().lower()
    if order == "xyz":
        return arr
    if order == "zyx":
        return np.transpose(arr, (2, 1, 0))
    raise ValueError("array_order must be 'xyz' or 'zyx'")


def _gaussian_filter_xyz(array: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a Gaussian filter to an ``xyz`` array using reflective boundaries."""
    if float(sigma) <= 0.0:
        return np.asarray(array, dtype=np.float64)
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise RuntimeError("scipy is required for gaussian remodelling-site derivation") from exc
    return gaussian_filter(np.asarray(array, dtype=np.float64), sigma=float(sigma), mode="reflect")


def _remove_small_components(mask: np.ndarray, *, min_size: int) -> np.ndarray:
    """Remove small 6-connected binary components from an event mask.

    Timelapsed-style remodelling definitions reject tiny isolated components so
    that single noisy voxels are not counted as biological formation or
    resorption. Connectivity is deliberately simple 6-neighbour connectivity to
    match voxel-face contact.
    """
    binary = np.asarray(mask, dtype=bool)
    if int(min_size) <= 1 or not np.any(binary):
        return binary
    visited = np.zeros(binary.shape, dtype=bool)
    keep = np.zeros(binary.shape, dtype=bool)
    shape = binary.shape
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    seeds = np.argwhere(binary)
    for seed_arr in seeds:
        seed = tuple(int(v) for v in seed_arr)
        if visited[seed]:
            continue
        component: list[tuple[int, int, int]] = []
        queue: deque[tuple[int, int, int]] = deque([seed])
        visited[seed] = True
        while queue:
            voxel = queue.popleft()
            component.append(voxel)
            x, y, z = voxel
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if nx < 0 or ny < 0 or nz < 0 or nx >= shape[0] or ny >= shape[1] or nz >= shape[2]:
                    continue
                neighbor = (nx, ny, nz)
                if binary[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        if len(component) >= int(min_size):
            for voxel in component:
                keep[voxel] = True
    return keep


def derive_remodelling_labels_from_density(
    baseline_density: Any,
    followup_density: Any,
    *,
    mask: Any | None = None,
    baseline_segmentation: Any | None = None,
    followup_segmentation: Any | None = None,
    array_order: str = "xyz",
    density_threshold: float = 225.0,
    bone_threshold: float = 320.0,
    gaussian_sigma: float = 1.2,
    cluster_size: int = 12,
    resorption_label: int = 1,
    quiescence_label: int = 2,
    formation_label: int = 3,
) -> np.ndarray:
    """Derive Timelapsed-style remodelling labels from T0/T1 density images.

    Formation and resorption are only accepted when both criteria agree:

    - binary segmentation flips between T0 and T1; and
    - the Gaussian-filtered density changes by at least ``density_threshold``.

    The returned label image uses ``1=resorption``, ``2=quiescence``, and
    ``3=formation`` by default. Voxels outside the analysis mask remain ``0``.
    """
    baseline = _as_numpy_xyz(baseline_density, array_order=array_order).astype(np.float64, copy=False)
    followup = _as_numpy_xyz(followup_density, array_order=array_order).astype(np.float64, copy=False)
    if baseline.shape != followup.shape:
        raise ValueError("baseline_density and followup_density shapes must match")
    analysis_mask = np.ones(baseline.shape, dtype=bool) if mask is None else (_as_numpy_xyz(mask, array_order=array_order) > 0)
    if analysis_mask.shape != baseline.shape:
        raise ValueError("mask shape must match baseline_density")

    baseline_filtered = _gaussian_filter_xyz(baseline, float(gaussian_sigma))
    followup_filtered = _gaussian_filter_xyz(followup, float(gaussian_sigma))
    if baseline_segmentation is None:
        seg_baseline = baseline_filtered > float(bone_threshold)
    else:
        seg_baseline = _as_numpy_xyz(baseline_segmentation, array_order=array_order) > 0
    if followup_segmentation is None:
        seg_followup = followup_filtered > float(bone_threshold)
    else:
        seg_followup = _as_numpy_xyz(followup_segmentation, array_order=array_order) > 0
    if seg_baseline.shape != baseline.shape or seg_followup.shape != baseline.shape:
        raise ValueError("segmentation shapes must match baseline_density")

    seg_baseline = seg_baseline & analysis_mask
    seg_followup = seg_followup & analysis_mask
    grayscale_difference = (followup_filtered - baseline_filtered) * analysis_mask

    formation = (~seg_baseline) & seg_followup & (grayscale_difference > float(density_threshold))
    resorption = seg_baseline & (~seg_followup) & (grayscale_difference < -float(density_threshold))
    formation = _remove_small_components(formation, min_size=int(cluster_size))
    resorption = _remove_small_components(resorption, min_size=int(cluster_size))
    quiescence = seg_baseline & (~formation) & (~resorption)

    labels = np.zeros(baseline.shape, dtype=np.int16)
    labels[resorption] = int(resorption_label)
    labels[quiescence] = int(quiescence_label)
    labels[formation] = int(formation_label)
    return labels


def _shift(mask: np.ndarray, dx: int, dy: int, dz: int) -> np.ndarray:
    """Shift a mask without wraparound, filling newly exposed voxels with False."""
    out = np.zeros_like(mask, dtype=bool)
    xs = slice(max(0, dx), min(mask.shape[0], mask.shape[0] + dx))
    ys = slice(max(0, dy), min(mask.shape[1], mask.shape[1] + dy))
    zs = slice(max(0, dz), min(mask.shape[2], mask.shape[2] + dz))
    src_xs = slice(max(0, -dx), min(mask.shape[0], mask.shape[0] - dx))
    src_ys = slice(max(0, -dy), min(mask.shape[1], mask.shape[1] - dy))
    src_zs = slice(max(0, -dz), min(mask.shape[2], mask.shape[2] - dz))
    out[xs, ys, zs] = mask[src_xs, src_ys, src_zs]
    return out


def _dilate_6(mask: np.ndarray) -> np.ndarray:
    """Return a one-voxel 6-neighbour dilation."""
    return (
        mask
        | _shift(mask, 1, 0, 0)
        | _shift(mask, -1, 0, 0)
        | _shift(mask, 0, 1, 0)
        | _shift(mask, 0, -1, 0)
        | _shift(mask, 0, 0, 1)
        | _shift(mask, 0, 0, -1)
    )


def _erode_6(mask: np.ndarray) -> np.ndarray:
    """Return a one-voxel 6-neighbour erosion."""
    return (
        mask
        & _shift(mask, 1, 0, 0)
        & _shift(mask, -1, 0, 0)
        & _shift(mask, 0, 1, 0)
        & _shift(mask, 0, -1, 0)
        & _shift(mask, 0, 0, 1)
        & _shift(mask, 0, 0, -1)
    )


def _balanced_binary_weights(y: np.ndarray) -> np.ndarray:
    """Return inverse-frequency weights for a binary logistic fit."""
    labels = np.asarray(y, dtype=np.float64)
    weights = np.ones(labels.shape, dtype=np.float64)
    n_pos = float(np.count_nonzero(labels == 1.0))
    n_neg = float(np.count_nonzero(labels == 0.0))
    n_total = n_pos + n_neg
    if n_pos > 0.0 and n_neg > 0.0 and n_total > 0.0:
        weights[labels == 1.0] = n_total / (2.0 * n_pos)
        weights[labels == 0.0] = n_total / (2.0 * n_neg)
    return weights


def _fit_logistic_binary(
    strain: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a two-parameter binary logistic regression with Newton iterations.

    The model is ``logit(P(y=1)) = beta0 + beta1 * strain``. We keep this
    implementation local so the release package remains small and the math is
    visible in one place.
    """
    x = np.column_stack((np.ones_like(strain, dtype=np.float64), strain.astype(np.float64, copy=False)))
    labels = np.asarray(y, dtype=np.float64)
    weights = np.ones(labels.shape, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != labels.shape:
        raise ValueError("sample_weight shape must match y")
    beta = np.zeros(2, dtype=np.float64)
    ridge = 1e-9

    for _ in range(max_iter):
        z = x @ beta
        z = np.clip(z, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(weights * p * (1.0 - p), 1e-9, None)
        grad = x.T @ (weights * (labels - p))
        hess = (x.T * w) @ x + ridge * np.eye(2, dtype=np.float64)
        step = np.linalg.solve(hess, grad)
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new

    z = np.clip(x @ beta, -40.0, 40.0)
    p = 1.0 / (1.0 + np.exp(-z))
    w = np.clip(weights * p * (1.0 - p), 1e-9, None)
    hess = (x.T * w) @ x + ridge * np.eye(2, dtype=np.float64)
    cov = np.linalg.inv(hess)
    se = float(np.sqrt(max(cov[1, 1], 1e-16)))
    z_stat = float(beta[1] / se)
    pval = float(math.erfc(abs(z_stat) / math.sqrt(2.0)))
    return beta, cov, pval


def _fit_logistic_binary_legacy(
    strain: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Legacy-aligned binary logistic fit using the local Newton solver."""
    return _fit_logistic_binary(strain, y)


def _fit_logistic_binary_class_balanced(
    strain: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Binary logistic fit with equal total weight for event and non-event classes."""
    return _fit_logistic_binary(strain, y, sample_weight=_balanced_binary_weights(y))


def _ensure_work_dir(work_dir: str | Path | None) -> Path:
    """Resolve and create output work directory."""
    if work_dir is None:
        local_work_dir = Path.cwd() / "mechanoregulation_run"
    else:
        local_work_dir = Path(work_dir).expanduser().resolve()
    local_work_dir.mkdir(parents=True, exist_ok=True)
    return local_work_dir


def _compute_binned_odds(
    labels: np.ndarray,
    strain: np.ndarray,
    *,
    resorption_label: int,
    quiescence_label: int,
    formation_label: int,
    n_bins: int,
    normalisation_value: float | None = None,
    min_bin_count: int = 30,
    min_class_count: int = 3,
) -> dict[str, Any]:
    """Compute secondary binned odds diagnostics from class-density enrichment.

    These are not the main reported OR_F/OR_R values. They are a diagnostic:
    within each SED bin, compare the local event/quiescence odds with the global
    event/quiescence odds. Low-support bins are hidden with ``NaN`` so they do
    not dominate plots.
    """
    values = np.asarray(strain, dtype=np.float64)
    scale = float(normalisation_value) if normalisation_value is not None else float("nan")
    if np.isfinite(scale) and scale > 0.0:
        values = 100.0 * values / scale
        edges = np.linspace(0.0, 100.0, int(n_bins) + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        pos = values[values > 0]
        if pos.size < 2:
            return {
                "bin_edges": [],
                "bin_centers": [],
                "formation_or": [],
                "resorption_or": [],
                "counts": [],
                "valid_bins": [],
                "overall_counts": {"formation": 0, "resorption": 0, "quiescence": 0},
            }
        smin = float(np.percentile(pos, 1.0))
        smax = float(np.percentile(pos, 99.0))
        if not np.isfinite(smin) or not np.isfinite(smax) or smin <= 0.0 or smax <= smin:
            smin = float(pos.min())
            smax = float(pos.max())
        if smax <= smin:
            return {
                "bin_edges": [],
                "bin_centers": [],
                "formation_or": [],
                "resorption_or": [],
                "counts": [],
                "valid_bins": [],
                "overall_counts": {"formation": 0, "resorption": 0, "quiescence": 0},
            }
        edges = np.logspace(np.log10(smin), np.log10(smax), int(n_bins) + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])

    f_all = int(np.count_nonzero(labels == int(formation_label)))
    r_all = int(np.count_nonzero(labels == int(resorption_label)))
    q_all = int(np.count_nonzero(labels == int(quiescence_label)))
    base_form_odds = (f_all + 0.5) / (q_all + 0.5)
    base_res_odds = (r_all + 0.5) / (q_all + 0.5)

    form_or: list[float] = []
    res_or: list[float] = []
    counts: list[dict[str, int]] = []
    valid_bins: list[bool] = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (values >= lo) & (values < hi if i < len(edges) - 2 else values <= hi)
        lbl = labels[in_bin]
        f = int(np.count_nonzero(lbl == int(formation_label)))
        r = int(np.count_nonzero(lbl == int(resorption_label)))
        q = int(np.count_nonzero(lbl == int(quiescence_label)))
        total = f + r + q
        valid_f = total >= int(min_bin_count) and f >= int(min_class_count) and q >= int(min_class_count)
        valid_r = total >= int(min_bin_count) and r >= int(min_class_count) and q >= int(min_class_count)
        valid_bins.append(bool(valid_f or valid_r))
        if valid_f:
            odds_f = (f + 0.5) / (q + 0.5)
            form_or.append(float(odds_f / base_form_odds))
        else:
            form_or.append(float("nan"))
        if valid_r:
            odds_r = (r + 0.5) / (q + 0.5)
            res_or.append(float(odds_r / base_res_odds))
        else:
            res_or.append(float("nan"))
        counts.append({"formation": f, "resorption": r, "quiescence": q, "total": total})

    return {
        "bin_edges": [float(v) for v in edges],
        "bin_centers": [float(v) for v in centers],
        "formation_or": form_or,
        "resorption_or": res_or,
        "counts": counts,
        "valid_bins": valid_bins,
        "min_bin_count": int(min_bin_count),
        "min_class_count": int(min_class_count),
        "source": "class_prevalence_normalized_bin_odds",
        "overall_counts": {"formation": f_all, "resorption": r_all, "quiescence": q_all},
    }


def _confusion_matrix_per_threshold_set(data_per_category: list[np.ndarray], threshold_list: list[int]) -> np.ndarray:
    """Build the normalized class-by-region matrix used for CCR.

    ``data_per_category`` is ordered as ``[R, Q, F]``. ``threshold_list`` splits
    the SED axis into low, middle, and high regions. The diagonal represents
    mechanically "correct" classification: R in low SED, Q in the lazy zone,
    and F in high SED.
    """
    lengths = [len(data) for data in data_per_category]
    if len(set(lengths)) != 1:
        raise ValueError(f"The data lists passed as input have different sizes: {lengths}.")
    confusion = np.zeros((len(data_per_category), len(data_per_category)), dtype=np.float64)
    bounds = [0, *threshold_list, len(data_per_category[0])]
    for row, category in enumerate(data_per_category):
        values = np.asarray(category, dtype=np.float64)
        for col, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            confusion[row, col] = float(np.sum(values[start:stop]))
    total = float(np.sum(confusion))
    if total > 0.0:
        confusion /= total
    return confusion


def _ccr_from_confusion_matrix(data_per_category: list[np.ndarray], threshold_list: list[int]) -> float:
    """Return the correct classification rate for one threshold set."""
    return float(np.trace(_confusion_matrix_per_threshold_set(data_per_category, threshold_list)))


def _max_ccr(
    data_per_category: list[np.ndarray],
    bin_edges: np.ndarray,
    *,
    support_counts: np.ndarray | None = None,
    min_segment_fraction: float = 0.05,
    max_threshold_percent: float = 95.0,
) -> dict[str, Any]:
    """Find the SED thresholds that maximize correct classification rate.

    For three classes, this searches two thresholds: low/lazy-zone and
    lazy-zone/high. The support constraints avoid choosing a visually absurd
    threshold at the final high-strain bin simply because that bin has very few
    voxels.
    """
    from itertools import combinations

    arrays = [np.asarray(data, dtype=np.float64) for data in data_per_category]
    if len(arrays) < 2:
        return {"max": float("nan"), "threshold_indices": [], "threshold_values": []}
    n_thresholds = len(arrays) - 1
    centers = np.squeeze((np.asarray(bin_edges, dtype=np.float64)[1:] + np.asarray(bin_edges, dtype=np.float64)[:-1]) / 2.0)
    if centers.size <= n_thresholds:
        return {"max": float("nan"), "threshold_indices": [], "threshold_values": []}
    ccr_shape = tuple(np.repeat(centers.size - 1, n_thresholds))
    ccr_matrix = np.full(ccr_shape, np.nan, dtype=np.float64)
    support = None if support_counts is None else np.asarray(support_counts, dtype=np.float64)
    min_segment_count = 0.0
    if support is not None:
        if support.shape != arrays[0].shape:
            raise ValueError("support_counts shape must match histogram data")
        min_segment_count = max(1.0, float(np.sum(support)) * float(min_segment_fraction))
    threshold_indices = np.arange(0, centers.size - 1)
    threshold_indices = threshold_indices[centers[threshold_indices] <= float(max_threshold_percent)]
    if threshold_indices.size < n_thresholds:
        threshold_indices = np.arange(0, centers.size - 1)
    for threshold_set in combinations(threshold_indices, n_thresholds):
        if support is not None:
            bounds = [0, *[int(v) for v in threshold_set], support.size]
            segment_counts = [float(np.sum(support[start:stop])) for start, stop in zip(bounds[:-1], bounds[1:], strict=True)]
            if any(count < min_segment_count for count in segment_counts):
                continue
        ccr_matrix[tuple(threshold_set)] = _ccr_from_confusion_matrix(arrays, list(threshold_set))
    if np.all(np.isnan(ccr_matrix)):
        if support is not None and min_segment_fraction > 0.0:
            return _max_ccr(
                data_per_category,
                bin_edges,
                support_counts=support_counts,
                min_segment_fraction=0.0,
                max_threshold_percent=max_threshold_percent,
            )
        return {"max": float("nan"), "threshold_indices": [], "threshold_values": []}
    max_value = float(np.nanmax(ccr_matrix))
    max_coords = np.where(ccr_matrix == max_value)
    candidate_coords = [tuple(int(coord[i]) for coord in max_coords) for i in range(max_coords[0].size)]
    if support is not None and candidate_coords:
        cumulative = np.cumsum(support)
        total = float(cumulative[-1])
        support_median_idx = int(np.searchsorted(cumulative, 0.5 * total, side="left")) if total > 0.0 else centers.size // 2

        def _tie_score(coords: tuple[int, ...]) -> tuple[float, float, tuple[int, ...]]:
            segment_balance = 0.0
            bounds = [0, *coords, support.size]
            if total > 0.0:
                fractions = np.array([np.sum(support[start:stop]) / total for start, stop in zip(bounds[:-1], bounds[1:], strict=True)])
                segment_balance = float(np.max(np.abs(fractions - (1.0 / len(bounds[:-1])))))
            return segment_balance, float(np.mean(np.abs(np.asarray(coords) - support_median_idx))), coords

        first = min(candidate_coords, key=_tie_score)
    else:
        first = tuple(int(coord[0]) for coord in max_coords)
    return {
        "max": max_value,
        "threshold_indices": list(first),
        "threshold_values": [float(centers[idx]) for idx in first],
        "matrix_shape": list(ccr_matrix.shape),
        "min_segment_fraction": float(min_segment_fraction) if support is not None else 0.0,
        "max_threshold_percent": float(max_threshold_percent),
    }


def _normalize_histograms_longwise(histograms: list[np.ndarray]) -> list[np.ndarray]:
    """Normalize each class histogram by its own total event count.

    This is the Schulte-style step that prevents quiescence from overwhelming
    formation and resorption simply because quiescent surface voxels are much
    more common.
    """
    normed: list[np.ndarray] = []
    for hist in histograms:
        values = np.asarray(hist, dtype=np.float64)
        total = float(np.sum(values))
        if total > 0.0:
            normed.append(values / total)
        else:
            normed.append(np.zeros_like(values, dtype=np.float64))
    return normed


def _normalize_histograms_binwise(histograms: list[np.ndarray]) -> list[np.ndarray]:
    """Convert class-normalized histograms into per-bin conditional curves."""
    arrays = [np.asarray(hist, dtype=np.float64) for hist in histograms]
    sums = np.zeros_like(arrays[0], dtype=np.float64)
    for arr in arrays:
        if arr.shape != sums.shape:
            raise ValueError("all histogram arrays must have the same shape")
        sums += arr
    return [np.divide(arr, sums, out=np.zeros_like(arr, dtype=np.float64), where=sums != 0) for arr in arrays]


def _schulte_conditional_curves(
    *,
    labels: np.ndarray,
    strain: np.ndarray,
    resorption_label: int,
    quiescence_label: int,
    formation_label: int,
    normalisation_value: str | float = "p99",
    n_bins: int = 101,
) -> dict[str, Any]:
    """Compute Schulte-style conditional probability curves.

    SED is normalized to the requested percentile, binned from 0-100%, and then
    class histograms are normalized in two stages:

    1. within each class, by the total number of events of that class;
    2. within each SED bin, so R/Q/F sum to one.

    The resulting curves answer: given this normalized SED bin, is this surface
    location enriched for resorption, quiescence, or formation?
    """
    values = np.asarray(strain, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("strain must contain finite values")
    if isinstance(normalisation_value, str):
        token = normalisation_value.strip().lower()
        if not token.startswith("p"):
            raise ValueError("normalisation_value strings must be percentile tokens like 'p99'")
        scale = float(np.percentile(finite, int(token[1:])))
    else:
        scale = float(normalisation_value)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(finite))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    bins = np.linspace(0.0, 100.0, int(n_bins), dtype=np.float64)
    normalized = 100.0 * values / scale
    hist_f, _ = np.histogram(normalized[labels == int(formation_label)], bins=bins, density=False)
    hist_q, _ = np.histogram(normalized[labels == int(quiescence_label)], bins=bins, density=False)
    hist_r, _ = np.histogram(normalized[labels == int(resorption_label)], bins=bins, density=False)

    long_f, long_q, long_r = _normalize_histograms_longwise([hist_f, hist_q, hist_r])
    prob_f, prob_q, prob_r = _normalize_histograms_binwise([long_f, long_q, long_r])
    centers = 0.5 * (bins[:-1] + bins[1:])
    ccr = _max_ccr([prob_r, prob_q, prob_f], bins, support_counts=hist_f + hist_q + hist_r)
    return {
        "strain": [float(v) for v in centers],
        "x_axis": "normalized_percent_linear",
        "normalisation_value": float(scale),
        "bins": [float(v) for v in bins],
        "ccr": ccr,
        "counts": {
            "formation": [int(v) for v in hist_f],
            "quiescence": [int(v) for v in hist_q],
            "resorption": [int(v) for v in hist_r],
            "total": [int(v) for v in hist_f + hist_q + hist_r],
        },
        "F": {"mean": [float(v) for v in prob_f]},
        "Q": {"mean": [float(v) for v in prob_q]},
        "R": {"mean": [float(v) for v in prob_r]},
    }


def _extract_surface_dilated_events(
    remodelling_xyz: np.ndarray,
    strain_xyz: np.ndarray,
    *,
    mask_xyz: np.ndarray | None = None,
    resorption_label: int,
    quiescence_label: int,
    formation_label: int,
    cap_percentile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return event labels and SED sampled on the baseline bone surface.

    Formation and resorption are first identified in the remodelling label
    image. The SED itself is not sampled at those event voxels. Instead, both
    event classes are dilated by one voxel and projected onto neighbouring
    quiescent baseline-surface voxels. This gives formation and resorption the
    same surface handling. If formation and resorption project to the same
    surface voxel, that location is treated as ambiguous and remains
    quiescent.
    """
    if mask_xyz is None:
        analysis_mask = np.ones(remodelling_xyz.shape, dtype=bool)
    else:
        analysis_mask = np.asarray(mask_xyz) > 0
        if analysis_mask.shape != remodelling_xyz.shape:
            raise ValueError("analysis_mask shape must match remodelling_image")
    # Only voxels with one of the three biological labels can participate in
    # the surface analysis. Label 0 is background/outside the remodelling
    # definition and is ignored.
    valid_labels = (
        (remodelling_xyz == int(resorption_label))
        | (remodelling_xyz == int(quiescence_label))
        | (remodelling_xyz == int(formation_label))
    )
    strain_support = np.isfinite(strain_xyz) & (strain_xyz > 0.0)
    eval_mask = analysis_mask & valid_labels & strain_support

    if not np.any(eval_mask):
        return np.array([], dtype=np.int16), np.array([], dtype=np.float64), {
            "n_eval_voxels": 0,
            "n_surface_voxels": 0,
            "n_sampled_voxels": 0,
            "n_formation": 0,
            "n_resorption": 0,
            "n_quiescence": 0,
        }

    # Cap extreme SED values before normalization. This keeps a few numerical
    # outliers from compressing the biologically relevant 0-100% axis.
    strain = strain_xyz.astype(np.float64, copy=True)
    strain[~eval_mask] = 0.0
    cap_value = float(np.percentile(strain[eval_mask], float(cap_percentile)))
    strain[eval_mask] = np.minimum(strain[eval_mask], cap_value)

    # Baseline mechanics are sampled symmetrically on quiescent baseline
    # surface. Direct formation and resorption voxels are never sampled.
    bone_support = eval_mask & (remodelling_xyz == int(quiescence_label))
    bone_surface = bone_support & (~_erode_6(bone_support))
    sample_mask = bone_surface

    # Projection is symmetric: both F and R events are dilated by one voxel and
    # then mapped onto neighbouring quiescent baseline surface voxels.
    form_dil = analysis_mask & _dilate_6(analysis_mask & (remodelling_xyz == int(formation_label)))
    res_dil = analysis_mask & _dilate_6(analysis_mask & (remodelling_xyz == int(resorption_label)))
    form_on_surface = sample_mask & form_dil
    res_on_surface = sample_mask & res_dil

    sampled_labels = np.full(remodelling_xyz.shape, int(quiescence_label), dtype=np.int16)
    sampled_labels[form_on_surface] = int(formation_label)
    sampled_labels[res_on_surface] = int(resorption_label)

    # Ambiguous F/R projections on the same sampled surface cancel out.
    both = form_on_surface & res_on_surface
    if np.any(both):
        sampled_labels[both] = int(quiescence_label)

    sampled_strain = strain[sample_mask]
    sampled_events = sampled_labels[sample_mask]

    counts = {
        "n_eval_voxels": int(np.count_nonzero(eval_mask)),
        "n_surface_voxels": int(np.count_nonzero(bone_surface)),
        "n_sampled_voxels": int(np.count_nonzero(sample_mask)),
        "n_formation": int(np.count_nonzero(sampled_events == int(formation_label))),
        "n_resorption": int(np.count_nonzero(sampled_events == int(resorption_label))),
        "n_quiescence": int(np.count_nonzero(sampled_events == int(quiescence_label))),
        "n_cancelled_overlap": int(np.count_nonzero(both)),
        "surface_event_mapping": "symmetric_surface_cancel_overlap",
    }
    return sampled_events, sampled_strain, counts


def _curve_from_models(strain_support: np.ndarray, beta_form: np.ndarray, beta_res: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert two binary logistic fits into a three-class probability curve.

    Formation and resorption are each modelled as event-vs-rest odds. We combine
    those odds with a quiescence baseline odds of one, giving probabilities that
    sum to one across R/Q/F at each SED value.
    """
    s = strain_support.astype(np.float64, copy=False)
    form_odds = np.exp(np.clip(beta_form[0] + beta_form[1] * s, -40.0, 40.0))
    res_odds = np.exp(np.clip(beta_res[0] + beta_res[1] * s, -40.0, 40.0))
    total = 1.0 + form_odds + res_odds
    pf = form_odds / total
    pr = res_odds / total
    pq = 1.0 / total
    return pr, pq, pf


def _model_conditional_curves(
    *,
    schulte_curves: dict[str, Any],
    beta_form: np.ndarray,
    beta_res: np.ndarray,
    boot_beta_form: np.ndarray | None = None,
    boot_beta_res: np.ndarray | None = None,
    ci_alpha: float = 0.01,
) -> dict[str, Any]:
    """Evaluate smooth logistic probability curves and bootstrap envelopes."""
    support = np.asarray(schulte_curves["strain"], dtype=np.float64)
    scale = float(schulte_curves["normalisation_value"])
    pr, pq, pf = _curve_from_models(support, beta_form, beta_res)
    f_curve: dict[str, Any] = {"mean": [float(v) for v in pf]}
    q_curve: dict[str, Any] = {"mean": [float(v) for v in pq]}
    r_curve: dict[str, Any] = {"mean": [float(v) for v in pr]}

    if boot_beta_form is not None and boot_beta_res is not None:
        boot_form = np.asarray(boot_beta_form, dtype=np.float64)
        boot_res = np.asarray(boot_beta_res, dtype=np.float64)
        valid = np.isfinite(boot_form).all(axis=1) & np.isfinite(boot_res).all(axis=1)
        if np.any(valid):
            boot_pr: list[np.ndarray] = []
            boot_pq: list[np.ndarray] = []
            boot_pf: list[np.ndarray] = []
            for beta_f, beta_r in zip(boot_form[valid], boot_res[valid], strict=True):
                r_i, q_i, f_i = _curve_from_models(support, beta_f, beta_r)
                boot_pr.append(r_i)
                boot_pq.append(q_i)
                boot_pf.append(f_i)
            ql = 100.0 * (float(ci_alpha) / 2.0)
            qh = 100.0 * (1.0 - float(ci_alpha) / 2.0)
            for curve, samples in ((r_curve, boot_pr), (q_curve, boot_pq), (f_curve, boot_pf)):
                arr = np.asarray(samples, dtype=np.float64)
                curve["median"] = [float(v) for v in np.percentile(arr, 50.0, axis=0, method="midpoint")]
                curve["low"] = [float(v) for v in np.percentile(arr, ql, axis=0, method="midpoint")]
                curve["high"] = [float(v) for v in np.percentile(arr, qh, axis=0, method="midpoint")]

    return {
        "strain": [float(v) for v in support],
        "x_axis": "normalized_percent_linear",
        "curve_type": "logistic_odds_normalized_percent_model",
        "normalisation_value": scale,
        "bins": schulte_curves.get("bins", []),
        "ccr": schulte_curves.get("ccr", {}),
        "F": f_curve,
        "Q": q_curve,
        "R": r_curve,
        "schulte": schulte_curves,
    }


def _first_curve_crossing(x: np.ndarray, y: np.ndarray, *, prefer: str) -> float:
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


def _logistic_lazy_zone(conditional_curves: dict[str, Any]) -> dict[str, float]:
    """Estimate lazy-zone bounds from logistic R/Q and F/Q crossings."""
    x = np.asarray(conditional_curves.get("strain", []), dtype=np.float64)
    if x.size == 0:
        return {"low": float("nan"), "high": float("nan")}
    pr_curve = conditional_curves.get("R", {})
    pq_curve = conditional_curves.get("Q", {})
    pf_curve = conditional_curves.get("F", {})
    pr = np.asarray(pr_curve.get("median", pr_curve.get("mean", [])), dtype=np.float64)
    pq = np.asarray(pq_curve.get("median", pq_curve.get("mean", [])), dtype=np.float64)
    pf = np.asarray(pf_curve.get("median", pf_curve.get("mean", [])), dtype=np.float64)
    if pr.size != x.size or pq.size != x.size or pf.size != x.size:
        return {"low": float("nan"), "high": float("nan")}
    return {
        "low": _first_curve_crossing(x, pr - pq, prefer="falling"),
        "high": _first_curve_crossing(x, pf - pq, prefer="rising"),
    }


def _plot_curves(
    *,
    strain_support: np.ndarray,
    conditional_curves: dict[str, Any],
    binned: dict[str, Any],
    odds_ratio_f: float,
    odds_ratio_r: float,
    work_dir: Path,
    run_name: str,
) -> dict[str, Path]:
    """Write logistic and Schulte-style conditional probability figures."""
    import matplotlib.pyplot as plt

    out: dict[str, Path] = {}

    fig1 = work_dir / f"{run_name}_conditional_curves.png"
    s = strain_support
    pr_curve = conditional_curves["R"]
    pq_curve = conditional_curves["Q"]
    pf_curve = conditional_curves["F"]
    pr = np.asarray(pr_curve.get("median", pr_curve.get("mean", [])), dtype=np.float64)
    pq = np.asarray(pq_curve.get("median", pq_curve.get("mean", [])), dtype=np.float64)
    pf = np.asarray(pf_curve.get("median", pf_curve.get("mean", [])), dtype=np.float64)

    plt.figure(figsize=(7, 5))
    for key, color in (("R", RESORPTION_COLOR), ("Q", QUIESCENCE_COLOR), ("F", FORMATION_COLOR)):
        curve = conditional_curves.get(key, {})
        if "low" in curve and "high" in curve:
            low = np.asarray(curve["low"], dtype=np.float64)
            high = np.asarray(curve["high"], dtype=np.float64)
            plt.fill_between(s, low, high, color=color, alpha=0.16, linewidth=0)
    plt.plot(s, pr, color=RESORPTION_COLOR, label="P(R|s)")
    plt.plot(s, pq, color=QUIESCENCE_COLOR, label="P(Q|s)")
    plt.plot(s, pf, color=FORMATION_COLOR, label="P(F|s)")
    logistic_lazy_zone = conditional_curves.get("logistic_lazy_zone", {})
    lazy_low = logistic_lazy_zone.get("low", float("nan")) if isinstance(logistic_lazy_zone, dict) else float("nan")
    lazy_high = logistic_lazy_zone.get("high", float("nan")) if isinstance(logistic_lazy_zone, dict) else float("nan")
    if np.isfinite(float(lazy_low)) and np.isfinite(float(lazy_high)):
        plt.axvspan(float(lazy_low), float(lazy_high), color=LAZY_ZONE_COLOR, alpha=0.28, linewidth=0, label="logistic lazy zone")
        plt.axvline(float(lazy_low), color=RESORPTION_COLOR, linestyle="--", linewidth=1.0, alpha=0.7, label="logistic low")
        plt.axvline(float(lazy_high), color=FORMATION_COLOR, linestyle="--", linewidth=1.0, alpha=0.7, label="logistic high")
    or_f = 100.0 * (float(odds_ratio_f) - 1.0)
    or_r = 100.0 * (float(odds_ratio_r) - 1.0)
    plt.title(f"Parametric logistic diagnostic (OR_F={or_f:.2g}%, OR_R={or_r:.2g}%)")
    plt.xlabel("Normalized strain/SED (%)")
    plt.ylabel("probability")
    plt.ylim(0.0, 1.0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig1, dpi=160)
    plt.close()
    out["conditional_curves"] = fig1

    schulte = conditional_curves.get("schulte", {})
    schulte_s = np.asarray(schulte.get("strain", []), dtype=np.float64) if isinstance(schulte, dict) else np.array([])
    if schulte_s.size:
        fig_schulte = work_dir / f"{run_name}_schulte_binned_curves.png"
        plt.figure(figsize=(7, 5))
        for key, color in (("R", RESORPTION_COLOR), ("Q", QUIESCENCE_COLOR), ("F", FORMATION_COLOR)):
            curve = schulte.get(key, {}) if isinstance(schulte, dict) else {}
            values = np.asarray(curve.get("mean", []), dtype=np.float64)
            if values.size == schulte_s.size:
                plt.plot(
                    schulte_s,
                    values,
                    color=color,
                    marker="o",
                    markersize=2.5,
                    linewidth=1.0,
                    label=f"P({key}|s)",
                )
        ccr = conditional_curves.get("ccr", {})
        thresholds = ccr.get("threshold_values", []) if isinstance(ccr, dict) else []
        if len(thresholds) >= 2:
            low = float(thresholds[0])
            high = float(thresholds[1])
            plt.axvspan(low, high, color=LAZY_ZONE_COLOR, alpha=0.28, linewidth=0, label="CCR lazy zone")
            plt.axvline(low, color=RESORPTION_COLOR, linestyle="--", linewidth=1.0, alpha=0.7, label="CCR low")
            plt.axvline(high, color=FORMATION_COLOR, linestyle="--", linewidth=1.0, alpha=0.7, label="CCR high")
        if isinstance(ccr, dict) and np.isfinite(float(ccr.get("max", float("nan")))):
            plt.title(f"Schulte binned curves (CCR={float(ccr['max']):.3f})")
        plt.xlabel("Normalized strain/SED (%)")
        plt.ylabel("class-density normalized probability")
        plt.ylim(0.0, 1.0)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(fig_schulte, dpi=160)
        plt.close()
        out["schulte_binned_curves"] = fig_schulte

    return out


def mechanoregulation(
    *,
    remodelling_image: Any | None = None,
    baseline_density: Any | None = None,
    followup_density: Any | None = None,
    baseline_strain: Any | None = None,
    analysis_mask: Any | None = None,
    baseline_segmentation: Any | None = None,
    followup_segmentation: Any | None = None,
    array_order: str = "xyz",
    resorption_label: int = 1,
    quiescence_label: int = 2,
    formation_label: int = 3,
    remodelling_density_threshold: float = 225.0,
    remodelling_bone_threshold: float = 320.0,
    remodelling_gaussian_sigma: float = 1.2,
    remodelling_cluster_size: int = 12,
    run_name: str = "mechanoregulation",
    work_dir: str | Path | None = None,
    cap_percentile: float = 99.0,
    n_boot: int = 1000,
    seed: int = 0,
    bootstrap_sampling_perc: float = 10.0,
    ci_alpha: float = 0.01,
    resorption_or_definition: str = "decreasing_strain",
    legacy_clip_to_unit: bool = True,
    odds_n_bins: int = 12,
    plot: bool = False,
    return_full: bool = False,
) -> tuple[float, float] | MechanoregulationResult:
    """Run voxel-wise mechanoregulation analysis.

    The analysis follows four explicit steps:

    1. Create or read a remodelling label image:
       ``1=resorption``, ``2=quiescence``, ``3=formation``.
    2. Sample baseline SED only on quiescent baseline surface voxels near F/R
       events, with symmetric F/R projection and overlap cancellation.
    3. Normalize sampled SED to 0-100 using the capped analysis maximum
       (normally the 99th percentile cap).
    4. Fit binary logistic models for formation and resorption and bootstrap
       their slopes to report odds ratios and probability curves.

    Args:
        remodelling_image: Optional label image where labels are
            ``1=resorption``, ``2=quiescence``, and ``3=formation``. This is the
            preferred input after timelapse processing or synthetic advection.
        baseline_density: Baseline grayscale image. Used only when
            ``remodelling_image`` is not supplied.
        followup_density: Follow-up grayscale image. Used together with
            ``baseline_density`` to derive remodelling labels.
        baseline_strain: Baseline SED/strain image from ParOSol. Required.
        analysis_mask: Optional ROI mask. All event extraction and SED sampling
            are restricted to this mask.
        array_order: ``"xyz"`` for NumPy arrays already in analysis order, or
            ``"zyx"`` for arrays in SimpleITK's NumPy order.
        cap_percentile: Percentile used to cap extreme SED values before
            normalizing the analysis axis.
        n_boot: Number of class-balanced bootstrap samples for OR confidence
            intervals and smooth probability bands.
        bootstrap_sampling_perc: Percentage of the smallest class sampled from
            each of F/R/Q in each bootstrap replicate.
        resorption_or_definition: ``"decreasing_strain"`` reports OR_R as
            enrichment with lower SED, which matches the usual biological
            interpretation of resorption.
        legacy_clip_to_unit: Optional compatibility clipping for older examples
            whose SED was already normalized to 0-1.
        plot: If true, write diagnostic figures into ``work_dir``.
        return_full: If true, return :class:`MechanoregulationResult`; otherwise
            return only ``(OR_F, OR_R)`` for older lightweight callers.

    Returns:
        Either ``(OR_F, OR_R)`` or a full :class:`MechanoregulationResult`.
    """
    # Step 1: normalize remodelling input representation.
    if remodelling_image is None:
        if baseline_density is None or followup_density is None:
            raise ValueError("remodelling_image or both baseline_density and followup_density are required")
        remodelling_xyz = derive_remodelling_labels_from_density(
            baseline_density,
            followup_density,
            mask=analysis_mask,
            baseline_segmentation=baseline_segmentation,
            followup_segmentation=followup_segmentation,
            array_order=array_order,
            density_threshold=float(remodelling_density_threshold),
            bone_threshold=float(remodelling_bone_threshold),
            gaussian_sigma=float(remodelling_gaussian_sigma),
            cluster_size=int(remodelling_cluster_size),
            resorption_label=resorption_label,
            quiescence_label=quiescence_label,
            formation_label=formation_label,
        )
    else:
        remodelling_xyz = _as_numpy_xyz(remodelling_image, array_order=array_order).astype(np.int16, copy=False)

    # Step 2: require baseline strain source from a post-timelapse solver stage.
    if baseline_strain is None:
        raise ValueError("baseline_strain is required for post-timelapse mechanoregulation")
    strain_xyz = _as_numpy_xyz(baseline_strain, array_order=array_order).astype(np.float32, copy=False)
    if strain_xyz.shape != remodelling_xyz.shape:
        raise ValueError("baseline_strain shape must match remodelling_image")
    mask_xyz = None
    if analysis_mask is not None:
        mask_xyz = _as_numpy_xyz(analysis_mask, array_order=array_order)
        if mask_xyz.shape != remodelling_xyz.shape:
            raise ValueError("analysis_mask shape must match remodelling_image")
    baseline_source = "provided_baseline"

    # Step 3: legacy-consistent event extraction from surface-dilated labels.
    labels, strain, counts = _extract_surface_dilated_events(
        remodelling_xyz,
        strain_xyz,
        mask_xyz=mask_xyz,
        resorption_label=resorption_label,
        quiescence_label=quiescence_label,
        formation_label=formation_label,
        cap_percentile=cap_percentile,
    )
    if strain.size < 3:
        raise ValueError("not enough sampled voxels for mechanoregulation analysis")
    if legacy_clip_to_unit:
        strain = np.minimum(strain, 1.0)

    finite_strain = strain[np.isfinite(strain)]
    normalisation_scale = float(np.max(finite_strain)) if finite_strain.size else 1.0
    if not np.isfinite(normalisation_scale) or normalisation_scale <= 0.0:
        normalisation_scale = 1.0
    strain_percent = np.clip(100.0 * strain / normalisation_scale, 0.0, 100.0)

    # Step 4: fit logistic models for formation and resorption events.
    y_form = (labels == int(formation_label)).astype(np.float64)
    y_res = (labels == int(resorption_label)).astype(np.float64)
    if np.all(y_form == y_form[0]) or np.all(y_res == y_res[0]):
        raise ValueError("sampled labels must contain both positive and negative classes for F and R")

    # The full-data logistic slopes are used as fallbacks. The reported OR point
    # estimates come from the bootstrap median below because that matches the
    # bootstrap confidence interval construction.
    beta_form, _cov_form, pval_form_full = _fit_logistic_binary_legacy(strain_percent, y_form)
    beta_res, _cov_res, pval_res_full = _fit_logistic_binary_legacy(strain_percent, y_res)
    beta_form_curve, _cov_form_curve, _pval_form_curve = _fit_logistic_binary_class_balanced(strain_percent, y_form)
    beta_res_curve, _cov_res_curve, _pval_res_curve = _fit_logistic_binary_class_balanced(strain_percent, y_res)
    res_or_def = str(resorption_or_definition).strip().lower()
    if res_or_def not in {"increasing_strain", "decreasing_strain"}:
        raise ValueError("resorption_or_definition must be 'increasing_strain' or 'decreasing_strain'")

    # Step 5: evaluate smooth model curves and keep Schulte histograms as diagnostics.
    schulte_curves = _schulte_conditional_curves(
        labels=labels,
        strain=strain,
        resorption_label=resorption_label,
        quiescence_label=quiescence_label,
        formation_label=formation_label,
        normalisation_value=normalisation_scale,
        n_bins=101,
    )
    # Bootstrap sampling is class-balanced. Each replicate contains the same
    # number of F, R, and Q voxels, so quiescence cannot dominate the fitted
    # curve merely because it is much more frequent on the surface.
    rng = np.random.default_rng(int(seed))
    boot_beta_form = np.full((int(n_boot), 2), np.nan, dtype=np.float64)
    boot_beta_res = np.full((int(n_boot), 2), np.nan, dtype=np.float64)
    boot_p_form = np.full(int(n_boot), np.nan, dtype=np.float64)
    boot_p_res = np.full(int(n_boot), np.nan, dtype=np.float64)

    # Step 6: class-balanced bootstrap for OR/CI uncertainty.
    idx_f = np.flatnonzero(y_form == 1.0)
    idx_r = np.flatnonzero(y_res == 1.0)
    idx_q = np.flatnonzero((y_form == 0.0) & (y_res == 0.0))
    min_class = int(min(idx_f.size, idx_r.size, idx_q.size))
    if min_class < 1:
        raise ValueError("sampled labels must contain F, R, and Q classes for bootstrap")
    sample_n = int(round(min_class * float(bootstrap_sampling_perc) / 100.0))
    sample_n = max(1, sample_n)

    for i in range(int(n_boot)):
        boot_idx = np.concatenate(
            (
                rng.choice(idx_f, size=sample_n, replace=True),
                rng.choice(idx_r, size=sample_n, replace=True),
                rng.choice(idx_q, size=sample_n, replace=True),
            )
        )
        s_b = strain_percent[boot_idx]
        yf_b = y_form[boot_idx]
        yr_b = y_res[boot_idx]
        if np.all(yf_b == yf_b[0]) or np.all(yr_b == yr_b[0]):
            continue
        try:
            beta_f_b, _cov_f_b, p_f_b = _fit_logistic_binary_class_balanced(s_b, yf_b)
            beta_r_b, _cov_r_b, p_r_b = _fit_logistic_binary_class_balanced(s_b, yr_b)
        except np.linalg.LinAlgError:
            continue
        boot_beta_form[i, :] = beta_f_b
        boot_beta_res[i, :] = beta_r_b
        boot_p_form[i] = float(p_f_b)
        boot_p_res[i] = float(p_r_b)

    alpha = float(ci_alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("ci_alpha must be between 0 and 1")
    ql = 100.0 * (alpha / 2.0)
    qh = 100.0 * (1.0 - alpha / 2.0)

    # The smooth curves shown in the right-hand panel are generated from the
    # class-balanced logistic fits, while the jagged left-hand curves remain the
    # direct Schulte binned conditional probabilities.
    conditional_curves = _model_conditional_curves(
        schulte_curves=schulte_curves,
        beta_form=beta_form_curve,
        beta_res=beta_res_curve,
        boot_beta_form=boot_beta_form,
        boot_beta_res=boot_beta_res,
        ci_alpha=alpha,
    )
    conditional_curves["class_weighting"] = "balanced_event_vs_non_event"
    conditional_curves["logistic_lazy_zone"] = _logistic_lazy_zone(conditional_curves)

    def _or_from_boot_slopes(
        slopes: np.ndarray,
        *,
        fallback_slope: float,
        increasing: bool,
    ) -> tuple[float, tuple[float, float]]:
        """Convert bootstrap slope samples into an OR point estimate and CI."""
        finite = slopes[np.isfinite(slopes)]
        if finite.size == 0:
            slope_mean = float(fallback_slope)
            slope_lo = float(fallback_slope)
            slope_hi = float(fallback_slope)
        else:
            slope_mean = float(np.percentile(finite, 50.0, method="midpoint"))
            slope_lo = float(np.percentile(finite, ql, method="midpoint"))
            slope_hi = float(np.percentile(finite, qh, method="midpoint"))

        if increasing:
            or_pt = float(np.exp(slope_mean))
            or_lo = float(np.exp(slope_lo))
            or_hi = float(np.exp(slope_hi))
        else:
            or_pt = float(np.exp(-slope_mean))
            or_lo = float(np.exp(-slope_hi))
            or_hi = float(np.exp(-slope_lo))
        return or_pt, (or_lo, or_hi)

    orf, orf_ci = _or_from_boot_slopes(
        boot_beta_form[:, 1],
        fallback_slope=float(beta_form[1]),
        increasing=True,
    )
    orr_increasing_strain, orr_increasing_strain_ci = _or_from_boot_slopes(
        boot_beta_res[:, 1],
        fallback_slope=float(beta_res[1]),
        increasing=True,
    )
    orr_decreasing_strain, orr_decreasing_strain_ci = _or_from_boot_slopes(
        boot_beta_res[:, 1],
        fallback_slope=float(beta_res[1]),
        increasing=False,
    )
    if res_or_def == "increasing_strain":
        orr = orr_increasing_strain
        orr_ci = orr_increasing_strain_ci
    else:
        orr = orr_decreasing_strain
        orr_ci = orr_decreasing_strain_ci
    pval_form = float(np.nanmean(boot_p_form)) if np.any(np.isfinite(boot_p_form)) else float(pval_form_full)
    pval_res = float(np.nanmean(boot_p_res)) if np.any(np.isfinite(boot_p_res)) else float(pval_res_full)

    binned = _compute_binned_odds(
        labels,
        strain,
        resorption_label=resorption_label,
        quiescence_label=quiescence_label,
        formation_label=formation_label,
        n_bins=int(odds_n_bins),
        normalisation_value=float(conditional_curves["normalisation_value"]),
    )
    binned["x_axis"] = "normalized_percent_linear"
    binned["ccr"] = conditional_curves.get("ccr", {})

    # Step 7: optional plotting. Do not create the default work directory unless
    # a caller actually requested files; pure numerical calls should be silent.
    plot_paths: dict[str, Path] | None = None
    if plot:
        local_work_dir = _ensure_work_dir(work_dir)
        plot_paths = _plot_curves(
            strain_support=np.asarray(conditional_curves["strain"], dtype=np.float64),
            conditional_curves=conditional_curves,
            binned=binned,
            odds_ratio_f=float(orf),
            odds_ratio_r=float(orr),
            work_dir=local_work_dir,
            run_name=run_name,
        )

    settings = {
        "resorption_label": int(resorption_label),
        "quiescence_label": int(quiescence_label),
        "formation_label": int(formation_label),
        "cap_percentile": float(cap_percentile),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "bootstrap_sampling_perc": float(bootstrap_sampling_perc),
        "ci_alpha": float(ci_alpha),
        "legacy_clip_to_unit": bool(legacy_clip_to_unit),
        "logistic_strain_scale": "normalized_percent_of_analysis_max_after_cap",
        "logistic_or_unit": "one_normalized_percentage_point",
        "fit_backend_form": "local_newton_logit",
        "fit_backend_res": "local_newton_logit",
        "odds_n_bins": int(odds_n_bins),
        "sampling_strategy": "class-balanced bootstrap with replacement (legacy-aligned)",
        "surface_event_mapping": "symmetric_surface_cancel_overlap",
        "baseline_source": baseline_source,
        "resorption_or_definition": res_or_def,
        "remodelling_source": "density_flip_and_threshold" if remodelling_image is None else "provided_label_image",
        "remodelling_density_threshold": float(remodelling_density_threshold),
        "remodelling_bone_threshold": float(remodelling_bone_threshold),
        "remodelling_gaussian_sigma": float(remodelling_gaussian_sigma),
        "remodelling_cluster_size": int(remodelling_cluster_size),
    }

    result = MechanoregulationResult(
        orf=orf,
        orr=orr,
        orf_ci=orf_ci,
        orr_ci=orr_ci,
        orr_increasing_strain=orr_increasing_strain,
        orr_decreasing_strain=orr_decreasing_strain,
        orr_increasing_strain_ci=orr_increasing_strain_ci,
        orr_decreasing_strain_ci=orr_decreasing_strain_ci,
        pvalue_form=float(pval_form),
        pvalue_res=float(pval_res),
        conditional_curves=conditional_curves,
        binned_odds_diagnostics=binned,
        sample_counts=counts,
        settings=settings,
        plot_paths=plot_paths,
    )

    if return_full:
        return result
    return result.orf, result.orr
