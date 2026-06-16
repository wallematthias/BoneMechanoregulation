import numpy as np
import pytest

import bonemechreg.mechreg as mechreg_module
from bonemechreg.mechreg import MechanoregulationResult, derive_remodelling_labels_from_density, mechanoregulation


def _make_synthetic_case() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.05, 0.95, 30, dtype=np.float64)
    strain = np.repeat(x[:, None], 30, axis=1)[..., None]
    labels = np.full(strain.shape, 2, dtype=np.int16)
    labels[strain < 0.35] = 1
    labels[strain > 0.65] = 3
    return labels, strain


def _make_gaussian_surface_case(
    *,
    seed: int = 7,
    n_res: int = 16_000,
    n_for: int = 16_000,
    quiescence_factor: int = 5,
    strain_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic surface-only case with overlapping class strain Gaussians."""
    rng = np.random.default_rng(seed)
    n_qui = int(quiescence_factor * n_res)

    mu = {1: 0.47, 2: 0.50, 3: 0.53}
    sd = {1: 0.16, 2: 0.16, 3: 0.16}

    res_s = rng.normal(mu[1], sd[1], size=n_res)
    qui_s = rng.normal(mu[2], sd[2], size=n_qui)
    for_s = rng.normal(mu[3], sd[3], size=n_for)

    strain_flat = np.concatenate([res_s, qui_s, for_s]).astype(np.float32) * float(strain_scale)
    strain_flat = np.clip(strain_flat, 0.05, 0.95)
    labels_flat = np.concatenate(
        [
            np.full(n_res, 1, dtype=np.int16),
            np.full(n_qui, 2, dtype=np.int16),
            np.full(n_for, 3, dtype=np.int16),
        ]
    )

    remodelling = np.full((2, labels_flat.size, 1), 2, dtype=np.int16)
    remodelling[0, :, 0] = labels_flat
    baseline = np.zeros(remodelling.shape, dtype=np.float32)
    baseline[1, :, 0] = strain_flat
    baseline[0, :, 0] = strain_flat
    return remodelling, baseline


def _make_separated_imbalanced_surface_case() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    res_s = np.clip(rng.normal(0.12, 0.025, size=2_000), 0.02, 0.25)
    qui_s = np.clip(rng.normal(0.50, 0.09, size=40_000), 0.22, 0.78)
    for_s = np.clip(rng.normal(0.88, 0.025, size=2_000), 0.75, 0.98)
    strain_flat = np.concatenate([res_s, qui_s, for_s]).astype(np.float32)
    labels_flat = np.concatenate(
        [
            np.full(res_s.size, 1, dtype=np.int16),
            np.full(qui_s.size, 2, dtype=np.int16),
            np.full(for_s.size, 3, dtype=np.int16),
        ]
    )
    remodelling = np.full((2, labels_flat.size, 1), 2, dtype=np.int16)
    remodelling[0, :, 0] = labels_flat
    baseline = np.zeros(remodelling.shape, dtype=np.float32)
    baseline[1, :, 0] = strain_flat
    baseline[0, :, 0] = strain_flat
    return remodelling, baseline


def _make_overlapping_imbalanced_surface_case() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13)
    n_event = 2_000
    res_s = np.clip(rng.normal(0.35, 0.18, size=n_event), 0.02, 0.98)
    qui_s = np.clip(rng.normal(0.50, 0.18, size=20 * n_event), 0.02, 0.98)
    for_s = np.clip(rng.normal(0.65, 0.18, size=n_event), 0.02, 0.98)
    strain_flat = np.concatenate([res_s, qui_s, for_s]).astype(np.float32)
    labels_flat = np.concatenate(
        [
            np.full(res_s.size, 1, dtype=np.int16),
            np.full(qui_s.size, 2, dtype=np.int16),
            np.full(for_s.size, 3, dtype=np.int16),
        ]
    )
    remodelling = np.full((2, labels_flat.size, 1), 2, dtype=np.int16)
    remodelling[0, :, 0] = labels_flat
    baseline = np.zeros(remodelling.shape, dtype=np.float32)
    baseline[1, :, 0] = strain_flat
    baseline[0, :, 0] = strain_flat
    return remodelling, baseline


def _lower_upper_nonzero_mean(probability: np.ndarray, support: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(probability) & np.isfinite(support) & (probability > 0)
    lower = valid & (support <= 50.0)
    upper = valid & (support >= 50.0)
    lower_mean = float(np.mean(probability[lower])) if np.any(lower) else 0.0
    upper_mean = float(np.mean(probability[upper])) if np.any(upper) else 0.0
    return lower_mean, upper_mean


def test_surface_dilated_event_extraction_geometry() -> None:
    remodelling = np.full((3, 3, 3), 2, dtype=np.int16)
    remodelling[1, 1, 1] = 3
    remodelling[0, 0, 0] = 1
    strain = np.ones((3, 3, 3), dtype=np.float32)

    labels, _strain, counts = mechreg_module._extract_surface_dilated_events(
        remodelling,
        strain,
        resorption_label=1,
        quiescence_label=2,
        formation_label=3,
        cap_percentile=99.0,
    )

    assert counts["n_surface_voxels"] == 25
    assert counts["n_sampled_voxels"] == 25
    assert np.count_nonzero(labels == 3) == 6
    assert np.count_nonzero(labels == 1) == 3
    assert counts["surface_event_mapping"] == "symmetric_surface_cancel_overlap"


def test_mechanoregulation_requires_matching_baseline_sed_shape() -> None:
    remodelling = np.full((3, 3, 3), 2, dtype=np.int16)
    baseline = np.ones((2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="baseline_strain shape must match remodelling_image"):
        mechanoregulation(remodelling_image=remodelling, baseline_strain=baseline)


def test_mechanoregulation_requires_baseline_sed() -> None:
    remodelling = np.full((3, 3, 3), 2, dtype=np.int16)

    with pytest.raises(ValueError, match="baseline_strain is required"):
        mechanoregulation(remodelling_image=remodelling, baseline_strain=None)


def test_derive_remodelling_labels_requires_binary_flip_and_density_change() -> None:
    baseline = np.zeros((5, 5, 5), dtype=np.float32)
    followup = np.zeros_like(baseline)
    baseline_seg = np.zeros_like(baseline, dtype=np.uint8)
    followup_seg = np.zeros_like(baseline, dtype=np.uint8)

    baseline[1, 1, 1] = 900.0
    baseline_seg[1, 1, 1] = 1

    followup[3, 3, 3] = 900.0
    followup_seg[3, 3, 3] = 1

    # These have only one half of the Timelapsed rule and should stay background.
    baseline[1, 3, 1] = 900.0
    baseline_seg[1, 3, 1] = 1
    followup[1, 3, 1] = 500.0
    followup_seg[1, 3, 1] = 1
    followup[3, 1, 3] = 900.0

    labels = derive_remodelling_labels_from_density(
        baseline,
        followup,
        baseline_segmentation=baseline_seg,
        followup_segmentation=followup_seg,
        density_threshold=225.0,
        gaussian_sigma=0.0,
        cluster_size=1,
    )

    assert labels[1, 1, 1] == 1
    assert labels[3, 3, 3] == 3
    assert labels[1, 3, 1] == 2
    assert labels[3, 1, 3] == 0


def test_mechanoregulation_synthetic_trends() -> None:
    remodelling, baseline = _make_synthetic_case()

    out = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=250,
        seed=0,
        return_full=True,
    )

    assert out.orf > 1.0
    assert out.settings["resorption_or_definition"] == "decreasing_strain"
    assert out.orr > 1.0
    assert out.orr_decreasing_strain == out.orr
    assert out.orr_increasing_strain < 1.0
    assert out.orf_ci[0] <= out.orf <= out.orf_ci[1]
    assert out.orr_ci[0] <= out.orr <= out.orr_ci[1]

    pf = np.asarray(out.conditional_curves["F"]["mean"], dtype=np.float64)
    pq = np.asarray(out.conditional_curves["Q"]["mean"], dtype=np.float64)
    pr = np.asarray(out.conditional_curves["R"]["mean"], dtype=np.float64)
    valid = (pf + pq + pr) > 0
    assert np.allclose((pf + pq + pr)[valid], 1.0)
    support = np.asarray(out.conditional_curves["strain"], dtype=np.float64)
    pf_low, pf_high = _lower_upper_nonzero_mean(pf, support)
    pr_low, pr_high = _lower_upper_nonzero_mean(pr, support)
    assert pf_high > pf_low
    assert pr_low > pr_high


def test_mechanoregulation_gaussian_surface_case_sound_or_and_curves() -> None:
    remodelling, baseline = _make_gaussian_surface_case(quiescence_factor=5, strain_scale=1.0)

    out = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=200,
        seed=0,
        return_full=True,
    )

    # Check expected class imbalance in sampled surface voxels.
    assert out.sample_counts["n_quiescence"] > (4 * out.sample_counts["n_formation"])
    assert out.sample_counts["n_quiescence"] > (4 * out.sample_counts["n_resorption"])

    # Sound OR behavior in legacy convention (ORR for decreasing strain).
    assert np.isfinite(out.orf)
    assert np.isfinite(out.orr)
    assert out.orf > 1.0
    assert out.orr > 1.0

    # ORs are reported per one normalized percentage point of SED.
    assert 1.005 < out.orf < 1.05
    assert 1.005 < out.orr < 1.05

    p_f = np.asarray(out.conditional_curves["F"]["mean"], dtype=np.float64)
    p_q = np.asarray(out.conditional_curves["Q"]["mean"], dtype=np.float64)
    p_r = np.asarray(out.conditional_curves["R"]["mean"], dtype=np.float64)
    valid = (p_f + p_q + p_r) > 0
    assert np.allclose((p_f + p_q + p_r)[valid], 1.0)
    support = np.asarray(out.conditional_curves["strain"], dtype=np.float64)
    pf_low, pf_high = _lower_upper_nonzero_mean(p_f, support)
    pr_low, pr_high = _lower_upper_nonzero_mean(p_r, support)
    assert pf_high > pf_low
    assert pr_low > pr_high


def test_conditional_curve_support_is_normalized_percent_linear() -> None:
    remodelling, baseline = _make_gaussian_surface_case(quiescence_factor=5, strain_scale=4.0)

    out = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=50,
        seed=0,
        return_full=True,
    )

    support = np.asarray(out.conditional_curves["strain"], dtype=np.float64)
    assert support[0] == 0.5
    assert support[-1] == 99.5
    assert np.allclose(np.diff(support), 1.0)
    assert out.conditional_curves["x_axis"] == "normalized_percent_linear"
    assert out.conditional_curves["ccr"]["max"] > 0.0
    assert len(out.conditional_curves["ccr"]["threshold_values"]) == 2
    assert out.binned_odds_diagnostics["x_axis"] == "normalized_percent_linear"
    assert out.binned_odds_diagnostics["ccr"]["max"] == out.conditional_curves["ccr"]["max"]
    assert out.binned_odds_diagnostics["source"] == "class_prevalence_normalized_bin_odds"


def test_binned_odds_masks_sparse_edge_bins() -> None:
    labels = np.array([3, 2, *([2] * 40), *([1] * 40)], dtype=np.int16)
    strain = np.array([0.02, 0.03, *np.linspace(0.45, 0.55, 40), *np.linspace(0.75, 0.85, 40)], dtype=np.float64)

    odds = mechreg_module._compute_binned_odds(
        labels,
        strain,
        resorption_label=1,
        quiescence_label=2,
        formation_label=3,
        n_bins=4,
        normalisation_value=1.0,
        min_bin_count=5,
        min_class_count=3,
    )

    assert np.isnan(odds["formation_or"][0])
    assert odds["counts"][0]["formation"] == 1
    assert odds["counts"][0]["total"] == 2


def test_conditional_curves_are_smooth_model_curves_not_histogram_zigzags() -> None:
    remodelling, baseline = _make_gaussian_surface_case(quiescence_factor=5, strain_scale=1.0)

    out = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=50,
        seed=0,
        return_full=True,
    )

    p_f = np.asarray(out.conditional_curves["F"]["mean"], dtype=np.float64)
    p_q = np.asarray(out.conditional_curves["Q"]["mean"], dtype=np.float64)
    p_r = np.asarray(out.conditional_curves["R"]["mean"], dtype=np.float64)
    assert out.conditional_curves["curve_type"] == "logistic_odds_normalized_percent_model"
    assert "schulte" in out.conditional_curves
    assert "low" in out.conditional_curves["F"]
    assert "high" in out.conditional_curves["R"]
    assert np.all(np.diff(p_f) >= -1e-8)
    assert np.all(np.diff(p_r) <= 1e-8)
    assert np.argmax(p_q) not in {0, p_q.size - 1}


def test_conditional_curves_are_class_balanced_under_large_quiescent_imbalance() -> None:
    remodelling, baseline = _make_overlapping_imbalanced_surface_case()

    out = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=50,
        seed=0,
        return_full=True,
    )

    p_f = np.asarray(out.conditional_curves["F"]["mean"], dtype=np.float64)
    p_q = np.asarray(out.conditional_curves["Q"]["mean"], dtype=np.float64)
    p_r = np.asarray(out.conditional_curves["R"]["mean"], dtype=np.float64)
    support = np.asarray(out.conditional_curves["strain"], dtype=np.float64)

    assert np.max(p_f[support >= 90.0]) > 0.50
    assert np.max(p_r[support <= 10.0]) > 0.50
    assert np.argmax(p_q) not in {0, p_q.size - 1}
    assert p_q[np.argmax(p_q)] > 0.35


def test_mechanoregulation_analysis_mask_limits_surface_sampling() -> None:
    remodelling = np.full((5, 5, 5), 2, dtype=np.int16)
    baseline = np.ones((5, 5, 5), dtype=np.float32)
    baseline[2, 2, 2] = 10.0
    remodelling[2, 2, 2] = 3
    remodelling[4, 4, 4] = 1

    mask = np.zeros_like(remodelling, dtype=np.uint8)
    mask[:3, :3, :3] = 1

    labels, _strain, counts = mechreg_module._extract_surface_dilated_events(
        remodelling,
        baseline,
        mask_xyz=mask,
        resorption_label=1,
        quiescence_label=2,
        formation_label=3,
        cap_percentile=99.0,
    )

    assert counts["n_eval_voxels"] == int(np.count_nonzero(mask))
    assert counts["n_resorption"] == 0
    assert counts["n_formation"] > 0
    assert not np.any(labels == 1)


def test_formation_events_are_projected_to_baseline_surface() -> None:
    remodelling = np.full((5, 5, 5), 2, dtype=np.int16)
    strain = np.ones((5, 5, 5), dtype=np.float32)

    # Newly formed tissue can have near-zero baseline strain at its own voxel.
    # It should mark adjacent baseline surface voxels, not be sampled directly.
    remodelling[0, 2, 2] = 3
    strain[0, 2, 2] = 0.001

    labels, sampled_strain, counts = mechreg_module._extract_surface_dilated_events(
        remodelling,
        strain,
        resorption_label=1,
        quiescence_label=2,
        formation_label=3,
        cap_percentile=99.0,
    )

    assert counts["n_formation"] > 0
    assert not np.any((labels == 3) & (sampled_strain < 0.01))


def test_symmetric_projection_excludes_direct_resorption_voxels() -> None:
    remodelling = np.full((5, 5, 5), 2, dtype=np.int16)
    strain = np.ones((5, 5, 5), dtype=np.float32)

    remodelling[0, 2, 2] = 1
    strain[0, 2, 2] = 0.001

    labels, sampled_strain, counts = mechreg_module._extract_surface_dilated_events(
        remodelling,
        strain,
        resorption_label=1,
        quiescence_label=2,
        formation_label=3,
        cap_percentile=99.0,
    )

    assert counts["n_resorption"] > 0
    assert not np.any((labels == 1) & (sampled_strain < 0.01))


def test_overlapping_projected_events_cancel_to_quiescence() -> None:
    remodelling = np.full((5, 5, 5), 2, dtype=np.int16)
    strain = np.ones((5, 5, 5), dtype=np.float32)
    remodelling[0, 2, 1] = 1
    remodelling[0, 2, 3] = 3

    labels, _sampled_strain, counts = mechreg_module._extract_surface_dilated_events(
        remodelling,
        strain,
        resorption_label=1,
        quiescence_label=2,
        formation_label=3,
        cap_percentile=99.0,
    )

    assert counts["n_cancelled_overlap"] == 1
    assert np.count_nonzero(labels == 1) == 4
    assert np.count_nonzero(labels == 3) == 4


def test_mechanoregulation_resorption_or_definitions_are_reciprocal() -> None:
    remodelling, baseline = _make_gaussian_surface_case(quiescence_factor=5, strain_scale=1.0)

    out_inc = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=120,
        seed=0,
        resorption_or_definition="increasing_strain",
        return_full=True,
    )
    out_dec = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=120,
        seed=0,
        resorption_or_definition="decreasing_strain",
        return_full=True,
    )

    assert np.isclose(out_inc.orf, out_dec.orf, rtol=1e-12, atol=0.0)
    assert np.isclose(out_dec.orr, 1.0 / out_inc.orr, rtol=1e-12, atol=0.0)
    assert np.isclose(out_dec.orr_ci[0], 1.0 / out_inc.orr_ci[1], rtol=1e-12, atol=0.0)
    assert np.isclose(out_dec.orr_ci[1], 1.0 / out_inc.orr_ci[0], rtol=1e-12, atol=0.0)
    assert np.isclose(out_inc.orr_increasing_strain, out_dec.orr_increasing_strain, rtol=1e-12, atol=0.0)
    assert np.isclose(out_inc.orr_decreasing_strain, out_dec.orr_decreasing_strain, rtol=1e-12, atol=0.0)
    assert np.isclose(out_dec.orr_decreasing_strain, 1.0 / out_dec.orr_increasing_strain, rtol=1e-12, atol=0.0)


def test_mechanoregulation_api_contract() -> None:
    remodelling, baseline = _make_synthetic_case()
    orf, orr = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=100,
        seed=0,
    )
    assert isinstance(orf, float)
    assert isinstance(orr, float)

    full = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=100,
        seed=0,
        return_full=True,
    )
    assert isinstance(full, MechanoregulationResult)
    for field in (
        "orf",
        "orr",
        "orf_ci",
        "orr_ci",
        "orr_increasing_strain",
        "orr_decreasing_strain",
        "orr_increasing_strain_ci",
        "orr_decreasing_strain_ci",
        "pvalue_form",
        "pvalue_res",
        "conditional_curves",
        "binned_odds_diagnostics",
        "sample_counts",
        "settings",
        "plot_paths",
    ):
        assert hasattr(full, field)


def test_analysis_module_no_longer_exposes_load_estimation() -> None:
    assert not hasattr(mechreg_module, "load_estimation")


def test_mechanoregulation_bootstrap_is_deterministic() -> None:
    remodelling, baseline = _make_synthetic_case()

    out1 = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=250,
        seed=123,
        return_full=True,
    )
    out2 = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=250,
        seed=123,
        return_full=True,
    )

    assert out1.orf == out2.orf
    assert out1.orr == out2.orr
    assert out1.orf_ci == out2.orf_ci
    assert out1.orr_ci == out2.orr_ci
    assert out1.conditional_curves == out2.conditional_curves


def test_mechanoregulation_plot_behavior(tmp_path) -> None:
    remodelling, baseline = _make_synthetic_case()

    _ = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=50,
        seed=0,
        work_dir=tmp_path,
        run_name="plot_off",
        plot=False,
        return_full=True,
    )
    assert not list(tmp_path.glob("plot_off_*.png"))

    mpl = __import__("pytest").importorskip("matplotlib")
    assert mpl is not None

    out = mechanoregulation(
        remodelling_image=remodelling,
        baseline_strain=baseline,
        n_boot=50,
        seed=0,
        work_dir=tmp_path,
        run_name="plot_on",
        plot=True,
        return_full=True,
    )
    assert out.plot_paths is not None
    assert "schulte_binned_curves" in out.plot_paths
    for path in out.plot_paths.values():
        assert path.exists()
