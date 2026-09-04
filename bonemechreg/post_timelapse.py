"""Batch workflow for running mechanoregulation after TimelapsedHRpQCT.

This is the implementation behind ``mechanoregulation run``. It discovers
pairwise timelapse cases, solves missing baseline SED with ParOSol, runs the
surface-based mechanoregulation analysis, and writes one summary per case.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import SimpleITK as sitk

from bonemechreg.mechreg import mechanoregulation
from bonemechreg.timelapse import TimelapseCase, available_case_rois, case_outputs, discover_timelapse_cases
from bonemechreg.results import write_mechanoregulation_summary, write_mechanoregulation_summary_csv
from bonemechreg.parosol import solve_sed_to_file


def _write_surface_event_image(result: Any, reference: sitk.Image, output_path: Path) -> None:
    """Write analysed formation/resorption surface labels on the reference grid."""
    surface = getattr(result, "surface_event_image", None)
    if surface is None:
        return
    arr_xyz = np.asarray(surface, dtype=np.uint8)
    if arr_xyz.shape != tuple(reference.GetSize()):
        raise ValueError("surface_event_image shape must match remodelling image grid")
    image = sitk.GetImageFromArray(np.transpose(arr_xyz, (2, 1, 0)))
    image.CopyInformation(reference)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))


def _outputs_complete(outputs: dict[str, Path], *, sed_path: Path | None = None) -> bool:
    """Return true when all expected files for one ROI already exist."""
    sed_complete = bool(sed_path is not None and Path(sed_path).exists()) or outputs["sed"].exists()
    return (
        sed_complete
        and outputs["summary"].exists()
        and outputs["csv"].exists()
        and outputs["surface_events"].exists()
        and outputs["curves"].exists()
        and outputs["schulte_curves"].exists()
    )


def _case_outputs_complete(case: TimelapseCase) -> bool:
    """Return true when the SED and every available ROI summary exist."""
    for roi in available_case_rois(case):
        if not _outputs_complete(case_outputs(case, roi=roi), sed_path=case.baseline_sed_path):
            return False
    return True


def _assert_same_grid(reference: sitk.Image, candidate: sitk.Image, *, name: str) -> None:
    """Raise when two images cannot be compared voxel-by-voxel."""
    if _same_grid(reference, candidate):
        return
    raise ValueError(f"{name} does not share the remodelling image grid")


def _same_grid(reference: sitk.Image, candidate: sitk.Image) -> bool:
    """Return true when two images share voxel indexing and physical metadata."""
    if (
        reference.GetSize() == candidate.GetSize()
        and reference.GetSpacing() == candidate.GetSpacing()
        and reference.GetOrigin() == candidate.GetOrigin()
        and reference.GetDirection() == candidate.GetDirection()
    ):
        return True
    return False


def _nearest_mask_on_grid(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample a binary ROI mask onto the remodelling image grid."""
    return sitk.Resample(
        sitk.Cast(mask > 0, sitk.sitkUInt8),
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def _linear_scalar_on_grid(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample a scalar image onto the remodelling image grid."""
    if image.GetSize() == reference.GetSize() and np.allclose(image.GetSpacing(), reference.GetSpacing()):
        aligned = sitk.GetImageFromArray(sitk.GetArrayFromImage(image).astype(np.float32, copy=False))
        aligned.CopyInformation(reference)
        return aligned
    return sitk.Resample(
        sitk.Cast(image, sitk.sitkFloat32),
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def _binary_array(path: Path, *, reference: sitk.Image, name: str) -> np.ndarray:
    """Read a binary image and confirm it is aligned with the remodelling grid."""
    image = sitk.ReadImage(str(path))
    _assert_same_grid(reference, image, name=name)
    return sitk.GetArrayFromImage(image) > 0


def _write_baseline_material_labels(case: TimelapseCase, output_path: Path) -> Path:
    """Create the ParOSol material image from native baseline segmentation.

    The material map is deliberately derived from baseline state only:

    - ``100`` = trabecular baseline bone
    - ``127`` = cortical baseline bone
    - ``0`` = background, marrow, and future formation sites

    Timelapsed remodelling labels are not used as mechanics input; they are
    read later only as the F/Q/R outcome map.
    """
    if case.baseline_segmentation_path is None:
        raise ValueError("Timelapsed baseline segmentation is required for the mechanics solve")
    remodelling_image = sitk.ReadImage(str(case.remodelling_image_path))
    segmentation_image = sitk.ReadImage(str(case.baseline_segmentation_path))
    _assert_same_grid(remodelling_image, segmentation_image, name="baseline segmentation")

    bone = sitk.GetArrayFromImage(segmentation_image) > 0
    material = np.zeros(bone.shape, dtype=np.uint8)

    if case.trab_mask_path is not None:
        trab = _binary_array(case.trab_mask_path, reference=remodelling_image, name="trabecular mask")
        material[bone & trab] = 100
    if case.cort_mask_path is not None:
        cort = _binary_array(case.cort_mask_path, reference=remodelling_image, name="cortical mask")
        material[bone & cort] = 127

    # Some datasets provide only a binary segmentation. Treat unassigned bone
    # as trabecular so it remains mechanically active rather than disappearing.
    material[bone & (material == 0)] = 100

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image = sitk.GetImageFromArray(material, isVector=False)
    output_image.CopyInformation(remodelling_image)
    sitk.WriteImage(output_image, str(output_path))
    return output_path


def _read_analysis_mask(mask_path: Path | None, remodelling_image: sitk.Image, *, roi: str) -> sitk.Image | None:
    """Return one Timelapsed ROI mask when available."""
    if mask_path is None:
        return None
    mask = sitk.ReadImage(str(mask_path))
    if not _same_grid(remodelling_image, mask):
        mask = _nearest_mask_on_grid(mask, remodelling_image)
    _assert_same_grid(remodelling_image, mask, name=f"{roi} analysis mask")
    return mask


def _run_case(
    case: TimelapseCase,
    profile: str,
    overwrite: bool,
    *,
    reanalyze: bool = False,
    n_boot: int = 100,
    bootstrap_sampling_perc: float = 100.0,
    verbose: bool = False,
) -> None:
    """Run SED solving and mechanoregulation for one TimelapsedHRpQCT case."""
    outputs = case_outputs(case)
    outputs["sed"].parent.mkdir(parents=True, exist_ok=True)

    sed_path = case.baseline_sed_path if case.baseline_sed_path and Path(case.baseline_sed_path).exists() else outputs["sed"]
    if case.baseline_sed_path and Path(case.baseline_sed_path).exists():
        if verbose:
            print(f"[mechanoregulation] {case.case_id}: reusing matched FEA SED {case.baseline_sed_path}")
    elif (overwrite and not reanalyze) or not outputs["sed"].exists():
        if verbose:
            print(f"[mechanoregulation] {case.case_id}: writing baseline material labels")
        material_path = _write_baseline_material_labels(case, outputs["material"])
        if verbose:
            print(f"[mechanoregulation] {case.case_id}: solving baseline SED with {profile}")
        solve_sed_to_file(
            material_image_path=material_path,
            output_path=outputs["sed"],
            profile=profile,
            debug_dir=outputs["parosol_solve_dir"],
        )
        sed_path = outputs["sed"]
        if verbose:
            print(f"[mechanoregulation] {case.case_id}: wrote {outputs['sed']}")
    elif verbose:
        print(f"[mechanoregulation] {case.case_id}: reusing existing baseline SED {outputs['sed']}")

    remodelling_img = sitk.ReadImage(str(case.remodelling_image_path))
    baseline_sed_img = sitk.ReadImage(str(sed_path))
    if not _same_grid(remodelling_img, baseline_sed_img):
        baseline_sed_img = _linear_scalar_on_grid(baseline_sed_img, remodelling_img)
    _assert_same_grid(remodelling_img, baseline_sed_img, name="baseline SED")
    for roi, mask_path in available_case_rois(case).items():
        roi_outputs = case_outputs(case, roi=roi)
        if verbose:
            print(
                f"[mechanoregulation] {case.case_id}: running {roi} surface mechanoregulation analysis "
                f"(n_boot={int(n_boot)})"
            )
        analysis_mask = _read_analysis_mask(mask_path, remodelling_img, roi=roi)
        analysis_start = perf_counter()
        result = mechanoregulation(
            remodelling_image=remodelling_img,
            baseline_strain=baseline_sed_img,
            analysis_mask=analysis_mask,
            n_boot=int(n_boot),
            bootstrap_sampling_perc=float(bootstrap_sampling_perc),
            surface_event_mapping="surface_dilation_resorption_wins",
            odds_model="normalized_percent",
            resorption_or_definition="decreasing_strain",
            return_full=True,
            plot=True,
            work_dir=case.output_dir,
            run_name=roi_outputs["curves"].name.replace("_conditional_curves.png", ""),
        )
        if verbose:
            print(
                f"[mechanoregulation] {case.case_id}: {roi} surface analysis finished "
                f"in {perf_counter() - analysis_start:.1f}s"
            )
        write_mechanoregulation_summary(
            case=case,
            profile=profile,
            result=result,
            output_path=roi_outputs["summary"],
            roi=roi,
        )
        write_mechanoregulation_summary_csv(result, roi_outputs["csv"])
        _write_surface_event_image(result, remodelling_img, roi_outputs["surface_events"])
        if verbose:
            print(f"[mechanoregulation] {case.case_id}: wrote {roi_outputs['csv']}")
            if roi_outputs["surface_events"].exists():
                print(f"[mechanoregulation] {case.case_id}: wrote {roi_outputs['surface_events']}")


def run_post_timelapse_case(
    case: TimelapseCase,
    *,
    profile: str,
    overwrite: bool = False,
    reanalyze: bool = False,
    n_boot: int = 100,
    bootstrap_sampling_perc: float = 100.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run mechanoregulation for a single discovered TimelapsedHRpQCT case."""
    outputs = case_outputs(case)
    summary: dict[str, Any] = {
        "discovered": 1,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "dry_run": False,
        "case_id": case.case_id,
        "output_dir": str(case.output_dir),
    }
    if not overwrite and not reanalyze and _case_outputs_complete(case):
        if verbose:
            print(f"[mechanoregulation] {case.case_id}: outputs complete, skipping")
        summary["skipped"] = 1
        return summary
    try:
        _run_case(
            case,
            profile,
            overwrite,
            reanalyze=bool(reanalyze),
            n_boot=int(n_boot),
            bootstrap_sampling_perc=float(bootstrap_sampling_perc),
            verbose=verbose,
        )
    except Exception:
        summary["failed"] = 1
        if verbose:
            raise
        return summary
    summary["processed"] = 1
    return summary


def run_post_timelapse_mechanoregulation(
    *,
    dataset_root: str | Path,
    profile: str,
    overwrite: bool = False,
    reanalyze: bool = False,
    dry_run: bool = False,
    case_id: str | None = None,
    n_boot: int = 100,
    bootstrap_sampling_perc: float = 100.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run mechanoregulation for every discovered pairwise timelapse case.

    Args:
        dataset_root: Dataset root containing ``derivatives/TimelapsedHRpQCT``.
        profile: ParOSol scanner/profile name passed directly to ``parosol-py``.
        overwrite: Recompute SED and summaries even when outputs already exist.
        reanalyze: Recompute mechanoregulation summaries and curve PNGs while
            reusing an existing SED when possible.
        dry_run: Only count cases; do not write files.
        case_id: Optional exact Timelapsed case identifier to run.
        n_boot: Number of class-balanced bootstrap replicates used for the
            mechanoregulation odds-ratio confidence intervals.
        bootstrap_sampling_perc: Percent of the smallest class sampled in each
            bootstrap replicate.
        verbose: Re-raise case failures instead of counting them.

    Returns:
        A small summary dictionary with discovered/processed/skipped/failed
        counts. The CLI formats this for terminal output.
    """
    cases = discover_timelapse_cases(dataset_root, sed_profile=profile)
    if case_id:
        cases = [case for case in cases if case.case_id == str(case_id)]
    summary: dict[str, Any] = {
        "discovered": len(cases),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return summary

    for case in cases:
        if not overwrite and not reanalyze and _case_outputs_complete(case):
            if verbose:
                print(f"[mechanoregulation] {case.case_id}: outputs complete, skipping")
            summary["skipped"] += 1
            continue
        try:
            _run_case(
                case,
                profile,
                overwrite,
                reanalyze=bool(reanalyze),
                n_boot=int(n_boot),
                bootstrap_sampling_perc=float(bootstrap_sampling_perc),
                verbose=verbose,
            )
        except Exception:
            summary["failed"] += 1
            if verbose:
                raise
            continue
        summary["processed"] += 1
    return summary
