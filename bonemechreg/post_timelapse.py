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


def _outputs_complete(outputs: dict[str, Path]) -> bool:
    """Return true when all expected files for one ROI already exist."""
    return (
        outputs["sed"].exists()
        and outputs["summary"].exists()
        and outputs["csv"].exists()
        and outputs["curves"].exists()
        and outputs["schulte_curves"].exists()
    )


def _case_outputs_complete(case: TimelapseCase) -> bool:
    """Return true when the SED and every available ROI summary exist."""
    for roi in available_case_rois(case):
        roi_complete = _outputs_complete(case_outputs(case, roi=roi))
        legacy_full_complete = roi == "full" and _outputs_complete(case_outputs(case))
        if not (roi_complete or legacy_full_complete):
            return False
    return True


def _assert_same_grid(reference: sitk.Image, candidate: sitk.Image, *, name: str) -> None:
    """Raise when two images cannot be compared voxel-by-voxel."""
    if (
        reference.GetSize() != candidate.GetSize()
        or reference.GetSpacing() != candidate.GetSpacing()
        or reference.GetOrigin() != candidate.GetOrigin()
        or reference.GetDirection() != candidate.GetDirection()
    ):
        raise ValueError(f"{name} does not share the remodelling image grid")


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

    if (overwrite and not reanalyze) or not outputs["sed"].exists():
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
        if verbose:
            print(f"[mechanoregulation] {case.case_id}: wrote {outputs['sed']}")
    elif verbose:
        print(f"[mechanoregulation] {case.case_id}: reusing existing baseline SED {outputs['sed']}")

    remodelling_img = sitk.ReadImage(str(case.remodelling_image_path))
    baseline_sed_img = sitk.ReadImage(str(outputs["sed"]))
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
            odds_model="clipped_sed_unit",
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
        if roi == "full":
            # Preserve the historical unsuffixed table/plot names for older
            # scripts while making the new ROI-scoped files first-class.
            write_mechanoregulation_summary(
                case=case,
                profile=profile,
                result=result,
                output_path=outputs["summary"],
                roi=roi,
            )
            write_mechanoregulation_summary_csv(result, outputs["csv"])
            for source_key in ("curves", "schulte_curves"):
                source = roi_outputs[source_key]
                target = outputs[source_key]
                if source.exists() and source != target:
                    target.write_bytes(source.read_bytes())
        if verbose:
            print(f"[mechanoregulation] {case.case_id}: wrote {roi_outputs['csv']}")


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
    cases = discover_timelapse_cases(dataset_root)
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
