"""Batch workflow for running mechanoregulation after TimelapsedHRpQCT.

This is the implementation behind ``mechanoregulation run``. It discovers
pairwise timelapse cases, solves missing baseline SED with ParOSol, runs the
surface-based mechanoregulation analysis, and writes one summary per case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from bonemechreg.mechreg import mechanoregulation
from bonemechreg.timelapse import TimelapseCase, case_outputs, discover_timelapse_cases
from bonemechreg.results import write_mechanoregulation_summary, write_mechanoregulation_summary_csv
from bonemechreg.parosol import solve_sed_to_file


def _outputs_complete(outputs: dict[str, Path]) -> bool:
    """Return true when all expected files for a case already exist."""
    return (
        outputs["sed"].exists()
        and outputs["summary"].exists()
        and outputs["csv"].exists()
        and outputs["curves"].exists()
        and outputs["schulte_curves"].exists()
    )


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


def _read_analysis_mask(case: TimelapseCase, remodelling_image: sitk.Image) -> sitk.Image | None:
    """Return the Timelapsed full ROI mask when available."""
    if case.full_mask_path is None:
        return None
    mask = sitk.ReadImage(str(case.full_mask_path))
    _assert_same_grid(remodelling_image, mask, name="full analysis mask")
    return mask


def _run_case(case: TimelapseCase, profile: str, overwrite: bool, *, verbose: bool = False) -> None:
    """Run SED solving and mechanoregulation for one TimelapsedHRpQCT case."""
    outputs = case_outputs(case)
    outputs["sed"].parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not outputs["sed"].exists():
        if verbose:
            print(f"[mechanoregulation] {case.case_id}: writing baseline material labels")
        material_path = _write_baseline_material_labels(case, outputs["material"])
        if verbose:
            print(f"[mechanoregulation] {case.case_id}: solving baseline SED with {profile}")
        solve_sed_to_file(
            material_image_path=material_path,
            output_path=outputs["sed"],
            profile=profile,
        )
        if verbose:
            print(f"[mechanoregulation] {case.case_id}: wrote {outputs['sed']}")
    elif verbose:
        print(f"[mechanoregulation] {case.case_id}: reusing existing baseline SED {outputs['sed']}")

    if verbose:
        print(f"[mechanoregulation] {case.case_id}: running surface mechanoregulation analysis")
    remodelling_img = sitk.ReadImage(str(case.remodelling_image_path))
    baseline_sed_img = sitk.ReadImage(str(outputs["sed"]))
    analysis_mask = _read_analysis_mask(case, remodelling_img)
    result = mechanoregulation(
        remodelling_image=remodelling_img,
        baseline_strain=baseline_sed_img,
        analysis_mask=analysis_mask,
        return_full=True,
        plot=True,
        work_dir=case.output_dir,
        run_name=outputs["curves"].name.replace("_conditional_curves.png", ""),
    )
    write_mechanoregulation_summary(
        case=case,
        profile=profile,
        result=result,
        output_path=outputs["summary"],
    )
    write_mechanoregulation_summary_csv(result, outputs["csv"])
    if verbose:
        print(f"[mechanoregulation] {case.case_id}: wrote {outputs['csv']}")


def run_post_timelapse_mechanoregulation(
    *,
    dataset_root: str | Path,
    profile: str,
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run mechanoregulation for every discovered pairwise timelapse case.

    Args:
        dataset_root: Dataset root containing ``derivatives/TimelapsedHRpQCT``.
        profile: ParOSol scanner/profile name passed directly to ``parosol-py``.
        overwrite: Recompute SED and summaries even when outputs already exist.
        dry_run: Only count cases; do not write files.
        verbose: Re-raise case failures instead of counting them.

    Returns:
        A small summary dictionary with discovered/processed/skipped/failed
        counts. The CLI formats this for terminal output.
    """
    cases = discover_timelapse_cases(dataset_root)
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
        outputs = case_outputs(case)
        if not overwrite and _outputs_complete(outputs):
            if verbose:
                print(f"[mechanoregulation] {case.case_id}: outputs complete, skipping")
            summary["skipped"] += 1
            continue
        try:
            _run_case(case, profile, overwrite, verbose=verbose)
        except Exception:
            summary["failed"] += 1
            if verbose:
                raise
            continue
        summary["processed"] += 1
    return summary
