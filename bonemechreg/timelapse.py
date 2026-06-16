"""Discovery helpers for TimelapsedHRpQCT pairwise outputs.

The package expects TimelapsedHRpQCT to have already produced pairwise
``t0`` remodelling images. This module finds those cases and defines the output
filenames used by the mechanoregulation addon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class TimelapseCase:
    """One pairwise TimelapsedHRpQCT case ready for mechanoregulation."""

    subject_id: str
    case_id: str
    baseline_image_path: Path
    remodelling_image_path: Path
    output_dir: Path
    baseline_segmentation_path: Path | None = None
    trab_mask_path: Path | None = None
    cort_mask_path: Path | None = None
    full_mask_path: Path | None = None


def _find_baseline_image(remodelling_path: Path) -> Path | None:
    """Find the baseline image living beside a remodelling label image."""
    parent = remodelling_path.parent
    for pattern in ("*pairwise_t0*image*.nii.gz", "*pairwise_t0*.nii.gz", "*.nii.gz"):
        for candidate in sorted(parent.glob(pattern)):
            if candidate == remodelling_path:
                continue
            if "remodelling" in candidate.name.lower():
                continue
            return candidate
    return None


def _find_v2_baseline_image(remodelling_path: Path) -> Path | None:
    """Find the fused baseline image for a current TimelapsedHRpQCT v2 output."""
    session_id = _v2_baseline_session(remodelling_path)
    if session_id is None:
        return None
    site_dir = remodelling_path.parents[2]
    session_dir = site_dir / "transformed_images" / f"ses-{session_id}"
    candidates = sorted(session_dir.glob("*_image_fused.nii.gz"))
    return candidates[0] if candidates else None


def _v2_baseline_session(remodelling_path: Path) -> str | None:
    """Return the baseline session token encoded in a v2 remodelling filename."""
    match = re.search(r"_t0-(?P<t0>[^_]+)_t1-(?P<t1>[^_]+)_", remodelling_path.name)
    if match is None:
        return None
    return match.group("t0")


def _find_first_existing(patterns: tuple[str, ...], *, root: Path) -> Path | None:
    """Find the first path matching one of several glob patterns."""
    for pattern in patterns:
        candidates = sorted(root.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def _find_v2_native_inputs(remodelling_path: Path) -> dict[str, Path | None]:
    """Find native baseline stack segmentation and compartment masks for v2.

    TimelapsedHRpQCT writes remodelling labels in the common baseline analysis
    grid. The native stack-level baseline segmentation has the same grid for
    regular single-stack runs and is the correct source for baseline mechanics.
    The transformed/fused images are registration products and are not used as
    mechanics geometry here.
    """
    session_id = _v2_baseline_session(remodelling_path)
    if session_id is None:
        return {
            "baseline_segmentation_path": None,
            "trab_mask_path": None,
            "cort_mask_path": None,
            "full_mask_path": None,
        }
    site_dir = remodelling_path.parents[2]
    stack_dir = site_dir / f"ses-{session_id}" / "stacks"
    return {
        "baseline_segmentation_path": _find_first_existing(("*_seg.nii.gz",), root=stack_dir),
        "trab_mask_path": _find_first_existing(("*_mask-trab.nii.gz",), root=stack_dir),
        "cort_mask_path": _find_first_existing(("*_mask-cort.nii.gz",), root=stack_dir),
        "full_mask_path": _find_first_existing(("*_mask-full.nii.gz",), root=stack_dir),
    }


def discover_timelapse_cases(dataset_root: str | Path) -> list[TimelapseCase]:
    """Return all pairwise ``t0`` remodelling cases below a dataset root."""
    root = Path(dataset_root).expanduser().resolve()
    derivative_root = root / "derivatives" / "TimelapsedHRpQCT"
    if not derivative_root.exists():
        return []

    cases: list[TimelapseCase] = []
    for remodelling_path in sorted(derivative_root.glob("sub-*/analysis/pairwise_t0/**/*remodelling*.nii.gz")):
        baseline_path = _find_baseline_image(remodelling_path)
        if baseline_path is None:
            continue
        subject_id = remodelling_path.relative_to(derivative_root).parts[0]
        cases.append(
            TimelapseCase(
                subject_id=subject_id,
                case_id=remodelling_path.parent.name,
                baseline_image_path=baseline_path,
                remodelling_image_path=remodelling_path,
                output_dir=remodelling_path.parent / "mechanoregulation",
            )
        )
    for remodelling_path in sorted(derivative_root.glob("sub-*/site-*/analysis/visualize/*remodelling*.nii.gz")):
        baseline_path = _find_v2_baseline_image(remodelling_path)
        if baseline_path is None:
            continue
        relative_parts = remodelling_path.relative_to(derivative_root).parts
        subject_id = relative_parts[0]
        cases.append(
            TimelapseCase(
                subject_id=subject_id,
                case_id=remodelling_path.stem.replace(".nii", ""),
                baseline_image_path=baseline_path,
                remodelling_image_path=remodelling_path,
                output_dir=remodelling_path.parents[2] / "mechanoregulation",
                **_find_v2_native_inputs(remodelling_path),
            )
        )
    return cases


def case_outputs(case: TimelapseCase) -> dict[str, Path]:
    """Return standard output paths for SED, summary tables, and curve PNG."""
    stem = re.sub(r"_remodelling.*$", "", case.remodelling_image_path.name)
    return {
        "material": case.output_dir / f"{stem}_baseline_material_labels.nii.gz",
        "sed": case.output_dir / f"{stem}_sed.nii.gz",
        "summary": case.output_dir / f"{stem}_mechanoregulation_summary.json",
        "csv": case.output_dir / f"{stem}_mechanoregulation_summary.csv",
        "curves": case.output_dir / f"{stem}_conditional_curves.png",
        "schulte_curves": case.output_dir / f"{stem}_schulte_binned_curves.png",
    }
