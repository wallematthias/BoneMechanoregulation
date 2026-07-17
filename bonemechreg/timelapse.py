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


def _subject_id_from_path(path: Path) -> str:
    """Return the first BIDS-like subject folder or filename token."""
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    match = re.search(r"(^|_)sub-(?P<subject>[^_]+)", path.name)
    if match is not None:
        return f"sub-{match.group('subject')}"
    return "unknown"


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


def _existing_candidate_roots(root: Path) -> tuple[Path, ...]:
    """Return possible TimelapsedHRpQCT result roots for a selected folder."""
    candidates = (
        root / "derivatives" / "TimelapsedHRpQCT",
        root,
        root / "TimelapsedHRpQCT",
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_dir():
            continue
        unique.append(candidate)
        seen.add(candidate)
    return tuple(unique)


def _glob_timelapsed_remodelling(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Collect remodelling paths from project, derivative, subject, or site roots."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for candidate in sorted(root.glob(pattern)):
            if candidate in seen:
                continue
            paths.append(candidate)
            seen.add(candidate)
    return paths


def discover_timelapse_cases(dataset_root: str | Path) -> list[TimelapseCase]:
    """Return all pairwise ``t0`` remodelling cases below a Timelapsed result root.

    ``dataset_root`` may be the project root containing
    ``derivatives/TimelapsedHRpQCT``, the ``TimelapsedHRpQCT`` derivative root
    itself, a selected ``sub-*`` folder, or a selected ``site-*`` folder.
    """
    root = Path(dataset_root).expanduser().resolve()
    cases: list[TimelapseCase] = []
    seen_cases: set[Path] = set()
    for result_root in _existing_candidate_roots(root):
        for remodelling_path in _glob_timelapsed_remodelling(
            result_root,
            (
                "sub-*/analysis/pairwise_t0/**/*remodelling*.nii.gz",
                "analysis/pairwise_t0/**/*remodelling*.nii.gz",
            ),
        ):
            if remodelling_path in seen_cases:
                continue
            baseline_path = _find_baseline_image(remodelling_path)
            if baseline_path is None:
                continue
            seen_cases.add(remodelling_path)
            cases.append(
                TimelapseCase(
                    subject_id=_subject_id_from_path(remodelling_path),
                    case_id=remodelling_path.parent.name,
                    baseline_image_path=baseline_path,
                    remodelling_image_path=remodelling_path,
                    output_dir=remodelling_path.parent / "mechanoregulation",
                )
            )
        for remodelling_path in _glob_timelapsed_remodelling(
            result_root,
            (
                "sub-*/site-*/analysis/visualize/*remodelling*.nii.gz",
                "site-*/analysis/visualize/*remodelling*.nii.gz",
                "analysis/visualize/*remodelling*.nii.gz",
            ),
        ):
            if remodelling_path in seen_cases:
                continue
            baseline_path = _find_v2_baseline_image(remodelling_path)
            if baseline_path is None:
                continue
            seen_cases.add(remodelling_path)
            cases.append(
                TimelapseCase(
                    subject_id=_subject_id_from_path(remodelling_path),
                    case_id=remodelling_path.stem.replace(".nii", ""),
                    baseline_image_path=baseline_path,
                    remodelling_image_path=remodelling_path,
                    output_dir=remodelling_path.parents[2] / "mechanoregulation",
                    **_find_v2_native_inputs(remodelling_path),
                )
            )
    return cases


def available_case_rois(case: TimelapseCase) -> dict[str, Path | None]:
    """Return the ROI masks available for a Timelapsed case.

    ``full`` is always present. A ``None`` value means the full case is analyzed
    without an explicit mask because the source Timelapsed run did not provide
    one.
    """
    rois: dict[str, Path | None] = {"full": case.full_mask_path if case.full_mask_path and case.full_mask_path.exists() else None}
    if case.trab_mask_path is not None and case.trab_mask_path.exists():
        rois["trab"] = case.trab_mask_path
    if case.cort_mask_path is not None and case.cort_mask_path.exists():
        rois["cort"] = case.cort_mask_path
    return rois


def case_outputs(case: TimelapseCase, *, roi: str | None = None) -> dict[str, Path]:
    """Return standard output paths for SED, summary tables, and curve PNG."""
    stem = re.sub(r"_remodelling.*$", "", case.remodelling_image_path.name)
    analysis_stem = stem if roi is None else f"{stem}_roi-{str(roi).strip().lower()}"
    parosol_solve_dir = case.output_dir / "parosol_sed_solve" / stem
    mechanoregulation_run_dir = case.output_dir / "mechanoregulation_run_logs" / analysis_stem
    return {
        "material": case.output_dir / f"{stem}_baseline_material_labels.nii.gz",
        "sed": case.output_dir / f"{stem}_sed.nii.gz",
        "summary": case.output_dir / f"{analysis_stem}_mechanoregulation_summary.json",
        "csv": case.output_dir / f"{analysis_stem}_mechanoregulation_summary.csv",
        "curves": case.output_dir / f"{analysis_stem}_conditional_curves.png",
        "schulte_curves": case.output_dir / f"{analysis_stem}_schulte_binned_curves.png",
        "mechanoregulation_run_dir": mechanoregulation_run_dir,
        "mechanoregulation_command": mechanoregulation_run_dir / "mechanoregulation_command.txt",
        "mechanoregulation_stdout": mechanoregulation_run_dir / "mechanoregulation_stdout.log",
        "mechanoregulation_stderr": mechanoregulation_run_dir / "mechanoregulation_stderr.log",
        "mechanoregulation_exit_code": mechanoregulation_run_dir / "mechanoregulation_exit_code.txt",
        "parosol_solve_dir": parosol_solve_dir,
        "parosol_wrapper_command": parosol_solve_dir / "bonemechreg_parosol_command.txt",
        "parosol_wrapper_stdout": parosol_solve_dir / "bonemechreg_parosol_stdout.log",
        "parosol_wrapper_stderr": parosol_solve_dir / "bonemechreg_parosol_stderr.log",
        "parosol_wrapper_exit_code": parosol_solve_dir / "bonemechreg_parosol_exit_code.txt",
        "parosol_live_stdout": parosol_solve_dir / "parosol_live_stdout.log",
        "parosol_live_stderr": parosol_solve_dir / "parosol_live_stderr.log",
        "parosol_native_command": parosol_solve_dir / "parosol_command.txt",
        "parosol_native_stdout": parosol_solve_dir / "parosol_stdout.log",
        "parosol_native_stderr": parosol_solve_dir / "parosol_stderr.log",
    }
