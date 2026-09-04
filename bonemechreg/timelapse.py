"""Discovery helpers for TimelapsedHRpQCT pairwise outputs.

The package expects TimelapsedHRpQCT to have already produced pairwise
``t0`` remodelling images. This module finds those cases and defines the output
filenames used by the mechanoregulation addon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from bone_imaging_derivatives import discover_manifests, find_records
from bone_imaging_derivatives.layout import record_output_path


@dataclass(frozen=True)
class TimelapseCase:
    """One pairwise TimelapsedHRpQCT case ready for mechanoregulation."""

    subject_id: str
    case_id: str
    baseline_image_path: Path
    remodelling_image_path: Path
    output_dir: Path
    site: str = ""
    baseline_session_id: str = ""
    followup_session_id: str = ""
    baseline_sed_path: Path | None = None
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
    """Return the first normalized subject token from a folder or filename."""
    for part in path.parts:
        if part.startswith("sub-"):
            return part[4:]
    match = re.search(r"(^|_)sub-(?P<subject>[^_]+)", path.name)
    if match is not None:
        return match.group("subject")
    return "unknown"


def _site_from_name(path: Path) -> str:
    """Return the normalized VOI/site token encoded in a Timelapse filename."""
    match = re.search(r"(?:^|_)voi-(?P<site>[^_]+)", path.name, flags=re.IGNORECASE)
    if match is not None:
        return match.group("site").lower()
    match = re.search(r"(?:^|_)site-(?P<site>[^_]+)", path.name, flags=re.IGNORECASE)
    if match is not None:
        return match.group("site").lower()
    return ""


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


def _strip_nii_gz_stem(path: Path) -> str:
    """Return a stable stem for single- and double-suffix medical images."""
    name = path.name
    for suffix in (".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd", ".aim", ".AIM"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _remodelling_case_id(path: Path) -> str:
    """Return the Timelapse pair identifier shared by images and analysis outputs."""
    return re.sub(r"_remodelling.*$", "", _strip_nii_gz_stem(path))


def _existing_candidate_roots(root: Path) -> tuple[Path, ...]:
    """Return possible TimelapsedHRpQCT result roots for a selected folder."""
    candidates = (
        root / "derivatives" / "Timelapse",
        root if root.name == "Timelapse" else root / "Timelapse",
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


def _find_current_baseline_image(remodelling_path: Path) -> Path | None:
    """Find the baseline-space image for the current normalized Timelapse layout."""
    session_id = _v2_baseline_session(remodelling_path)
    site = _site_from_name(remodelling_path)
    if session_id is None:
        return None
    subject_dir = _subject_dir_from_remodelling(remodelling_path)
    if subject_dir is None:
        return None
    transformed_dir = subject_dir / f"ses-{session_id}" / "xct" / "transformed"
    patterns = []
    if site:
        patterns.append(f"*_ses-{session_id}_voi-{site}_image-fused.nii.gz")
    patterns.append(f"*_ses-{session_id}_*_image-fused.nii.gz")
    patterns.append("*_image-fused.nii.gz")
    return _find_first_existing(tuple(patterns), root=transformed_dir)


def _find_current_native_inputs(remodelling_path: Path) -> dict[str, Path | None]:
    """Find baseline segmentation/ROI masks in the current Timelapse layout."""
    session_id = _v2_baseline_session(remodelling_path)
    site = _site_from_name(remodelling_path)
    subject_dir = _subject_dir_from_remodelling(remodelling_path)
    if session_id is None or subject_dir is None:
        return {
            "baseline_segmentation_path": None,
            "trab_mask_path": None,
            "cort_mask_path": None,
            "full_mask_path": None,
        }
    transformed_dir = subject_dir / f"ses-{session_id}" / "xct" / "transformed"
    stack_dir = subject_dir / f"ses-{session_id}" / "xct" / "stacks"
    analysis_dir = subject_dir / "xct" / "analysis"
    common_dir = analysis_dir / "common_regions"
    site_prefix = f"*_ses-{session_id}_voi-{site}" if site else f"*_ses-{session_id}_*"
    common_prefix = f"*_voi-{site}" if site else "*"
    return {
        "baseline_segmentation_path": _find_first_existing(
            (
                f"{site_prefix}_desc-seg_mask-fused.nii.gz",
                f"{site_prefix}_stack-*_seg.nii.gz",
                "*_desc-seg_mask-fused.nii.gz",
                "*_seg.nii.gz",
            ),
            root=transformed_dir,
        )
        or _find_first_existing((f"{site_prefix}_stack-*_seg.nii.gz", "*_seg.nii.gz"), root=stack_dir),
        "trab_mask_path": _find_first_existing(
            (
                f"{common_prefix}_desc-trab_common-alltimepoints.nii.gz",
                "*_desc-trab_common-alltimepoints.nii.gz",
            ),
            root=common_dir,
        )
        or _find_first_existing(
            (
                f"{site_prefix}_desc-trab_mask-fused.nii.gz",
                "*_desc-trab_mask-fused.nii.gz",
            ),
            root=transformed_dir,
        )
        or _find_first_existing((f"{site_prefix}_stack-*_mask-trab.nii.gz", "*_mask-trab.nii.gz"), root=stack_dir),
        "cort_mask_path": _find_first_existing(
            (
                f"{common_prefix}_desc-cort_common-alltimepoints.nii.gz",
                "*_desc-cort_common-alltimepoints.nii.gz",
            ),
            root=common_dir,
        )
        or _find_first_existing(
            (
                f"{site_prefix}_desc-cort_mask-fused.nii.gz",
                "*_desc-cort_mask-fused.nii.gz",
            ),
            root=transformed_dir,
        )
        or _find_first_existing((f"{site_prefix}_stack-*_mask-cort.nii.gz", "*_mask-cort.nii.gz"), root=stack_dir),
        "full_mask_path": _find_first_existing(
            (
                f"{common_prefix}_desc-full_common-alltimepoints.nii.gz",
                "*_desc-full_common-alltimepoints.nii.gz",
            ),
            root=common_dir,
        )
        or _find_first_existing(
            (
                f"{site_prefix}_desc-full_mask-fused.nii.gz",
                "*_desc-full_mask-fused.nii.gz",
            ),
            root=transformed_dir,
        )
        or _find_first_existing((f"{site_prefix}_stack-*_mask-full.nii.gz", "*_mask-full.nii.gz"), root=stack_dir),
    }


def _subject_dir_from_remodelling(remodelling_path: Path) -> Path | None:
    """Return the enclosing ``sub-*`` directory for current Timelapse outputs."""
    for parent in remodelling_path.parents:
        if parent.name.startswith("sub-"):
            return parent
    return None


def _dataset_root_from_timelapse_root(result_root: Path) -> Path:
    """Return the selected dataset root for a Timelapse derivative root."""
    if result_root.name == "Timelapse" and result_root.parent.name == "derivatives":
        return result_root.parent.parent
    return result_root


def _find_fea_sed(
    dataset_root: Path,
    *,
    subject_id: str,
    site: str,
    session_id: str,
    sed_profile: str | None,
) -> Path | None:
    """Find the baseline-space SED emitted by the FEA batch workflow."""
    maps_dir = dataset_root / "derivatives" / "FEA" / f"sub-{subject_id}" / f"ses-{session_id}" / "xct" / "maps"
    if not maps_dir.exists():
        return None
    profile_token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(sed_profile or "").strip()).strip("-")
    patterns = []
    if profile_token:
        return _find_first_existing((f"*_voi-{site}_desc-{profile_token}_map-sed.nii.gz",), root=maps_dir)
    if site:
        patterns.append(f"*_voi-{site}_*_map-sed.nii.gz")
    patterns.append("*_map-sed.nii.gz")
    return _find_first_existing(tuple(patterns), root=maps_dir)


def _manifest_baseline_path(record, records) -> Path | None:
    """Find the baseline image explicitly linked to one Timelapsed record."""
    by_id = {candidate.record_id: candidate for candidate in records}
    for record_id in record.inputs:
        candidate = by_id.get(record_id)
        if candidate is not None and candidate.role in {"source_image_view", "transformed_image"}:
            return candidate.path
    candidates = [
        candidate
        for candidate in records
        if candidate.role in {"source_image_view", "transformed_image"}
        and candidate.subject_id == record.subject_id
        and candidate.site == record.site
        and candidate.session_id == record.session_id
    ]
    return candidates[0].path if len(candidates) == 1 else None


def _discover_manifest_timelapse_cases(root: Path, *, sed_profile: str | None = None) -> list[TimelapseCase]:
    """Build Timelapsed cases from shared derivative manifests when available."""
    records = find_records(discover_manifests(root), derivative="Timelapse")
    cases: list[TimelapseCase] = []
    for record in records:
        if record.role not in {"remodelling_image", "remodelling_pairwise_image", "remodelling_pairwise_table"}:
            continue
        baseline_path = _manifest_baseline_path(record, records)
        if baseline_path is None:
            continue
        case_id = _remodelling_case_id(record.path)
        t0 = record.metadata.get("t0", record.session_id) if isinstance(record.metadata, dict) else record.session_id
        t1 = record.metadata.get("t1", "") if isinstance(record.metadata, dict) else ""
        subject_id = str(record.subject_id).removeprefix("sub-")
        output_dir = record_output_path(
            root,
            "Mechanoregulation",
            subject_id,
            record.site,
            "runs",
            case_id,
        )
        cases.append(
            TimelapseCase(
                subject_id=subject_id,
                case_id=case_id,
                baseline_image_path=baseline_path,
                remodelling_image_path=record.path,
                output_dir=output_dir,
                site=str(record.site or ""),
                baseline_session_id=str(t0 or ""),
                followup_session_id=str(t1 or ""),
                baseline_sed_path=_find_fea_sed(
                    root,
                    subject_id=subject_id,
                    site=str(record.site or ""),
                    session_id=str(t0 or ""),
                    sed_profile=sed_profile,
                ),
            )
        )
    return cases


def discover_timelapse_cases(dataset_root: str | Path, *, sed_profile: str | None = None) -> list[TimelapseCase]:
    """Return all pairwise ``t0`` remodelling cases below a Timelapsed result root.

    ``dataset_root`` may be the project root containing
    ``derivatives/TimelapsedHRpQCT``, the ``TimelapsedHRpQCT`` derivative root
    itself, a selected ``sub-*`` folder, or a selected ``site-*`` folder.
    """
    root = Path(dataset_root).expanduser().resolve()
    manifest_cases = _discover_manifest_timelapse_cases(root, sed_profile=sed_profile)
    if manifest_cases:
        return manifest_cases
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
                    site=_site_from_name(remodelling_path),
                    baseline_session_id=_v2_baseline_session(remodelling_path) or "",
                )
            )
        current_dataset_root = _dataset_root_from_timelapse_root(result_root)
        for remodelling_path in _glob_timelapsed_remodelling(
            result_root,
            (
                "sub-*/xct/analysis/visualize/*remodelling*.nii.gz",
                "xct/analysis/visualize/*remodelling*.nii.gz",
                "analysis/visualize/*remodelling*.nii.gz",
            ),
        ):
            if remodelling_path in seen_cases:
                continue
            baseline_path = _find_current_baseline_image(remodelling_path)
            if baseline_path is None:
                continue
            subject_id = _subject_id_from_path(remodelling_path)
            site = _site_from_name(remodelling_path)
            t0 = _v2_baseline_session(remodelling_path) or ""
            t1_match = re.search(r"_t1-(?P<t1>[^_]+)_", remodelling_path.name, flags=re.IGNORECASE)
            t1 = t1_match.group("t1") if t1_match else ""
            seen_cases.add(remodelling_path)
            case_id = _remodelling_case_id(remodelling_path)
            cases.append(
                TimelapseCase(
                    subject_id=subject_id,
                    case_id=case_id,
                    baseline_image_path=baseline_path,
                    remodelling_image_path=remodelling_path,
                    output_dir=record_output_path(current_dataset_root, "Mechanoregulation", subject_id, site, "runs", case_id),
                    site=site,
                    baseline_session_id=t0,
                    followup_session_id=t1,
                    baseline_sed_path=_find_fea_sed(
                        current_dataset_root,
                        subject_id=subject_id,
                        site=site,
                        session_id=t0,
                        sed_profile=sed_profile,
                    ),
                    **_find_current_native_inputs(remodelling_path),
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
                    case_id=_remodelling_case_id(remodelling_path),
                    baseline_image_path=baseline_path,
                    remodelling_image_path=remodelling_path,
                    output_dir=remodelling_path.parents[2] / "mechanoregulation",
                    site=_site_from_name(remodelling_path),
                    baseline_session_id=_v2_baseline_session(remodelling_path) or "",
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
    def existing_path(value: Path | str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.exists() else None

    rois: dict[str, Path | None] = {"full": existing_path(case.full_mask_path)}
    trab_path = existing_path(case.trab_mask_path)
    cort_path = existing_path(case.cort_mask_path)
    if trab_path is not None:
        rois["trab"] = trab_path
    if cort_path is not None:
        rois["cort"] = cort_path
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
        "surface_events": case.output_dir / f"{analysis_stem}_surface-events.nii.gz",
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
