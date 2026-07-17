"""Thin ParOSol wrapper used by the post-timelapse workflow.

BoneMechanoregulation does not implement finite-element mechanics itself. It
builds a baseline material-label image, asks the native ``parosol-py`` scanner
profile to solve the SED field, and then reads the exported
``fields/sed.nii.gz`` image.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk


def _run_parosol_profile(
    *, image_path: Path, profile: str, debug_dir: Path | None = None
) -> np.ndarray:
    """Run the configured ParOSol profile and return SED as a NumPy array."""
    return _run_parosol_profile_cli(
        image_path=image_path,
        profile=profile,
        debug_dir=debug_dir,
    )


def _run_parosol_profile_cli(
    *, image_path: Path, profile: str, debug_dir: Path | None = None
) -> np.ndarray:
    """Execute the native ``parosol-py`` profile shortcut.

    The input image must already be a material-label image in baseline space.
    The standard XtremeCTI/XtremeCTII profiles expect ``100`` for trabecular
    bone, ``127`` for cortical bone, and all other labels as non-bone.
    """
    if debug_dir is None:
        with TemporaryDirectory(prefix="bonemechreg_parosol_") as tmp:
            return _run_parosol_profile_in_dir(
                image_path=image_path,
                profile=profile,
                output_dir=Path(tmp) / "case",
            )
    return _run_parosol_profile_in_dir(
        image_path=image_path,
        profile=profile,
        output_dir=debug_dir,
    )


def _run_parosol_profile_in_dir(*, image_path: Path, profile: str, output_dir: Path) -> np.ndarray:
    """Run parosol-py in ``output_dir`` and persist wrapper-level diagnostics."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = _parosol_cli_command(
        image_path=image_path,
        profile=profile,
        output_dir=output_dir,
    )
    _write_wrapper_log(output_dir / "bonemechreg_parosol_command.txt", " ".join(argv) + "\n")
    stdout_path = output_dir / "bonemechreg_parosol_stdout.log"
    stderr_path = output_dir / "bonemechreg_parosol_stderr.log"
    live_stdout_path = output_dir / "parosol_live_stdout.log"
    live_stderr_path = output_dir / "parosol_live_stderr.log"
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            proc = subprocess.Popen(argv, stdout=stdout_handle, stderr=stderr_handle, text=True)
            returncode = proc.wait()
    except FileNotFoundError as exc:
        _write_wrapper_log(stdout_path, "")
        _write_wrapper_log(live_stdout_path, "")
        _write_wrapper_log(stderr_path, str(exc) + "\n")
        _write_wrapper_log(live_stderr_path, str(exc) + "\n")
        _write_wrapper_log(output_dir / "bonemechreg_parosol_exit_code.txt", "")
        raise RuntimeError(f"Could not start parosol-py subprocess; see {output_dir}") from exc
    _write_wrapper_log(live_stdout_path, stdout_path.read_text(encoding="utf-8"))
    _write_wrapper_log(live_stderr_path, stderr_path.read_text(encoding="utf-8"))
    _write_wrapper_log(output_dir / "bonemechreg_parosol_exit_code.txt", f"{returncode}\n")
    if returncode != 0:
        raise RuntimeError(f"parosol-py profile {profile!r} failed with exit code {returncode}; see {output_dir}")
    sed_path = output_dir / "fields" / "sed.nii.gz"
    if not sed_path.exists():
        raise RuntimeError(f"parosol-py did not export fields/sed.nii.gz; see {output_dir}")
    return sitk.GetArrayFromImage(sitk.ReadImage(str(sed_path))).astype(np.float32, copy=False)


def _parosol_cli_command(*, image_path: Path, profile: str, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "parosol_py.cli",
        str(image_path),
        "--profile",
        str(profile),
        "--output",
        str(output_dir),
    ]


def _write_wrapper_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _mirrored_text_log:
    def __init__(self, primary: Path, mirror: Path):
        self.primary = Path(primary)
        self.mirror = Path(mirror)
        self._primary_handle = None
        self._mirror_handle = None

    def __enter__(self):
        self.primary.parent.mkdir(parents=True, exist_ok=True)
        self.mirror.parent.mkdir(parents=True, exist_ok=True)
        self._primary_handle = self.primary.open("w", encoding="utf-8")
        self._mirror_handle = self.mirror.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in (self._primary_handle, self._mirror_handle):
            if handle is not None:
                handle.close()

    def write(self, text: str) -> int:
        if self._primary_handle is not None:
            self._primary_handle.write(text)
            self._primary_handle.flush()
        if self._mirror_handle is not None:
            self._mirror_handle.write(text)
            self._mirror_handle.flush()
        return len(text)

    def flush(self) -> None:
        for handle in (self._primary_handle, self._mirror_handle):
            if handle is not None:
                handle.flush()


def solve_sed_to_file(
    *,
    material_image_path: str | Path,
    output_path: str | Path,
    profile: str,
    debug_dir: str | Path | None = None,
) -> Path:
    """Solve baseline SED with ParOSol and write it next to case outputs.

    Args:
        material_image_path: NIfTI material-label image in the same grid as the
            remodelling labels. Standard XtremeCT profiles expect label ``100``
            for trabecular baseline bone and label ``127`` for cortical
            baseline bone. Every other label is ignored by the profile.
        output_path: Destination ``.nii.gz`` path for the exported SED field.
        profile: Scanner profile alias, usually ``XtremeCTI`` or ``XtremeCTII``.
        debug_dir: Optional durable parosol-py run directory for command,
            stdout/stderr, native logs, and intermediate files.

    Returns:
        The written SED path.
    """
    material_image_path = Path(material_image_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        solve_dir = Path(debug_dir).expanduser().resolve() if debug_dir is not None else None
        sed_zyx = _run_parosol_profile(
            image_path=material_image_path,
            profile=str(profile),
            debug_dir=solve_dir,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("parosol-py is required to solve baseline SED") from exc

    reference = sitk.ReadImage(str(material_image_path))
    out = sitk.GetImageFromArray(np.asarray(sed_zyx, dtype=np.float32), isVector=False)
    out.CopyInformation(reference)
    sitk.WriteImage(out, str(output_path))
    return output_path
