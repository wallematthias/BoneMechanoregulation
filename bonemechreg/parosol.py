"""Thin ParOSol wrapper used by the post-timelapse workflow.

BoneMechanoregulation does not implement finite-element mechanics itself. It
builds a baseline material-label image, asks the native ``parosol-py`` scanner
profile to solve the SED field, and then reads the exported
``fields/sed.nii.gz`` image.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk


def _run_parosol_profile(*, image_path: Path, profile: str) -> np.ndarray:
    """Run the configured ParOSol profile and return SED as a NumPy array."""
    return _run_parosol_profile_cli(image_path=image_path, profile=profile)


def _run_parosol_profile_cli(*, image_path: Path, profile: str) -> np.ndarray:
    """Execute the native ``parosol-py`` profile shortcut.

    The input image must already be a material-label image in baseline space.
    The standard XtremeCTI/XtremeCTII profiles expect ``100`` for trabecular
    bone, ``127`` for cortical bone, and all other labels as non-bone.
    """
    try:
        from parosol_py.cli import main as parosol_main
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("No module named 'parosol_py'") from exc

    with TemporaryDirectory(prefix="bonemechreg_parosol_") as tmp:
        output_dir = Path(tmp) / "case"
        exit_code = int(
            parosol_main(
                [
                    str(image_path),
                    "--profile",
                    str(profile),
                    "--output",
                    str(output_dir),
                ]
            )
        )
        if exit_code != 0:
            raise RuntimeError(f"parosol-py profile {profile!r} failed with exit code {exit_code}")
        sed_path = output_dir / "fields" / "sed.nii.gz"
        if not sed_path.exists():
            raise RuntimeError("parosol-py did not export fields/sed.nii.gz")
        return sitk.GetArrayFromImage(sitk.ReadImage(str(sed_path))).astype(np.float32, copy=False)


def solve_sed_to_file(
    *,
    material_image_path: str | Path,
    output_path: str | Path,
    profile: str,
) -> Path:
    """Solve baseline SED with ParOSol and write it next to case outputs.

    Args:
        material_image_path: NIfTI material-label image in the same grid as the
            remodelling labels. Standard XtremeCT profiles expect label ``100``
            for trabecular baseline bone and label ``127`` for cortical
            baseline bone. Every other label is ignored by the profile.
        output_path: Destination ``.nii.gz`` path for the exported SED field.
        profile: Scanner profile alias, usually ``XtremeCTI`` or ``XtremeCTII``.

    Returns:
        The written SED path.
    """
    material_image_path = Path(material_image_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sed_zyx = _run_parosol_profile(image_path=material_image_path, profile=str(profile))
    except ModuleNotFoundError as exc:
        raise RuntimeError("parosol-py is required to solve baseline SED") from exc

    reference = sitk.ReadImage(str(material_image_path))
    out = sitk.GetImageFromArray(np.asarray(sed_zyx, dtype=np.float32), isVector=False)
    out.CopyInformation(reference)
    sitk.WriteImage(out, str(output_path))
    return output_path
