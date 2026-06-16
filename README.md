# BoneMechanoregulation

Post-timelapse mechanoregulation analysis for HR-pQCT datasets.

This package is a downstream addon. Run `TimelapsedHRpQCT` first, then run
BoneMechanoregulation on the Timelapsed output dataset root. The addon reuses
the Timelapsed pairwise remodelling label images, solves baseline SED from the
native baseline segmentation, and writes mechanoregulation summaries next to
each pairwise case.

## Prerequisite

First create TimelapsedHRpQCT remodelling outputs.

With the core TimelapsedHRpQCT CLI, the pipeline convention is:

```bash
timelapsedhrpqct run /path/to/raw_aim_input --output-root /path/to/TimelapsedHRpQCT
```

In the 3D Slicer toolbox, use the `Timelapsed HR-pQCT` module:

1. select the AIM dataset root;
2. choose a results folder, defaulting to `<dataset_root>/TimelapsedHRpQCT`;
3. click `Run pipeline`;
4. inspect the generated remodelling images.

The Slicer wrapper is the recommended route for interactive preparation, mask
review, and visual QA before running this mechanoregulation addon.

## Install

From PyPI:

```bash
pip install bone-mechanoregulation
```

For local development:

```bash
pip install -e .
```

`parosol-py>=0.1.13` must be available in the same Python environment. The
mechanoregulation workflow writes a standard ParOSol material-label input from
the native baseline segmentation, then calls the native `parosol-py`
XtremeCTI/XtremeCTII profile for the finite-element solve.

## Install In 3D Slicer

BoneMechanoregulation can also be installed into Slicer's Python environment so
the same command-line analysis is available from Slicer-side workflows or
scripted modules.

From Slicer's Python interactor or a Slicer Python shell:

```python
import sys
import subprocess

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "bone-mechanoregulation",
])
```

For local development from a checkout:

```python
import sys
import subprocess

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-e",
    "/path/to/BoneMechanoregulation",
])
```

The package supports the same Python version range used by the current
TimelapsedHRpQCT packaging matrix: Python 3.11, 3.12, and 3.13. If a Slicer
release ships an older Python, use an external Python environment for the batch
analysis or install through the matching Slicer toolbox once it provides a
compatible runtime.

## Batch Usage

Run this after TimelapsedHRpQCT has produced its derivative tree:

```bash
mechanoregulation run \
  /path/to/TimelapsedHRpQCT \
  --profile XtremeCTII
```

The positional `dataset_root` follows the TimelapsedHRpQCT core CLI convention:
the root path is the main command argument. Here it is the TimelapsedHRpQCT
output root, not the raw AIM input folder. The addon expects to find pairwise
remodelling cases under:

```text
<dataset-root>/derivatives/TimelapsedHRpQCT/sub-*/analysis/pairwise_t0/*/
```

Current TimelapsedHRpQCT outputs under
`sub-*/site-*/analysis/visualize/*remodelling.nii.gz` are also supported. For
these outputs, the mechanics solve uses the native stack-level baseline
segmentation and compartment masks under `sub-*/site-*/ses-*/stacks/`.

Useful flags:

- `--profile XtremeCTI` or `--profile XtremeCTII`: native `parosol-py` scanner
  profile. The generated material image uses the standard profile labels:
  `100` for trabecular baseline bone, `127` for cortical baseline bone, and
  all other labels as non-bone.
- `--dry-run`: count discoverable cases without writing files.
- `--overwrite`: recompute outputs even if they already exist.
- `--verbose`: re-raise case errors instead of counting failed cases.

## Outputs

For current TimelapsedHRpQCT outputs, results are written into a site-level
`mechanoregulation` folder beside `analysis`, `ses-*`, and
`transformed_images`:

```text
sub-<id>/site-<site>/
  analysis/
    visualize/
      *_remodelling.nii.gz
  mechanoregulation/
    <case>_sed.nii.gz
    <case>_baseline_material_labels.nii.gz
    <case>_conditional_curves.png
    <case>_mechanoregulation_summary.csv
    <case>_mechanoregulation_summary.json
```

Older Timelapsed layouts that already store each pairwise case in its own
folder keep writing a `mechanoregulation` subfolder beside that pairwise case.

Output files:

- `<case>_baseline_material_labels.nii.gz`: ParOSol input image built from the
  native baseline segmentation. Label `100` is trabecular baseline bone, label
  `127` is cortical baseline bone, and background plus future formation sites
  are `0`.
- `<case>_sed.nii.gz`: baseline SED solved with `parosol-py` and aligned to the
  Timelapsed remodelling image grid. If this file already exists, it is reused unless
  `--overwrite` is set.
- `<case>_conditional_curves.png`: two-panel summary figure. The left panel
  shows Schulte-style binned conditional probability curves for resorption,
  quiescence, and formation. The right panel shows bootstrap logistic
  probability curves.
- `<case>_mechanoregulation_summary.csv`: one-row table for downstream
  statistics.
- `<case>_mechanoregulation_summary.json`: full reproducibility payload,
  including curves, settings, sample counts, and confidence intervals.

The CSV is intentionally flat and stable. A TimelapsedHRpQCT or Slicer export
button can merge it into cohort-level exports by scanning each pairwise case's
`mechanoregulation/*.csv` file and joining by subject/site/session-pair stem.
That export integration is separate from this package; this package writes the
per-case CSV files that such an exporter can pick up.

## Output Measures

The analysis samples baseline SED on the quiescent baseline surface. Formation
and resorption events are projected symmetrically onto neighbouring baseline
surface voxels. If projected formation and resorption overlap on the same
surface voxel, that voxel is counted as quiescent/ambiguous.

CSV columns:

- `CCR`: maximum correct classification rate from Schulte-style conditional
  probability curves. It summarizes how well low SED maps to resorption,
  middle SED maps to quiescence, and high SED maps to formation.
- `CCR_low_threshold`: normalized SED threshold separating the resorption side
  from the lazy zone in the binned CCR analysis.
- `CCR_high_threshold`: normalized SED threshold separating the lazy zone from
  the formation side in the binned CCR analysis.
- `binned_lazy_zone_low`, `binned_lazy_zone_high`: aliases of the CCR
  thresholds, reported explicitly as the binned lazy-zone bounds.
- `logistic_lazy_zone_low`, `logistic_lazy_zone_high`: lazy-zone bounds from
  the smooth logistic probability curves, estimated from R/Q and F/Q curve
  crossings.
- `OR_F`: percent increase in formation odds per one normalized SED
  percentage-point increase. This is named `OR_F` for reporting compatibility,
  but its value is `100 * (OR_F_ratio - 1)`.
- `OR_R`: percent increase in resorption odds per one normalized SED
  percentage-point decrease. This is named `OR_R` for reporting compatibility,
  but its value is `100 * (OR_R_ratio - 1)`.
- `OR_F_CI_low`, `OR_F_CI_high`: bootstrap confidence interval for reported
  percent `OR_F`.
- `OR_R_CI_low`, `OR_R_CI_high`: bootstrap confidence interval for reported
  percent `OR_R`.
- `OR_F_ratio`, `OR_R_ratio`: raw logistic odds ratios retained for
  reproducibility.
- `formation_odds_increase_percent_per_sed_percent`: percent odds increase for
  formation per one normalized SED percentage-point increase, computed as
  `100 * (OR_F_ratio - 1)`. This duplicates reported `OR_F`.
- `resorption_odds_increase_percent_per_sed_percent_decrease`: percent odds
  increase for resorption per one normalized SED percentage-point decrease,
  computed as `100 * (OR_R_ratio - 1)`. This duplicates reported `OR_R`.
- `formation_odds_increase_percent_CI_low`,
  `formation_odds_increase_percent_CI_high`: confidence interval for the
  formation percent odds increase.
- `resorption_odds_increase_percent_CI_low`,
  `resorption_odds_increase_percent_CI_high`: confidence interval for the
  resorption percent odds increase.
- `n_surface_voxels`: number of baseline surface voxels available before final
  class sampling.
- `n_sampled_voxels`: number of surface voxels used in the final
  mechanoregulation analysis.
- `n_formation`, `n_resorption`, `n_quiescence`: final class counts after
  symmetric surface projection.
- `n_cancelled_overlap`: number of surface voxels where projected formation and
  resorption overlapped and were cancelled to quiescence.

## Standalone Usage

For synthetic advection folders or one-off debugging, use:

```bash
mechanoregulation analyze /path/to/folder
```

The standalone command looks for a baseline image, a remodelling label image or
T1 follow-up, an optional mask, and a baseline SED image or nearby exported
ParOSol `fields/sed.nii.gz`. It writes:

```text
<folder>/mechanoregulation/
  <run>_mechanoregulation_summary.csv
  <run>_mechanoregulation_summary.json
  <run>_mechanoregulation_curves.png
```
