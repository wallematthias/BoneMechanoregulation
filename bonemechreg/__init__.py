"""Small surface-based mechanoregulation analysis package.

The top-level package exports only the core analysis result type and analysis
functions. Mechanics solving and workflow helpers live in explicit modules such
as :mod:`bonemechreg.parosol` and :mod:`bonemechreg.post_timelapse`.
"""

__all__ = [
    "MechanoregulationResult",
    "derive_remodelling_labels_from_density",
    "mechanoregulation",
    "__version__",
]

__version__ = "0.1.0"
from bonemechreg.mechreg import MechanoregulationResult, derive_remodelling_labels_from_density, mechanoregulation
