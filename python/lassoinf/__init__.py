from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lassoinf")
except PackageNotFoundError:
    # package is not installed, perhaps we are in a git repo
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "unknown"

from .lasso import LassoInference
from .affine_constraints import AffineConstraints
from .custom_estimand import (SelectionCoordinates,
                              ScreenedSelection,
                              ContrastEstimand,
                              contrast_inference,
                              custom_estimand_inference,
                              estimand_summary,
                              inactive_summary)
