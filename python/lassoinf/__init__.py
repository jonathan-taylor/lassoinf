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

from .lasso import LassoInference, spec_from_glmnet
from .glmnet import extract_glmnet_problem
from .affine_constraints import AffineConstraints
