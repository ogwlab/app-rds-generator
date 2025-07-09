"""
RDS Generator Package

Random Dot Stereogram Generator for Psychology Research
"""

from .core import RDSGenerator
from .config import RDSConfig

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

try:
    __version__ = version("rds-generator")
except Exception:
    __version__ = "unknown"

__author__ = "Ogawa Lab, Kwansei Gakuin University"
__email__ = "hirokazu.ogawa@kwansei.ac.jp"

__all__ = ["RDSGenerator", "RDSConfig"]