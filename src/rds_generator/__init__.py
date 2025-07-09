"""
RDS Generator Package

Random Dot Stereogram Generator for Psychology Research
"""

from .core import RDSGenerator
from .config import RDSConfig

__version__ = "1.0.0"
__author__ = "Ogawa Lab, Kwansei Gakuin University"
__email__ = "hirokazu.ogawa@kwansei.ac.jp"

__all__ = ["RDSGenerator", "RDSConfig"]