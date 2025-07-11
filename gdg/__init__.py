"""
Group Dataset Generator (gdg) - A package for generating permutation group datasets.
"""

from .base_generator import BaseGroupGenerator
from .symmetric_generator import SymmetricGroupGenerator
from .alternating_generator import AlternatingGroupGenerator
from .cyclic_generator import CyclicGroupGenerator
from .dihedral_generator import DihedralGroupGenerator
from .quaternion_generator import QuaternionGroupGenerator
from .klein_generator import KleinFourGroupGenerator
from .elementary_abelian_generator import ElementaryAbelianGroupGenerator
from .frobenius_generator import FrobeniusGroupGenerator
from .psl_generator import PSLGroupGenerator
from .mathieu_generator import MathieuGroupGenerator

__all__ = [
    "BaseGroupGenerator",
    "SymmetricGroupGenerator",
    "AlternatingGroupGenerator",
    "CyclicGroupGenerator",
    "DihedralGroupGenerator",
    "QuaternionGroupGenerator",
    "KleinFourGroupGenerator",
    "ElementaryAbelianGroupGenerator",
    "FrobeniusGroupGenerator",
    "PSLGroupGenerator",
    "MathieuGroupGenerator",
]

__version__ = "0.1.0"
