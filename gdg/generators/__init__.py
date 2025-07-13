"""Group generators for creating permutation datasets."""

from .alternating import AlternatingGroupGenerator
from .cyclic import CyclicGroupGenerator
from .dihedral import DihedralGroupGenerator
from .elementary_abelian import ElementaryAbelianGroupGenerator
from .frobenius import FrobeniusGroupGenerator
from .klein import KleinFourGroupGenerator
from .mathieu import MathieuGroupGenerator
from .psl import PSLGroupGenerator
from .quaternion import QuaternionGroupGenerator
from .symmetric import SymmetricGroupGenerator

__all__ = [
    "AlternatingGroupGenerator",
    "CyclicGroupGenerator",
    "DihedralGroupGenerator",
    "ElementaryAbelianGroupGenerator",
    "FrobeniusGroupGenerator",
    "KleinFourGroupGenerator",
    "MathieuGroupGenerator",
    "PSLGroupGenerator",
    "QuaternionGroupGenerator",
    "SymmetricGroupGenerator",
]
