"""Group generators for creating permutation datasets."""

from .alternating import AlternatingGenerator
from .cyclic import CyclicGenerator
from .dihedral import DihedralGenerator
from .elementary_abelian import ElementaryAbelianGenerator
from .frobenius import FrobeniusGenerator
from .klein import KleinGenerator
from .mathieu import MathieuGenerator
from .psl import PSLGenerator
from .quaternion import QuaternionGenerator
from .symmetric import SymmetricGenerator

__all__ = [
    "AlternatingGenerator",
    "CyclicGenerator",
    "DihedralGenerator",
    "ElementaryAbelianGenerator",
    "FrobeniusGenerator",
    "KleinGenerator",
    "MathieuGenerator",
    "PSLGenerator",
    "QuaternionGenerator",
    "SymmetricGenerator",
]
