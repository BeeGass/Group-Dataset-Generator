#!/usr/bin/env python3
"""
Generator for alternating group datasets.
"""

import numpy as np
from itertools import permutations
from typing import List, Dict

from ..base_generator import BaseGroupGenerator


class AlternatingGroupGenerator(BaseGroupGenerator):
    """Generator for alternating groups An."""

    def get_group_name(self) -> str:
        return "alternating"

    def _parity(self, perm: tuple) -> int:
        """Calculate parity of a permutation (0 for even, 1 for odd)."""
        n = len(perm)
        inversions = 0
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    inversions += 1
        return inversions % 2

    def generate_group(self, n: int) -> List[np.ndarray]:
        """Generate alternating group An (even permutations of n elements)."""
        if n < 3:
            raise ValueError(f"n must be at least 3 for alternating groups, got {n}")

        # Warn for large n
        import math

        if n > 10:
            print(
                f"Warning: A{n} has {math.factorial(n) // 2:,} elements. This may take significant time and memory."
            )

        # Generate all even permutations
        perms = []
        for perm in permutations(range(n)):
            if self._parity(perm) == 0:  # Even permutation
                perms.append(np.array(perm))

        return perms

    def get_valid_parameters(self) -> List[Dict]:
        """Return valid parameters for alternating groups."""
        return [{"n": n} for n in range(3, 10)]  # A3 to A9


def main():
    """Generate alternating group datasets."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate alternating group datasets")
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=list(range(3, 10)),
        help="Degrees to generate (default: 3-9)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../datasets",
        help="Output directory",
    )
    parser.add_argument(
        "--num-train", type=int, default=100000, help="Number of training samples"
    )
    parser.add_argument(
        "--num-test", type=int, default=20000, help="Number of test samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generator = AlternatingGroupGenerator(seed=args.seed)
    output_base = Path(args.output_dir)

    for n in args.degrees:
        print(f"\n{'=' * 60}")
        print(f"Generating A{n}")
        print("=" * 60)

        try:
            output_dir = output_base / f"a{n}_data"
            dataset_dict, metadata = generator.generate_dataset(
                params={"n": n},
                num_train_samples=args.num_train,
                num_test_samples=args.num_test,
                output_dir=output_dir,
            )

            print(f"✓ Successfully generated A{n}")

        except Exception as e:
            print(f"✗ Error generating A{n}: {e}")


if __name__ == "__main__":
    main()
