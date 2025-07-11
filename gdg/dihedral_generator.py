#!/usr/bin/env python3
"""
Generator for dihedral group datasets.
"""

import numpy as np
from typing import List, Dict

try:
    from .base_generator import BaseGroupGenerator
except ImportError:
    from base_generator import BaseGroupGenerator


class DihedralGroupGenerator(BaseGroupGenerator):
    """Generator for dihedral groups Dn."""

    def get_group_name(self) -> str:
        return "dihedral"

    def generate_group(self, n: int) -> List[np.ndarray]:
        """Generate dihedral group Dn (symmetries of regular n-gon)."""
        if n < 3:
            raise ValueError(f"n must be at least 3 for dihedral groups, got {n}")

        perms = []

        # n rotations
        for k in range(n):
            # Rotation by k * (360/n) degrees
            perm = np.array([(i + k) % n for i in range(n)])
            perms.append(perm)

        # n reflections
        for k in range(n):
            # Reflection followed by rotation by k positions
            perm = np.array([(k - i) % n for i in range(n)])
            perms.append(perm)

        return perms

    def get_valid_parameters(self) -> List[Dict]:
        """Return valid parameters for dihedral groups."""
        return [{"n": n} for n in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]]


def main():
    """Generate dihedral group datasets."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate dihedral group datasets")
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20],
        help="Degrees to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../individual_datasets",
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

    generator = DihedralGroupGenerator(seed=args.seed)
    output_base = Path(args.output_dir)

    for n in args.degrees:
        print(f"\n{'=' * 60}")
        print(f"Generating D{n}")
        print("=" * 60)

        try:
            output_dir = output_base / f"d{n}_data"
            dataset_dict, metadata = generator.generate_dataset(
                params={"n": n},
                num_train_samples=args.num_train,
                num_test_samples=args.num_test,
                output_dir=output_dir,
            )

            print(f"✓ Successfully generated D{n}")

        except Exception as e:
            print(f"✗ Error generating D{n}: {e}")


if __name__ == "__main__":
    main()
