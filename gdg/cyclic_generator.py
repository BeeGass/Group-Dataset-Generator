#!/usr/bin/env python3
"""
Generator for cyclic group datasets.
"""

import numpy as np
from typing import List, Dict

try:
    from .base_generator import BaseGroupGenerator
except ImportError:
    from base_generator import BaseGroupGenerator


class CyclicGroupGenerator(BaseGroupGenerator):
    """Generator for cyclic groups Cn."""

    def get_group_name(self) -> str:
        return "cyclic"

    def generate_group(self, n: int) -> List[np.ndarray]:
        """Generate cyclic group Cn."""
        if n < 1:
            raise ValueError(f"n must be positive, got {n}")

        # Cyclic group elements are powers of a generator
        # We represent them as permutations
        perms = []
        for k in range(n):
            # k-th power of the basic n-cycle (0 1 2 ... n-1)
            perm = np.array([(i + k) % n for i in range(n)])
            perms.append(perm)

        return perms

    def get_valid_parameters(self) -> List[Dict]:
        """Return valid parameters for cyclic groups."""
        return [{"n": n} for n in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]]


def main():
    """Generate cyclic group datasets."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate cyclic group datasets")
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30],
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

    generator = CyclicGroupGenerator(seed=args.seed)
    output_base = Path(args.output_dir)

    for n in args.degrees:
        print(f"\n{'=' * 60}")
        print(f"Generating C{n}")
        print("=" * 60)

        try:
            output_dir = output_base / f"c{n}_data"
            dataset_dict, metadata = generator.generate_dataset(
                params={"n": n},
                num_train_samples=args.num_train,
                num_test_samples=args.num_test,
                output_dir=output_dir,
            )

            print(f"✓ Successfully generated C{n}")

        except Exception as e:
            print(f"✗ Error generating C{n}: {e}")


if __name__ == "__main__":
    main()
