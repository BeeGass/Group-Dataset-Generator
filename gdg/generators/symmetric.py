#!/usr/bin/env python3
"""
Generator for symmetric group datasets.
"""

import numpy as np
from itertools import permutations
from typing import List, Dict

from ..base_generator import BaseGroupGenerator


class SymmetricGroupGenerator(BaseGroupGenerator):
    """Generator for symmetric groups Sn."""

    def get_group_name(self) -> str:
        return "symmetric"

    def generate_group(self, n: int) -> List[np.ndarray]:
        """Generate symmetric group Sn (all permutations of n elements)."""
        if n < 1:
            raise ValueError(f"n must be positive, got {n}")

        # Warn for large n
        import math

        if n > 10:
            print(
                f"Warning: S{n} has {math.factorial(n):,} elements. This may take significant time and memory."
            )

        # Generate all permutations
        perms = []
        for perm in permutations(range(n)):
            perms.append(np.array(perm))

        return perms

    def get_valid_parameters(self) -> List[Dict]:
        """Return valid parameters for symmetric groups."""
        return [{"n": n} for n in range(3, 10)]  # S3 to S9


def main():
    """Generate symmetric group datasets."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate symmetric group datasets")
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

    generator = SymmetricGroupGenerator(seed=args.seed)
    output_base = Path(args.output_dir)

    for n in args.degrees:
        print(f"\n{'=' * 60}")
        print(f"Generating S{n}")
        print("=" * 60)

        try:
            output_dir = output_base / f"s{n}_data"
            dataset_dict, metadata = generator.generate_dataset(
                params={"n": n},
                num_train_samples=args.num_train,
                num_test_samples=args.num_test,
                output_dir=output_dir,
            )

            print(f"✓ Successfully generated S{n}")

        except Exception as e:
            print(f"✗ Error generating S{n}: {e}")


if __name__ == "__main__":
    main()
