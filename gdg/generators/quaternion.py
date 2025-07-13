#!/usr/bin/env python3
"""
Generator for quaternion group datasets.
"""

import numpy as np
from typing import List, Dict

from ..base_generator import BaseGroupGenerator


class QuaternionGroupGenerator(BaseGroupGenerator):
    """Generator for generalized quaternion groups Q_{2^k}."""

    def get_group_name(self) -> str:
        return "quaternion"

    def generate_group(self, k: int) -> List[np.ndarray]:
        """Generate generalized quaternion group Q_{2^k}."""
        if k < 3:
            raise ValueError(
                f"k must be at least 3 for quaternion groups (Q8 or larger), got {k}"
            )

        n = 2**k  # Order of the group
        m = n // 2  # Q_{2^k} has a cyclic subgroup of order 2^{k-1}

        # Q_{2^k} can be represented as permutations on n elements
        # We use a representation based on the presentation:
        # Q_{2^k} = <a, b | a^{2^{k-1}} = 1, b^2 = a^{2^{k-2}}, bab^{-1} = a^{-1}>

        perms = []

        # Elements of form a^i for i = 0, ..., m-1
        for i in range(m):
            # a acts as a cyclic permutation on first m elements
            # and another cyclic permutation on last m elements
            perm = np.zeros(n, dtype=int)
            for j in range(m):
                perm[j] = (j + i) % m
                perm[m + j] = m + ((j + i) % m)
            perms.append(perm)

        # Elements of form a^i * b for i = 0, ..., m-1
        for i in range(m):
            # b swaps the two halves with appropriate twisting
            perm = np.zeros(n, dtype=int)
            for j in range(m):
                perm[j] = m + ((i - j) % m)
                perm[m + j] = (i - j) % m
            perms.append(perm)

        return perms

    def get_valid_parameters(self) -> List[Dict]:
        """Return valid parameters for quaternion groups."""
        return [{"k": k} for k in [3, 4, 5]]  # Q8, Q16, Q32


def main():
    """Generate quaternion group datasets."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate quaternion group datasets")
    parser.add_argument(
        "--powers",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="Powers k for Q_{2^k} (default: 3,4,5 for Q8,Q16,Q32)",
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

    generator = QuaternionGroupGenerator(seed=args.seed)
    output_base = Path(args.output_dir)

    for k in args.powers:
        n = 2**k
        print(f"\n{'=' * 60}")
        print(f"Generating Q{n}")
        print("=" * 60)

        try:
            output_dir = output_base / f"q{n}_data"
            dataset_dict, metadata = generator.generate_dataset(
                params={"k": k},
                num_train_samples=args.num_train,
                num_test_samples=args.num_test,
                output_dir=output_dir,
            )

            print(f"✓ Successfully generated Q{n}")

        except Exception as e:
            print(f"✗ Error generating Q{n}: {e}")


if __name__ == "__main__":
    main()
