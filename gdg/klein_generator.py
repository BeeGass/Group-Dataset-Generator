#!/usr/bin/env python3
"""
Generator for Klein Four group dataset.
"""

import numpy as np
from typing import List, Dict

try:
    from .base_generator import BaseGroupGenerator
except ImportError:
    from base_generator import BaseGroupGenerator


class KleinFourGroupGenerator(BaseGroupGenerator):
    """Generator for Klein Four group V4."""

    def get_group_name(self) -> str:
        return "klein"

    def generate_group(self) -> List[np.ndarray]:
        """Generate Klein Four group V4."""
        # V4 = Z2 × Z2, represented as permutations on 4 elements
        # It's the symmetry group of a rectangle (not square)

        perms = [
            np.array([0, 1, 2, 3]),  # identity
            np.array([1, 0, 3, 2]),  # horizontal reflection
            np.array([2, 3, 0, 1]),  # vertical reflection
            np.array([3, 2, 1, 0]),  # 180° rotation
        ]

        return perms

    def get_valid_parameters(self) -> List[Dict]:
        """Return valid parameters for Klein Four group."""
        return [{}]  # No parameters needed


def main():
    """Generate Klein Four group dataset."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate Klein Four group dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../individual_datasets",
        help="Output directory",
    )
    parser.add_argument(
        "--num-train", type=int, default=40000, help="Number of training samples"
    )
    parser.add_argument(
        "--num-test", type=int, default=10000, help="Number of test samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generator = KleinFourGroupGenerator(seed=args.seed)
    output_base = Path(args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"Generating V4 (Klein Four Group)")
    print("=" * 60)

    try:
        output_dir = output_base / "v4_data"
        dataset_dict, metadata = generator.generate_dataset(
            params={},
            num_train_samples=args.num_train,
            num_test_samples=args.num_test,
            output_dir=output_dir,
        )

        print(f"✓ Successfully generated V4")

    except Exception as e:
        print(f"✗ Error generating V4: {e}")


if __name__ == "__main__":
    main()
