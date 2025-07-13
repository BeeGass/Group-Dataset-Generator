#!/usr/bin/env python3
"""
Generator for elementary abelian group datasets.
"""

import numpy as np
from typing import List, Dict
from itertools import product

from ..base_generator import BaseGroupGenerator


class ElementaryAbelianGroupGenerator(BaseGroupGenerator):
    """Generator for elementary abelian groups Z_p^k."""

    def get_group_name(self) -> str:
        return "elementary_abelian"

    def generate_group(self, p: int, k: int) -> List[np.ndarray]:
        """Generate elementary abelian group Z_p^k."""
        if p < 2:
            raise ValueError(f"p must be at least 2, got {p}")
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")

        # Check if p is prime
        if p > 2:
            for i in range(2, int(p**0.5) + 1):
                if p % i == 0:
                    raise ValueError(f"p must be prime, but {p} is not prime")

        n = p**k  # Order of the group

        # For small groups, generate as cyclic product
        if n <= 16:
            # Generate using direct product of cyclic groups
            generators = []

            # Create k generators, each cycling through p^(k-1) blocks
            for gen_idx in range(k):
                perm = np.arange(n, dtype=int)
                block_size = p**gen_idx
                num_blocks = n // (block_size * p)

                for block in range(num_blocks):
                    for i in range(p):
                        for j in range(block_size):
                            src = block * block_size * p + i * block_size + j
                            dst = (
                                block * block_size * p + ((i + 1) % p) * block_size + j
                            )
                            perm[src] = dst

                generators.append(perm)

            # Generate all group elements from generators
            perms = [np.arange(n)]  # Identity
            generated = {tuple(np.arange(n))}
            to_process = generators.copy()

            while to_process:
                g = to_process.pop()
                g_tuple = tuple(g)
                if g_tuple not in generated:
                    perms.append(g)
                    generated.add(g_tuple)

                    # Generate new elements by composing with all existing
                    for existing in perms[:]:
                        new_perm = existing[g]
                        if tuple(new_perm) not in generated:
                            to_process.append(new_perm)
        else:
            # For larger groups, use a more efficient representation
            # Generate permutations corresponding to addition in Z_p^k
            perms = []

            # Create mapping from tuples to indices
            tuple_to_idx = {}
            idx_to_tuple = {}
            for i, t in enumerate(product(range(p), repeat=k)):
                tuple_to_idx[t] = i
                idx_to_tuple[i] = t

            # Generate permutation for each element
            for vec in product(range(p), repeat=k):
                perm = np.zeros(n, dtype=int)

                for i in range(n):
                    source_tuple = idx_to_tuple[i]
                    # Add vec to source (mod p)
                    target_tuple = tuple(
                        (source_tuple[j] + vec[j]) % p for j in range(k)
                    )
                    perm[i] = tuple_to_idx[target_tuple]

                perms.append(perm)

        return perms

    def get_valid_parameters(self) -> List[Dict]:
        """Return valid parameters for elementary abelian groups."""
        return [
            {"p": 2, "k": 1},  # Z2
            {"p": 2, "k": 2},  # Z2^2 (Klein four group)
            {"p": 2, "k": 3},  # Z2^3
            {"p": 2, "k": 4},  # Z2^4
            {"p": 2, "k": 5},  # Z2^5
            {"p": 3, "k": 1},  # Z3
            {"p": 3, "k": 2},  # Z3^2
            {"p": 3, "k": 3},  # Z3^3
            {"p": 5, "k": 1},  # Z5
            {"p": 5, "k": 2},  # Z5^2
        ]


def main():
    """Generate elementary abelian group datasets."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate elementary abelian group datasets"
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["2,1", "2,2", "2,3", "2,4", "2,5", "3,1", "3,2", "3,3", "5,1", "5,2"],
        help="Groups to generate as p,k pairs (default: all standard ones)",
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

    generator = ElementaryAbelianGroupGenerator(seed=args.seed)
    output_base = Path(args.output_dir)

    for group_str in args.groups:
        p, k = map(int, group_str.split(","))
        print(f"\n{'=' * 60}")
        print(f"Generating Z{p}^{k}")
        print("=" * 60)

        try:
            output_dir = output_base / f"z{p}_{k}_data"
            dataset_dict, metadata = generator.generate_dataset(
                params={"p": p, "k": k},
                num_train_samples=args.num_train,
                num_test_samples=args.num_test,
                output_dir=output_dir,
            )

            print(f"✓ Successfully generated Z{p}^{k}")

        except Exception as e:
            print(f"✗ Error generating Z{p}^{k}: {e}")


if __name__ == "__main__":
    main()
