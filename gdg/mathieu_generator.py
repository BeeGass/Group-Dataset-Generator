#!/usr/bin/env python3
"""
Generator for Mathieu group datasets.
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from .base_generator import BaseGroupGenerator
except ImportError:
    from base_generator import BaseGroupGenerator


def _cycles_to_permutation(cycles: list[tuple[int]], n: int) -> np.ndarray:
    """Converts a list of cycles (1-indexed) to a 0-indexed permutation array."""
    perm = np.arange(n)
    for cycle in cycles:
        cycle_0_indexed = [x - 1 for x in cycle]
        for i in range(len(cycle_0_indexed)):
            perm[cycle_0_indexed[i]] = cycle_0_indexed[(i + 1) % len(cycle_0_indexed)]
    return perm


class MathieuGroupGenerator(BaseGroupGenerator):
    """Generator for Mathieu groups M_n as permutations."""

    def get_group_name(self) -> str:
        return "mathieu"

    def generate_group(self, n: int) -> list[np.ndarray]:
        """
        Generates the full permutation group for M_n.
        """
        if n not in [11, 12, 22, 23, 24]:
            raise ValueError(
                f"Mathieu group M_{n} is not defined. n must be in {{11, 12, 22, 23, 24}}."
            )

        # Standard generators from ATLAS of Finite Groups and other sources, converted to 0-indexed permutations.
        if n == 11:
            # Source: ATLAS of Finite Group Representations
            gen_a = _cycles_to_permutation([(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)], 11)
            gen_b = _cycles_to_permutation([(3, 7, 11, 8), (4, 10, 5, 6)], 11)
            perm_gens = [gen_a, gen_b]
        elif n == 12:
            # M12 generators that extend M11 by adding point 12
            # These are the standard generators used by GAP
            gen_a = _cycles_to_permutation([(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)], 12)
            gen_b = _cycles_to_permutation([(3, 7, 11, 8), (4, 10, 5, 6)], 12)
            gen_c = _cycles_to_permutation(
                [(1, 12), (2, 11), (3, 6), (4, 8), (5, 9), (7, 10)], 12
            )
            perm_gens = [gen_a, gen_b, gen_c]
        elif n == 22:
            # Source: ATLAS of Finite Group Representations
            gen_a = _cycles_to_permutation(
                [
                    (1, 2),
                    (3, 4),
                    (5, 6),
                    (7, 8),
                    (9, 10),
                    (11, 12),
                    (13, 14),
                    (15, 16),
                    (17, 18),
                    (19, 20),
                    (21, 22),
                ],
                22,
            )
            gen_b = _cycles_to_permutation(
                [
                    (1, 3, 5, 9, 17),
                    (2, 4, 6, 10, 18),
                    (7, 13, 21, 11, 14),
                    (8, 15, 22, 12, 16),
                ],
                22,
            )
            perm_gens = [gen_a, gen_b]
        elif n == 23:
            # Source: ATLAS of Finite Group Representations
            gen_a = _cycles_to_permutation(
                [
                    (
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                    )
                ],
                23,
            )
            gen_b = _cycles_to_permutation(
                [
                    (2, 16, 9, 6, 8),
                    (3, 12, 13, 18, 4),
                    (7, 17, 10, 11, 22),
                    (14, 19, 21, 20, 15),
                ],
                23,
            )
            perm_gens = [gen_a, gen_b]
        else:  # n == 24
            # Source: Robert A. Wilson
            gen_a = _cycles_to_permutation(
                [
                    (
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                    )
                ],
                24,
            )
            gen_b = _cycles_to_permutation(
                [
                    (1, 24),
                    (2, 23),
                    (3, 12),
                    (4, 17),
                    (5, 14),
                    (6, 22),
                    (7, 13),
                    (8, 16),
                    (9, 18),
                    (10, 21),
                    (11, 20),
                    (15, 19),
                ],
                24,
            )
            perm_gens = [gen_a, gen_b]

        print(f"Expanding generators for M_{n} to full group...")
        full_group = self._expand_generators_to_full_group(perm_gens, n)
        return full_group

    def _expand_generators_to_full_group(
        self, perm_gens: list[np.ndarray], num_points: int
    ) -> list[np.ndarray]:
        """Generates all elements of a group from a set of generators using an optimized BFS."""
        group_orders = {11: 7920, 12: 95040, 22: 443520, 23: 10200960, 24: 244823040}
        expected_order = group_orders.get(num_points)

        if num_points >= 23:
            print(f"\nWarning: M{num_points} has {expected_order:,} elements.")
            print(
                f"This will require approximately {expected_order * num_points * 4 / 1024**3:.1f} GB of memory."
            )
            if num_points == 24:
                print(
                    "Consider generating a smaller subset or using a streaming approach."
                )

        identity_tuple = tuple(range(num_points))
        gen_tuples = [tuple(p) for p in perm_gens]
        inv_gen_tuples = []
        for g in gen_tuples:
            inv = [0] * num_points
            for i in range(num_points):
                inv[g[i]] = i
            inv_gen_tuples.append(tuple(inv))
        all_gens = gen_tuples + inv_gen_tuples

        q = deque([identity_tuple])
        group_elements_set = {identity_tuple}
        element_count = 1

        with tqdm(total=expected_order, desc="Expanding group") as pbar:
            pbar.update(1)
            while q:
                current = q.popleft()
                for gen in all_gens:
                    new_perm = tuple(current[gen[i]] for i in range(num_points))
                    if new_perm not in group_elements_set:
                        group_elements_set.add(new_perm)
                        q.append(new_perm)
                        element_count += 1
                        pbar.update(1)
                        if expected_order and element_count >= expected_order:
                            break
                if expected_order and element_count >= expected_order:
                    break

        result = [np.array(perm, dtype=np.int32) for perm in group_elements_set]
        if expected_order and len(result) != expected_order:
            print(
                f"Warning: Generated {len(result)} elements, expected {expected_order}"
            )

        return result

    def get_valid_parameters(self) -> list[dict]:
        """Return a list of valid parameter sets for Mathieu groups."""
        return [{"n": 11}, {"n": 12}, {"n": 22}, {"n": 23}, {"n": 24}]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Mathieu group permutation datasets"
    )
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=[11, 12],
        help="List of Mathieu groups to generate by degree n (e.g., 11 24)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../individual_datasets",
        help="Output directory",
    )
    parser.add_argument(
        "--num-train", type=int, default=10000, help="Number of training samples"
    )
    parser.add_argument(
        "--num-test", type=int, default=2000, help="Number of test samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generator = MathieuGroupGenerator(seed=args.seed)
    output_base = Path(args.output_dir)

    for n in args.groups:
        try:
            print(f"\n{'=' * 60}\nProcessing request for M_{n}\n{'=' * 60}")
            params = {"n": n}
            output_dir = output_base / f"m{n}_data"
            generator.generate_dataset(
                params=params,
                num_train_samples=args.num_train,
                num_test_samples=args.num_test,
                output_dir=output_dir,
            )
            print(f"✓ Successfully processed M_{n}")
        except Exception as e:
            print(f"✗ An unexpected error occurred while generating M_{n}: {e}")


if __name__ == "__main__":
    main()
