#!/usr/bin/env python3
"""
Generator for Frobenius group datasets.

This script implements a general construction for a Frobenius group of order n = mk,
where m is a prime number. The group is constructed as a semi-direct product of
the cyclic group C_m and a subgroup of Aut(C_m) of order k. This is represented
as a permutation group on m points.
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


# --- Helper Functions ---
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


class FrobeniusGroupGenerator(BaseGroupGenerator):
    """Generator for Frobenius groups of order n=mk as permutations on m points."""

    def get_group_name(self) -> str:
        return "frobenius"

    def generate_group(self, n: int) -> list[np.ndarray]:
        """
        Generates a Frobenius group of order n.
        The group is constructed as C_m ⋊ C_k where n = mk, m is prime,
        and k divides m-1. The permutation is on m points.
        """
        m, k = self._find_frobenius_factors(n)
        if m is None:
            raise ValueError(
                f"Cannot construct a standard Frobenius group of order {n}. "
                "Requires order n=mk where m is prime and k divides m-1."
            )

        degree = m
        print(
            f"Constructing Frobenius group of order {n} (degree {m}) as C_{m} ⋊ C_{k}"
        )

        # Generator 1: Translation x -> x + 1 (mod m)
        # This generates the Frobenius kernel K, isomorphic to C_m.
        translation_gen = np.roll(np.arange(degree), -1)

        # Generator 2: Multiplication x -> a*x (mod m)
        # This generates the Frobenius complement H, isomorphic to C_k.
        # We need to find an element 'a' of order k in the multiplicative group (Z/mZ)*.
        multiplicative_gen_val = self._find_multiplicative_generator(m, k)

        multiplication_gen = np.array(
            [(i * multiplicative_gen_val) % m for i in range(degree)]
        )

        perm_gens = [translation_gen, multiplication_gen]

        print(f"Expanding generators for Frobenius group of order {n}...")
        full_group = self._expand_generators_to_full_group(perm_gens, degree, n)
        return full_group

    def _find_frobenius_factors(self, n: int) -> tuple[int, int] | None:
        """Finds factors m, k for a Frobenius group of order n=mk."""
        for m in range(2, n):
            if n % m == 0 and is_prime(m):
                k = n // m
                if k > 0 and (m - 1) % k == 0:
                    return m, k
        return None, None

    def _find_multiplicative_generator(self, m: int, k: int) -> int:
        """Finds an element of order k in the multiplicative group (Z/mZ)*."""
        if k == 1:
            return 1

        # Find a primitive root g for (Z/mZ)*
        primitive_root = self._get_primitive_root(m)

        # An element of order k is g^((m-1)/k)
        power = (m - 1) // k
        return pow(primitive_root, power, m)

    def _get_primitive_root(self, p: int) -> int:
        """Finds a primitive root modulo a prime p."""
        if not is_prime(p):
            raise ValueError("p must be a prime number.")

        phi = p - 1
        prime_factors = self._get_prime_factors(phi)

        for g in range(2, p):
            is_primitive = True
            for factor in prime_factors:
                if pow(g, phi // factor, p) == 1:
                    is_primitive = False
                    break
            if is_primitive:
                return g
        raise RuntimeError(f"Could not find a primitive root for {p}")

    def _get_prime_factors(self, num: int) -> set:
        factors = set()
        d = 2
        temp_num = num
        while d * d <= temp_num:
            if temp_num % d == 0:
                factors.add(d)
                while temp_num % d == 0:
                    temp_num //= d
            d += 1
        if temp_num > 1:
            factors.add(temp_num)
        return factors

    def _expand_generators_to_full_group(
        self, perm_gens: list[np.ndarray], degree: int, expected_order: int
    ) -> list[np.ndarray]:
        identity_tuple = tuple(range(degree))
        gen_tuples = [tuple(p) for p in perm_gens]

        q = deque([identity_tuple])
        group_elements_set = {identity_tuple}

        with tqdm(total=expected_order, desc="Expanding group") as pbar:
            while q:
                current = q.popleft()
                for gen in gen_tuples:
                    new_perm = tuple(current[gen[i]] for i in range(degree))
                    if new_perm not in group_elements_set:
                        group_elements_set.add(new_perm)
                        q.append(new_perm)
                        pbar.update(1)
                        if len(group_elements_set) == expected_order:
                            q.clear()
                            break
                if len(group_elements_set) == expected_order:
                    break

        pbar.n = len(group_elements_set)
        pbar.refresh()

        if len(group_elements_set) != expected_order:
            print(
                f"Warning: Generated {len(group_elements_set)} elements, expected {expected_order}"
            )

        return [np.array(p, dtype=np.int32) for p in group_elements_set]

    def get_valid_parameters(self) -> list[dict]:
        return [{"n": 20}, {"n": 21}, {"n": 42}, {"n": 55}, {"n": 56}]


def main():
    """Generate Frobenius group datasets."""

    parser = argparse.ArgumentParser(description="Generate Frobenius group datasets")
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=[20, 21],
        help="Frobenius groups to generate (default: 20, 21)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../individual_datasets",
        help="Output directory",
    )
    parser.add_argument(
        "--num-train", type=int, default=80000, help="Number of training samples"
    )
    parser.add_argument(
        "--num-test", type=int, default=20000, help="Number of test samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generator = FrobeniusGroupGenerator(seed=args.seed)
    output_base = Path(args.output_dir)

    for n in args.groups:
        print(f"\n{'=' * 60}")
        print(f"Generating F{n}")
        print("=" * 60)

        try:
            output_dir = output_base / f"f{n}_data"
            dataset_dict, metadata = generator.generate_dataset(
                params={"n": n},
                num_train_samples=args.num_train,
                num_test_samples=args.num_test,
                output_dir=output_dir,
            )

            print(f"✓ Successfully generated F{n}")

        except Exception as e:
            print(f"✗ Error generating F{n}: {e}")


if __name__ == "__main__":
    main()
