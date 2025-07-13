#!/usr/bin/env python3
"""
Generator for PSL (Projective Special Linear) group datasets.
"""

import numpy as np
from typing import List, Dict
from collections import deque
import argparse
from pathlib import Path
import itertools

from ..base_generator import BaseGroupGenerator


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


def get_prime_power(q):
    if q <= 1:
        return None
    if is_prime(q):
        return q, 1
    for p_base in range(2, int(q**0.5) + 1):
        if q % p_base == 0 and is_prime(p_base):
            n_exp = 1
            power = p_base
            while power < q:
                power *= p_base
                n_exp += 1
            if power == q:
                return p_base, n_exp
    return None


# --- Finite Field Arithmetic (largely unchanged) ---
class Polynomial:
    def __init__(self, coeffs, p):
        self.coeffs = [c % p for c in coeffs]
        self.p = p
        self._reduce()

    def _reduce(self):
        while len(self.coeffs) > 1 and self.coeffs[-1] == 0:
            self.coeffs.pop()

    def degree(self):
        return len(self.coeffs) - 1

    def __str__(self):
        if self.degree() < 0:
            return "0"
        s = []
        for i, c in enumerate(self.coeffs):
            if c == 0:
                continue
            if i == 0:
                s.append(str(c))
            elif i == 1:
                s.append(f"{c}*x" if c != 1 else "x")
            else:
                s.append(f"{c}*x^{i}" if c != 1 else f"x^{i}")
        return " + ".join(reversed(s))

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        max_len = max(len(self.coeffs), len(other.coeffs))
        new_coeffs = [
            (self.coeffs[i] if i < len(self.coeffs) else 0)
            + (other.coeffs[i] if i < len(other.coeffs) else 0)
            for i in range(max_len)
        ]
        return Polynomial(new_coeffs, self.p)

    def __sub__(self, other):
        max_len = max(len(self.coeffs), len(other.coeffs))
        new_coeffs = [
            (self.coeffs[i] if i < len(self.coeffs) else 0)
            - (other.coeffs[i] if i < len(other.coeffs) else 0)
            for i in range(max_len)
        ]
        return Polynomial(new_coeffs, self.p)

    def __mul__(self, other):
        new_deg = self.degree() + other.degree()
        new_coeffs = [0] * (new_deg + 1)
        for i, c1 in enumerate(self.coeffs):
            for j, c2 in enumerate(other.coeffs):
                new_coeffs[i + j] += c1 * c2
        return Polynomial(new_coeffs, self.p)

    def __mod__(self, other):
        if other.degree() < 0:
            raise ZeroDivisionError
        num = Polynomial(self.coeffs[:], self.p)
        den = other
        if num.degree() < den.degree():
            return num
        while num.degree() >= den.degree():
            deg_diff = num.degree() - den.degree()
            inv = pow(den.coeffs[-1], -1, self.p)
            coeff = (num.coeffs[-1] * inv) % self.p
            term = Polynomial([0] * deg_diff + [coeff], self.p)
            num = num - (term * den)  # Use subtraction method
        return num

    def __eq__(self, other):
        return self.coeffs == other.coeffs and self.p == self.p

    def evaluate(self, x):
        res = 0
        for i, c in enumerate(self.coeffs):
            res = (res + c * pow(x, i, self.p)) % self.p
        return res


def find_irreducible_poly(p, n):
    if n == 1:
        return Polynomial([0, 1], p)
    for coeffs_tuple in itertools.product(range(p), repeat=n):
        poly = Polynomial(list(coeffs_tuple) + [1], p)
        if not any(poly.evaluate(i) == 0 for i in range(p)):
            return poly
    raise ValueError(f"Could not find an irreducible polynomial for F_{p}^{n}")


class FiniteField:
    def __init__(self, p, n):
        if not is_prime(p):
            raise ValueError("p must be a prime number.")
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        self.p, self.n, self.q = p, n, p**n
        if n == 1:
            self.reducing_poly = None
        elif p == 2 and n == 2:
            self.reducing_poly = Polynomial([1, 1, 1], 2)
        elif p == 3 and n == 2:
            self.reducing_poly = Polynomial([1, 0, 1], 3)
        elif p == 2 and n == 3:
            self.reducing_poly = Polynomial([1, 1, 0, 1], 2)
        else:
            self.reducing_poly = find_irreducible_poly(p, n)
        self.zero = FieldElement([0], self)
        self.one = FieldElement([1], self)

    def __str__(self):
        return f"Finite Field F_{self.p}^{self.n} (F_{self.q})"


class FieldElement:
    def __init__(self, coeffs, field):
        self.poly = Polynomial(coeffs, field.p)
        self.field = field
        if field.n > 1:
            self.poly = self.poly % field.reducing_poly

    @classmethod
    def from_integer(cls, i, field):
        coeffs = []
        temp_i = i
        while temp_i > 0:
            coeffs.append(temp_i % field.p)
            temp_i //= field.p
        return cls(coeffs if coeffs else [0], field)

    def __add__(self, other):
        return FieldElement((self.poly + other.poly).coeffs, self.field)

    def __sub__(self, other):
        return FieldElement((self.poly - other.poly).coeffs, self.field)

    def __mul__(self, other):
        res_poly = self.poly * other.poly
        if self.field.n > 1:
            res_poly = res_poly % self.field.reducing_poly
        return FieldElement(res_poly.coeffs, self.field)

    def inverse(self):
        if self == self.field.zero:
            raise ZeroDivisionError
        return self ** (self.field.q - 2)

    def __pow__(self, n):
        """Exponentiation for field elements."""
        if n == 0:
            return self.field.one
        if n < 0:
            return self.inverse() ** (-n)

        # Binary exponentiation
        result = self.field.one
        base = self
        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2
        return result

    def __eq__(self, other):
        return self.poly == other.poly and self.field == other.field

    def __hash__(self):
        return hash(tuple(self.poly.coeffs))

    def __repr__(self):
        return str(self.poly)


# --- Matrix Generator Functions ---
def generate_psl2_matrices(p, n):
    F = FiniteField(p, n)
    order = F.q - 1
    gen_a = None
    for i in range(1, F.q):
        elem = FieldElement.from_integer(i, F)
        if elem == F.zero:
            continue
        powers = set()
        current = F.one
        is_generator = True
        for _ in range(order):
            current *= elem
            if current in powers:
                is_generator = False
                break
            powers.add(current)
        if is_generator and len(powers) == order:
            gen_a = elem
            break
    if gen_a is None:
        gen_a = FieldElement([0, 1], F)  # Fallback
    zero, one, neg_one = F.zero, F.one, F.zero - F.one
    S = np.array([[zero, neg_one], [one, zero]], dtype=object)
    T_a = np.array([[one, gen_a], [zero, one]], dtype=object)
    return [S, T_a], F


def generate_psl3_matrices(p, n):
    F = FiniteField(p, n)
    one, zero = F.one, F.zero
    gens = {
        "E12(1)": [[one, one, zero], [zero, one, zero], [zero, zero, one]],
        "E21(1)": [[one, zero, zero], [one, one, zero], [zero, zero, one]],
        "E23(1)": [[one, zero, zero], [zero, one, one], [zero, zero, one]],
        "E32(1)": [[one, zero, zero], [zero, one, zero], [zero, one, one]],
    }
    return [np.array(m, dtype=object) for m in gens.values()], F


# --- Main Generator Class ---
class PSLGroupGenerator(BaseGroupGenerator):
    """Generator for PSL(dim,q) groups as permutations."""

    def get_group_name(self) -> str:
        return "psl"

    def generate_group(self, dim: int, p: int, n: int) -> list[np.ndarray]:
        # Special case: PSL(3,2) ≅ PSL(2,7)
        if dim == 3 and p == 2 and n == 1:
            print("PSL(3,2) is isomorphic to PSL(2,7), generating PSL(2,7) instead")
            return self.generate_group(dim=2, p=7, n=1)

        if dim == 2:
            matrix_gens, F = generate_psl2_matrices(p=p, n=n)
        elif dim == 3:
            matrix_gens, F = generate_psl3_matrices(p=p, n=n)
        else:
            raise ValueError(f"PSL({dim},q) not implemented.")

        print("Finding points in projective space...")
        points = self._get_projective_space_points(dim, F)
        point_map = {tuple(p): i for i, p in enumerate(points)}

        print("Converting matrix generators to permutation generators...")
        perm_gens = self._convert_matrices_to_permutations(
            matrix_gens, points, point_map, F
        )

        print("Expanding generators to full group...")
        full_group = self._expand_generators_to_full_group(perm_gens, len(points))
        return full_group

    def _get_projective_space_points(
        self, dim: int, F: FiniteField
    ) -> list[np.ndarray]:
        points = []
        seen_lines = set()

        # Iterate through all vectors in F_q^dim
        for vec_coords in itertools.product(range(F.q), repeat=dim):
            if all(c == 0 for c in vec_coords):
                continue

            vec = np.array(
                [FieldElement.from_integer(c, F) for c in vec_coords], dtype=object
            )

            # Normalize to get a canonical representative for the line
            first_nonzero_idx = next(i for i, x in enumerate(vec) if x != F.zero)
            inv = vec[first_nonzero_idx].inverse()
            normalized_vec = np.array([v * inv for v in vec], dtype=object)

            if tuple(normalized_vec) not in seen_lines:
                points.append(normalized_vec)
                seen_lines.add(tuple(normalized_vec))
        return points

    def _convert_matrices_to_permutations(self, matrix_gens, points, point_map, F):
        perm_gens = []
        num_points = len(points)
        for M in matrix_gens:
            perm = np.zeros(num_points, dtype=int)
            for i, point_vec in enumerate(points):
                # Apply matrix to vector: Mv
                new_vec = M @ point_vec

                # Normalize the resulting vector to find its point
                first_nonzero_idx = next(
                    j for j, x in enumerate(new_vec) if x != F.zero
                )
                inv = new_vec[first_nonzero_idx].inverse()
                normalized_new_vec = np.array([v * inv for v in new_vec], dtype=object)

                # Find the index of the new point
                j = point_map[tuple(normalized_new_vec)]
                perm[i] = j
            perm_gens.append(perm)
        return perm_gens

    def _expand_generators_to_full_group(self, perm_gens, num_points):
        identity = np.arange(num_points)
        q = deque([identity])
        group_elements = {tuple(identity)}

        while q:
            current = q.popleft()
            for gen in perm_gens:
                # Composition: current o gen
                new_perm = current[gen]
                if tuple(new_perm) not in group_elements:
                    group_elements.add(tuple(new_perm))
                    q.append(new_perm)

        return [np.array(p) for p in group_elements]

    def get_valid_parameters(self) -> list[dict]:
        return [
            {"dim": 2, "p": 3, "n": 1},
            {"dim": 2, "p": 5, "n": 1},
            {"dim": 3, "p": 2, "n": 1},
        ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate PSL group permutation datasets"
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["2,3", "2,5", "3,2"],
        help="Groups as dim,q pairs (e.g., 2,9)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../datasets",
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

    generator = PSLGroupGenerator(seed=args.seed)
    output_base = Path(args.output_dir)

    for group_str in args.groups:
        try:
            dim, q = map(int, group_str.split(","))
            print(f"\n{'=' * 60}\nProcessing request for PSL({dim},{q})\n{'=' * 60}")
            p_n_pair = get_prime_power(q)
            if not p_n_pair:
                print(f"✗ Error: q={q} is not a valid prime power. Skipping.")
                continue
            p, n = p_n_pair
            params = {"dim": dim, "p": p, "n": n}
            output_dir = output_base / f"psl{dim}_{q}_data"
            generator.generate_dataset(
                params=params,
                num_train_samples=args.num_train,
                num_test_samples=args.num_test,
                output_dir=output_dir,
            )
            print(f"✓ Successfully processed PSL({dim},{q})")
        except Exception as e:
            print(
                f"✗ An unexpected error occurred while generating PSL({dim},{q}): {e}"
            )


if __name__ == "__main__":
    main()
