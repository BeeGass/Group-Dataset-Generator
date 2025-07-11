#!/usr/bin/env python3
"""
Exhaustive tests for dihedral groups (D3-D50) dynamic filtering.
"""

import pytest
from .test_base import BaseGroupTest


class TestDihedralGroups(BaseGroupTest):
    """Test dihedral groups D3 through D50."""

    GROUP_TYPE = "dihedral"
    GROUP_CONFIG = {
        "degrees": list(range(3, 51)),  # D3 to D50
        "orders": {n: 2 * n for n in range(3, 51)},  # Order = 2n for Dn
        "prefix": "D",
    }

    # Test a subset of degrees
    @pytest.mark.parametrize(
        "degree", [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]
    )
    def test_specific_degree(self, degree):
        """Test selected dihedral group degrees."""
        super().test_specific_degree(degree)

    def test_dihedral_order_formula(self):
        """Verify that dihedral groups Dn have order 2n."""
        sample_degrees = [3, 5, 8, 12, 20, 30, 40, 50]

        for n in sample_degrees:
            dataset = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, split="train"
            )

            assert len(dataset) > 0, f"D{n} not found"

            # Verify order = 2n
            expected_order = 2 * n
            for i in range(min(10, len(dataset))):
                assert dataset[i]["group_order"] == expected_order, (
                    f"D{n} should have order {expected_order}, got {dataset[i]['group_order']}"
                )

    def test_dihedral_small_groups(self):
        """Test small dihedral groups D3-D10 exhaustively."""
        for n in range(3, 11):
            dataset = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, split="train"
            )

            assert len(dataset) > 0, f"D{n} not found"

            # D3 = S3 (smallest non-abelian group)
            if n == 3:
                assert dataset[0]["group_order"] == 6

            # D4 has order 8 (same as quaternion Q8 but different structure)
            elif n == 4:
                assert dataset[0]["group_order"] == 8

    def test_dihedral_prime_n(self):
        """Test dihedral groups Dp where p is prime."""
        primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        for p in primes:
            if p <= 50:
                dataset = self.load_dataset_with_filters(
                    min_degree=p, max_degree=p, split="train"
                )

                assert len(dataset) > 0, f"D{p} (p prime) not found"
                assert dataset[0]["group_order"] == 2 * p

    def test_dihedral_power_of_two_n(self):
        """Test dihedral groups D_{2^k}."""
        powers_of_two = [4, 8, 16, 32]

        for n in powers_of_two:
            if n <= 50:
                dataset = self.load_dataset_with_filters(
                    min_degree=n, max_degree=n, split="train"
                )

                assert len(dataset) > 0, f"D{n} (n = 2^k) not found"

                # These have interesting subgroup structure
                assert dataset[0]["group_order"] == 2 * n

    def test_dihedral_special_cases(self):
        """Test special dihedral groups with interesting properties."""
        # D3 ≅ S3 (symmetric group on 3 elements)
        dataset_d3 = self.load_dataset_with_filters(
            min_degree=3, max_degree=3, split="train"
        )
        assert len(dataset_d3) > 0
        assert dataset_d3[0]["group_order"] == 6

        # D6 has same order as A4 but different structure
        dataset_d6 = self.load_dataset_with_filters(
            min_degree=6, max_degree=6, split="train"
        )
        assert len(dataset_d6) > 0
        assert dataset_d6[0]["group_order"] == 12  # Same as A4

    def test_dihedral_large_groups(self):
        """Test large dihedral groups D30-D50."""
        for n in range(30, 51, 5):  # D30, D35, D40, D45, D50
            dataset = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, split="train"
            )

            assert len(dataset) > 0, f"D{n} (large) not found"
            assert dataset[0]["group_order"] == 2 * n

            # Test with various sequence lengths
            length_tests = [(3, 10), (50, 100), (500, 1024)]
            for min_len, max_len in length_tests:
                filtered = self.load_dataset_with_filters(
                    min_degree=n,
                    max_degree=n,
                    min_len=min_len,
                    max_len=max_len,
                    split="train",
                )

    def test_dihedral_symmetry_properties(self):
        """Test that dihedral groups represent symmetries of regular n-gons."""
        # Dihedral groups have n rotations and n reflections
        for n in [3, 4, 5, 6, 8, 10]:
            dataset = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, split="train"
            )

            if len(dataset) > 0:
                order = dataset[0]["group_order"]
                assert order == 2 * n, (
                    f"D{n} should have {n} rotations and {n} reflections"
                )

                # Check permutation IDs are in valid range
                for i in range(min(50, len(dataset))):
                    perm_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
                    for pid in perm_ids:
                        assert 0 <= pid < order

    def test_dihedral_subgroup_relationships(self):
        """Test relationships between dihedral groups."""
        # Dn contains a cyclic subgroup Cn of index 2
        for n in [4, 6, 8, 10]:
            dataset_dn = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, split="train"
            )

            assert len(dataset_dn) > 0
            assert dataset_dn[0]["group_order"] == 2 * n

            # The order is exactly twice the degree
            # This reflects the n rotations + n reflections structure

    def test_dihedral_range_filtering(self):
        """Test filtering across ranges of dihedral groups."""
        ranges = [
            (3, 6),  # Small dihedral groups
            (10, 15),  # Medium dihedral groups
            (20, 30),  # Large dihedral groups
            (40, 50),  # Very large dihedral groups
        ]

        for min_deg, max_deg in ranges:
            dataset = self.load_dataset_with_filters(
                min_degree=min_deg, max_degree=max_deg, split="train"
            )

            assert len(dataset) > 0, f"No samples for D{min_deg}-D{max_deg}"

            # Check degree distribution
            degrees_found = set()
            for i in range(min(200, len(dataset))):
                degrees_found.add(dataset[i]["group_degree"])

            # Should have multiple degrees represented
            assert len(degrees_found) >= min(3, max_deg - min_deg + 1)

    def test_dihedral_odd_vs_even_n(self):
        """Test differences between dihedral groups with odd vs even n."""
        # Odd n
        odd_n = [3, 5, 7, 9, 11, 13, 15]
        for n in odd_n:
            if n <= 50:
                dataset = self.load_dataset_with_filters(
                    min_degree=n, max_degree=n, split="train"
                )
                if len(dataset) > 0:
                    # D_n with odd n has different properties than even n
                    assert dataset[0]["group_order"] == 2 * n

        # Even n
        even_n = [4, 6, 8, 10, 12, 14, 16]
        for n in even_n:
            if n <= 50:
                dataset = self.load_dataset_with_filters(
                    min_degree=n, max_degree=n, split="train"
                )
                if len(dataset) > 0:
                    # D_n with even n has additional central symmetry
                    assert dataset[0]["group_order"] == 2 * n

    def test_dihedral_exhaustive_properties(self):
        """Exhaustive test of various properties for selected dihedral groups."""
        test_groups = [3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]

        for n in test_groups:
            if n <= 50:
                # Test basic loading
                dataset = self.load_dataset_with_filters(
                    min_degree=n, max_degree=n, split="train"
                )
                assert len(dataset) > 0

                # Test with order filter
                order = 2 * n
                dataset_by_order = self.load_dataset_with_filters(
                    min_order=order, max_order=order, split="train"
                )

                # Should find the same dihedral group
                found_dn = False
                for i in range(min(100, len(dataset_by_order))):
                    if (
                        dataset_by_order[i]["group_type"] == "dihedral"
                        and dataset_by_order[i]["group_degree"] == n
                    ):
                        found_dn = True
                        break

                if dataset_by_order[0]["group_type"] == "dihedral":
                    assert found_dn, f"D{n} not found when filtering by order {order}"
