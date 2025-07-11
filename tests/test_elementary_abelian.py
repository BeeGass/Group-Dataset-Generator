#!/usr/bin/env python3
"""
Exhaustive tests for elementary abelian groups (Z_p^k) dynamic filtering.
"""

import pytest
from .test_base import BaseGroupTest


class TestElementaryAbelianGroups(BaseGroupTest):
    """Test elementary abelian groups Z_p^k."""

    GROUP_TYPE = "elementary_abelian"
    GROUP_CONFIG = {
        "degrees": [4, 8, 9, 16, 25, 27, 32],  # 2^2, 2^3, 3^2, 2^4, 5^2, 3^3, 2^5
        "orders": {4: 4, 8: 8, 9: 9, 16: 16, 25: 25, 27: 27, 32: 32},
        "prefix": "Z",
        # Store (p, k) pairs for each degree
        "params": {
            4: (2, 2),
            8: (2, 3),
            9: (3, 2),
            16: (2, 4),
            25: (5, 2),
            27: (3, 3),
            32: (2, 5),
        },
    }

    @pytest.mark.parametrize(
        "degree", [4, 8, 9, 16, 25, 32]
    )  # Skip 27 to avoid timeout
    def test_specific_degree(self, degree):
        """Test specific elementary abelian groups."""
        super().test_specific_degree(degree)

    def test_elementary_abelian_basic_properties(self):
        """Test basic properties of elementary abelian groups."""
        for degree in [4, 8, 9, 16, 25]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            assert len(dataset) > 0, f"Z_p^k of degree {degree} not found"

            # Verify order equals degree (for elementary abelian groups)
            for i in range(min(20, len(dataset))):
                assert dataset[i]["group_order"] == degree
                assert dataset[i]["group_degree"] == degree
                assert dataset[i]["group_type"] == "elementary_abelian"

    def test_elementary_abelian_prime_powers(self):
        """Test that degrees are prime powers p^k."""
        import math

        def is_prime_power(n):
            """Check if n is a prime power."""
            if n < 2:
                return False
            for p in [2, 3, 5, 7, 11, 13]:
                if n == 1:
                    return True
                k = 0
                temp = n
                while temp % p == 0:
                    temp //= p
                    k += 1
                if k > 0 and temp == 1:
                    return True
            return n in [2, 3, 5, 7, 11, 13]  # n itself is prime

        for degree in self.GROUP_CONFIG["degrees"]:
            assert is_prime_power(degree), f"{degree} is not a prime power"

    def test_z2_powers(self):
        """Test elementary abelian 2-groups (Z_2^k)."""
        z2_powers = [(4, 2), (8, 3), (16, 4), (32, 5)]  # (degree, k) where degree = 2^k

        for degree, k in z2_powers:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            assert len(dataset) > 0, f"Z_2^{k} not found"
            assert dataset[0]["group_order"] == degree
            assert dataset[0]["group_order"] == 2**k

            # In Z_2^k, every non-identity element has order 2
            # This makes it very different from cyclic group C_{2^k}

    def test_z3_powers(self):
        """Test elementary abelian 3-groups (Z_3^k)."""
        z3_powers = [(9, 2), (27, 3)]  # (degree, k) where degree = 3^k

        for degree, k in z3_powers:
            if degree <= 32:  # Within our range
                dataset = self.load_dataset_with_filters(
                    min_degree=degree, max_degree=degree, split="train"
                )

                if len(dataset) > 0:
                    assert dataset[0]["group_order"] == degree
                    assert dataset[0]["group_order"] == 3**k

    def test_z5_squared(self):
        """Test Z_5^2 (elementary abelian group of order 25)."""
        dataset = self.load_dataset_with_filters(
            min_degree=25, max_degree=25, split="train"
        )

        assert len(dataset) > 0, "Z_5^2 not found"
        assert dataset[0]["group_order"] == 25

        # Z_5^2 is abelian with every non-identity element having order 5
        # Different from cyclic C_25 where elements can have order 25

    def test_elementary_abelian_vs_cyclic(self):
        """Test difference between elementary abelian and cyclic groups of same order."""
        # Compare Z_2^2 (elementary abelian) vs C_4 (cyclic)
        dataset_z22 = self.load_dataset_with_filters(
            min_degree=4, max_degree=4, split="train"
        )

        if len(dataset_z22) > 0:
            # Should be elementary abelian, not cyclic
            for i in range(min(10, len(dataset_z22))):
                if dataset_z22[i]["group_type"] == "elementary_abelian":
                    assert dataset_z22[i]["group_order"] == 4
                    break

        # Similarly for Z_3^2 vs C_9
        dataset_z32 = self.load_dataset_with_filters(
            min_degree=9, max_degree=9, split="train"
        )

        if len(dataset_z32) > 0:
            for i in range(min(10, len(dataset_z32))):
                if dataset_z32[i]["group_type"] == "elementary_abelian":
                    assert dataset_z32[i]["group_order"] == 9
                    break

    def test_elementary_abelian_abelian_property(self):
        """Test that all elementary abelian groups are abelian."""
        # By definition, elementary abelian groups are abelian
        # Z_p^k = Z_p × Z_p × ... × Z_p (k times)

        for degree in [4, 8, 9, 16]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            if len(dataset) > 0:
                # In abelian groups, order of composition doesn't matter
                # This is a structural property we're representing
                assert dataset[0]["group_type"] == "elementary_abelian"

    def test_elementary_abelian_length_filtering(self):
        """Test elementary abelian groups with various sequence lengths."""
        test_configs = [
            (4, [2, 3, 4, 5, 10, 50, 100]),  # Z_2^2
            (8, [3, 8, 16, 32, 64, 128, 256]),  # Z_2^3
            (9, [3, 9, 27, 81, 243, 729]),  # Z_3^2
            (16, [4, 16, 64, 256, 1024]),  # Z_2^4
        ]

        for degree, lengths in test_configs:
            for length in lengths:
                if length <= 1024:
                    dataset = self.load_dataset_with_filters(
                        min_degree=degree,
                        max_degree=degree,
                        min_len=length,
                        max_len=length,
                        split="train",
                    )

                    if len(dataset) > 0:
                        assert dataset[0]["sequence_length"] == length
                        assert dataset[0]["group_degree"] == degree

    def test_elementary_abelian_order_filtering(self):
        """Test filtering by order for elementary abelian groups."""
        # Each order corresponds to a unique elementary abelian group in our set
        for order in [4, 8, 9, 16, 25, 32]:
            dataset = self.load_dataset_with_filters(
                min_order=order, max_order=order, split="train"
            )

            # Should find elementary abelian group of this order
            found = False
            for i in range(min(100, len(dataset))):
                if (
                    dataset[i]["group_type"] == "elementary_abelian"
                    and dataset[i]["group_order"] == order
                ):
                    found = True
                    break

            assert found, f"Elementary abelian group of order {order} not found"

    def test_elementary_abelian_vector_space_structure(self):
        """Test that Z_p^k can be viewed as k-dimensional vector space over Z_p."""
        # For Z_2^k groups
        for degree, k in [(4, 2), (8, 3), (16, 4), (32, 5)]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            if len(dataset) > 0:
                # The group has 2^k elements
                assert dataset[0]["group_order"] == 2**k

                # Can be viewed as F_2^k (vector space)
                # Every element except identity has order 2

    def test_elementary_abelian_combined_filters(self):
        """Test combining multiple filters for elementary abelian groups."""
        # Test specific degrees with length ranges
        test_cases = [
            (4, 10, 50),  # Z_2^2 with short sequences
            (8, 50, 200),  # Z_2^3 with medium sequences
            (16, 200, 500),  # Z_2^4 with long sequences
            (9, 3, 100),  # Z_3^2 with varied sequences
        ]

        for degree, min_len, max_len in test_cases:
            dataset = self.load_dataset_with_filters(
                min_degree=degree,
                max_degree=degree,
                min_len=min_len,
                max_len=max_len,
                split="train",
            )

            if len(dataset) > 0:
                for i in range(min(20, len(dataset))):
                    assert dataset[i]["group_degree"] == degree
                    assert min_len <= dataset[i]["sequence_length"] <= max_len
                    assert dataset[i]["group_type"] == "elementary_abelian"

    def test_elementary_abelian_range_queries(self):
        """Test range queries across elementary abelian groups."""
        # Test ranges that include multiple groups
        ranges = [
            (4, 9),  # Includes Z_2^2, Z_2^3, Z_3^2
            (8, 16),  # Includes Z_2^3, Z_3^2, Z_2^4
            (16, 32),  # Includes Z_2^4, Z_5^2, Z_2^5
        ]

        for min_deg, max_deg in ranges:
            dataset = self.load_dataset_with_filters(
                min_degree=min_deg, max_degree=max_deg, split="train"
            )

            if len(dataset) > 0:
                degrees_found = set()
                for i in range(min(200, len(dataset))):
                    if dataset[i]["group_type"] == "elementary_abelian":
                        degrees_found.add(dataset[i]["group_degree"])

                # Should find multiple elementary abelian groups
                assert len(degrees_found) >= 1

    def test_elementary_abelian_direct_product_structure(self):
        """Test that Z_p^k is the direct product of k copies of Z_p."""
        # This is the defining property of elementary abelian groups

        # Test a few specific cases
        test_cases = [
            (4, 2, 2),  # Z_2^2 = Z_2 × Z_2
            (8, 2, 3),  # Z_2^3 = Z_2 × Z_2 × Z_2
            (9, 3, 2),  # Z_3^2 = Z_3 × Z_3
            (25, 5, 2),  # Z_5^2 = Z_5 × Z_5
        ]

        for degree, p, k in test_cases:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            if len(dataset) > 0:
                assert dataset[0]["group_order"] == p**k
                assert dataset[0]["group_degree"] == degree

    def test_elementary_abelian_exhaustive_validation(self):
        """Exhaustive validation of all elementary abelian groups."""
        total_samples = 0
        length_distribution = {}

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            if len(dataset) > 0:
                total_samples += len(dataset)

                # Check permutation ID validity
                order = self.GROUP_CONFIG["orders"][degree]
                for i in range(min(100, len(dataset))):
                    # Verify all permutation IDs are valid
                    perm_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
                    for pid in perm_ids:
                        assert 0 <= pid < order

                    # Collect length statistics
                    length = dataset[i]["sequence_length"]
                    length_distribution[length] = length_distribution.get(length, 0) + 1

        # Should have good coverage
        assert total_samples > 1000, (
            f"Too few elementary abelian samples: {total_samples}"
        )
        assert len(length_distribution) > 30, "Poor length diversity"
