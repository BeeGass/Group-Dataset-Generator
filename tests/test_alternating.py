#!/usr/bin/env python3
"""
Exhaustive tests for alternating groups (A3-A12) dynamic filtering.
"""

import pytest
from .test_base import BaseGroupTest


class TestAlternatingGroups(BaseGroupTest):
    """Test alternating groups A3 through A12."""

    GROUP_TYPE = "alternating"
    GROUP_CONFIG = {
        "degrees": list(range(3, 13)),  # A3 to A12
        "orders": {
            3: 3,  # A3 (cyclic group of order 3)
            4: 12,  # A4
            5: 60,  # A5 (smallest non-abelian simple group)
            6: 360,  # A6
            7: 2520,  # A7
            8: 20160,  # A8
            9: 181440,  # A9
            10: 1814400,  # A10
            11: 19958400,  # A11
            12: 239500800,  # A12
        },
        "prefix": "A",
    }

    @pytest.mark.parametrize("degree", list(range(3, 13)))
    def test_specific_degree(self, degree):
        """Test each alternating group degree A3 through A12."""
        super().test_specific_degree(degree)

    def test_alternating_specific_properties(self):
        """Test properties specific to alternating groups."""
        # A3 is the smallest alternating group (cyclic of order 3)
        dataset_a3 = self.load_dataset_with_filters(
            min_degree=3, max_degree=3, split="train"
        )
        assert len(dataset_a3) > 0, "A3 samples not found"
        for i in range(min(10, len(dataset_a3))):
            assert dataset_a3[i]["group_order"] == 3

        # A5 is the smallest non-abelian simple group
        dataset_a5 = self.load_dataset_with_filters(
            min_degree=5, max_degree=5, split="train"
        )
        assert len(dataset_a5) > 0, "A5 samples not found"
        for i in range(min(10, len(dataset_a5))):
            assert dataset_a5[i]["group_order"] == 60

        # A12 is the largest in our set
        dataset_a12 = self.load_dataset_with_filters(
            min_degree=12, max_degree=12, split="train"
        )
        assert len(dataset_a12) > 0, "A12 samples not found"
        for i in range(min(10, len(dataset_a12))):
            assert dataset_a12[i]["group_order"] == 239500800

    def test_alternating_order_formula(self):
        """Verify that alternating groups have order n!/2 for n≥2."""
        import math

        for degree in self.GROUP_CONFIG["degrees"]:
            if degree >= 2:
                expected_order = math.factorial(degree) // 2
                actual_order = self.GROUP_CONFIG["orders"][degree]
                assert actual_order == expected_order, (
                    f"A{degree} order {actual_order} != {degree}!/2 = {expected_order}"
                )

    def test_alternating_vs_symmetric_relationship(self):
        """Test that An has exactly half the order of Sn."""
        # Load symmetric group config for comparison
        symmetric_orders = {
            3: 6,
            4: 24,
            5: 120,
            6: 720,
            7: 5040,
            8: 40320,
            9: 362880,
            10: 3628800,
            11: 39916800,
            12: 479001600,
        }

        for degree in self.GROUP_CONFIG["degrees"]:
            if degree >= 2:
                alternating_order = self.GROUP_CONFIG["orders"][degree]
                symmetric_order = symmetric_orders[degree]
                assert alternating_order * 2 == symmetric_order, (
                    f"A{degree} order {alternating_order} * 2 != S{degree} order {symmetric_order}"
                )

    def test_alternating_simple_groups(self):
        """Test properties of alternating groups that are simple (n≥5)."""
        for degree in range(5, 13):
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            assert len(dataset) > 0, f"A{degree} (simple group) not found"

            # Simple groups should have good representation
            assert len(dataset) >= 50, f"A{degree} (simple group) has too few samples"

    def test_alternating_special_cases(self):
        """Test special cases in alternating groups."""
        # A3 is cyclic (isomorphic to Z3)
        dataset_a3 = self.load_dataset_with_filters(
            min_degree=3, max_degree=3, split="train"
        )
        assert len(dataset_a3) > 0

        # A4 is solvable but not simple
        dataset_a4 = self.load_dataset_with_filters(
            min_degree=4, max_degree=4, split="train"
        )
        assert len(dataset_a4) > 0
        assert dataset_a4[0]["group_order"] == 12

    def test_alternating_length_distribution(self):
        """Test that alternating groups have good length distribution."""
        for degree in [5, 8, 10]:  # Sample some degrees
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            # Collect length distribution
            lengths = {}
            for i in range(min(1000, len(dataset))):
                length = dataset[i]["sequence_length"]
                lengths[length] = lengths.get(length, 0) + 1

            # Should have diverse lengths
            assert len(lengths) > 20, f"A{degree} has poor length diversity"

    def test_alternating_high_degrees(self):
        """Special tests for high-degree alternating groups."""
        for degree in [10, 11, 12]:
            # Test with various length constraints
            length_ranges = [(3, 10), (50, 100), (500, 1024)]

            for min_len, max_len in length_ranges:
                dataset = self.load_dataset_with_filters(
                    min_degree=degree,
                    max_degree=degree,
                    min_len=min_len,
                    max_len=max_len,
                    split="train",
                )

                if len(dataset) > 0:
                    # Verify constraints
                    for i in range(min(20, len(dataset))):
                        assert dataset[i]["group_degree"] == degree
                        assert min_len <= dataset[i]["sequence_length"] <= max_len

    def test_alternating_permutation_validity(self):
        """Test that all permutation IDs are valid for alternating groups."""
        for degree in [3, 6, 9, 12]:  # Sample degrees
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            order = self.GROUP_CONFIG["orders"][degree]

            for i in range(min(100, len(dataset))):
                # Check input sequence IDs
                input_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
                for pid in input_ids:
                    assert 0 <= pid < order, (
                        f"Invalid permutation ID {pid} for A{degree}"
                    )

                # Check target ID
                target_id = int(dataset[i]["target"])
                assert 0 <= target_id < order, (
                    f"Invalid target ID {target_id} for A{degree}"
                )
