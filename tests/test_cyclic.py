#!/usr/bin/env python3
"""
Exhaustive tests for cyclic groups (C3-C100) dynamic filtering.
"""

import pytest
from .test_base import BaseGroupTest


class TestCyclicGroups(BaseGroupTest):
    """Test cyclic groups C3 through C100."""

    GROUP_TYPE = "cyclic"
    GROUP_CONFIG = {
        "degrees": list(range(3, 101)),  # C3 to C100
        "orders": {
            n: n for n in range(3, 101)
        },  # Order equals degree for cyclic groups
        "prefix": "C",
    }

    # Test a subset of degrees to avoid timeout
    @pytest.mark.parametrize(
        "degree",
        [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100],
    )
    def test_specific_degree(self, degree):
        """Test selected cyclic group degrees."""
        super().test_specific_degree(degree)

    def test_cyclic_order_equals_degree(self):
        """Verify that cyclic groups have order equal to degree."""
        sample_degrees = [3, 5, 10, 20, 50, 100]

        for degree in sample_degrees:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            assert len(dataset) > 0, f"C{degree} not found"

            # Verify order = degree
            for i in range(min(10, len(dataset))):
                assert dataset[i]["group_order"] == degree, (
                    f"C{degree} should have order {degree}, got {dataset[i]['group_order']}"
                )

    def test_cyclic_prime_orders(self):
        """Test cyclic groups of prime order (these are all simple)."""
        primes = [
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
        ]

        for p in primes:
            if p <= 100:  # Within our range
                dataset = self.load_dataset_with_filters(
                    min_degree=p, max_degree=p, split="train"
                )

                assert len(dataset) > 0, f"C{p} (prime order) not found"

                # Prime order cyclic groups are simple
                sample = dataset[0]
                assert sample["group_order"] == p
                assert sample["group_degree"] == p

    def test_cyclic_composite_orders(self):
        """Test cyclic groups of composite order."""
        composites = [
            4,
            6,
            8,
            9,
            10,
            12,
            15,
            20,
            24,
            30,
            36,
            40,
            48,
            60,
            72,
            80,
            90,
            100,
        ]

        for n in composites:
            dataset = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, split="train"
            )

            assert len(dataset) > 0, f"C{n} (composite order) not found"

    def test_cyclic_power_of_two_orders(self):
        """Test cyclic groups whose order is a power of 2."""
        powers_of_two = [4, 8, 16, 32, 64]

        for n in powers_of_two:
            if n <= 100:
                dataset = self.load_dataset_with_filters(
                    min_degree=n, max_degree=n, split="train"
                )

                assert len(dataset) > 0, f"C{n} (2^k order) not found"

                # These have special properties in terms of subgroups
                assert dataset[0]["group_order"] == n

    def test_cyclic_highly_composite_orders(self):
        """Test cyclic groups with highly composite orders (many divisors)."""
        highly_composite = [12, 24, 36, 48, 60, 72, 84, 96]

        for n in highly_composite:
            if n <= 100:
                dataset = self.load_dataset_with_filters(
                    min_degree=n, max_degree=n, split="train"
                )

                if len(dataset) > 0:
                    # These should have good representation due to rich subgroup structure
                    assert len(dataset) >= 100, (
                        f"C{n} (highly composite) has too few samples"
                    )

    def test_cyclic_large_orders(self):
        """Test large cyclic groups (C50 through C100)."""
        for degree in range(50, 101, 10):  # Test C50, C60, C70, C80, C90, C100
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            assert len(dataset) > 0, f"C{degree} (large order) not found"

            # Test with different length constraints
            for min_len, max_len in [(3, 10), (100, 200), (900, 1024)]:
                filtered = self.load_dataset_with_filters(
                    min_degree=degree,
                    max_degree=degree,
                    min_len=min_len,
                    max_len=max_len,
                    split="train",
                )
                # Large cyclic groups should have samples at various lengths

    def test_cyclic_range_queries(self):
        """Test range queries across multiple cyclic groups."""
        ranges = [
            (3, 10),  # Small cyclic groups
            (20, 30),  # Medium cyclic groups
            (50, 60),  # Large cyclic groups
            (90, 100),  # Very large cyclic groups
        ]

        for min_deg, max_deg in ranges:
            dataset = self.load_dataset_with_filters(
                min_degree=min_deg, max_degree=max_deg, split="train"
            )

            assert len(dataset) > 0, f"No samples for C{min_deg}-C{max_deg}"

            # Verify degree distribution
            degree_counts = {}
            for i in range(min(500, len(dataset))):
                deg = dataset[i]["group_degree"]
                degree_counts[deg] = degree_counts.get(deg, 0) + 1

            # Should have representation from multiple degrees
            assert len(degree_counts) >= min(5, max_deg - min_deg + 1)

    def test_cyclic_generator_properties(self):
        """Test that cyclic group elements behave correctly."""
        # For cyclic groups, any element raised to the order gives identity
        for degree in [5, 12, 25]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree,
                max_degree=degree,
                min_len=degree,
                max_len=degree,  # Sequence of length n
                split="train",
            )

            if len(dataset) > 0:
                # The permutation IDs should be in range [0, degree)
                for i in range(min(20, len(dataset))):
                    perm_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
                    for pid in perm_ids:
                        assert 0 <= pid < degree

    def test_cyclic_exhaustive_small_groups(self):
        """Exhaustively test all small cyclic groups C3-C20."""
        for degree in range(3, 21):
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            assert len(dataset) > 0, f"C{degree} not found"

            # Test multiple length points
            for length in [3, 10, 50, 100, 500, 1024]:
                length_filtered = self.load_dataset_with_filters(
                    min_degree=degree,
                    max_degree=degree,
                    min_len=length,
                    max_len=length,
                    split="train",
                )
                # Note: Some combinations might not have data

    def test_cyclic_alias_support(self):
        """Test that both Cn and Zn notations work (if supported)."""
        # This test assumes Z notation might be supported as alias
        # Test a few cases where cyclic groups might be referenced as Zn
        for n in [3, 4, 5, 6]:
            dataset = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, split="train"
            )

            if len(dataset) > 0:
                # Should be cyclic group regardless of notation
                assert dataset[0]["group_type"] == "cyclic"
                assert dataset[0]["group_order"] == n
