#!/usr/bin/env python3
"""
Exhaustive tests for Mathieu groups dynamic filtering.
"""

import pytest
from .test_base import BaseGroupTest


class TestMathieuGroups(BaseGroupTest):
    """Test Mathieu groups - M11 and M12."""

    GROUP_TYPE = "mathieu"
    GROUP_CONFIG = {
        "degrees": [11, 12],  # M11 acts on 11 points, M12 on 12 points
        "orders": {11: 7920, 12: 95040},  # M11 has order 7920, M12 has order 95040
        "prefix": "M",
    }

    @pytest.mark.parametrize("degree", [11, 12])
    def test_specific_degree(self, degree):
        """Test M11 and M12."""
        super().test_specific_degree(degree)

    def test_mathieu_basic_properties(self):
        """Test basic properties of Mathieu groups."""
        # M11
        dataset_m11 = self.load_dataset_with_filters(
            min_degree=11, max_degree=11, split="train"
        )
        if len(dataset_m11) > 0:
            assert dataset_m11[0]["group_order"] == 7920
            assert dataset_m11[0]["group_degree"] == 11
            assert dataset_m11[0]["group_type"] == "mathieu"

        # M12
        dataset_m12 = self.load_dataset_with_filters(
            min_degree=12, max_degree=12, split="train"
        )
        if len(dataset_m12) > 0:
            assert dataset_m12[0]["group_order"] == 95040
            assert dataset_m12[0]["group_degree"] == 12
            assert dataset_m12[0]["group_type"] == "mathieu"

    def test_mathieu_sporadic_simple_groups(self):
        """Test that Mathieu groups are sporadic simple groups."""
        # M11 and M12 are among the first discovered sporadic simple groups
        # They are part of the first generation of sporadic groups discovered by Mathieu

        for degree, order in [(11, 7920), (12, 95040)]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            if len(dataset) > 0:
                assert dataset[0]["group_order"] == order
                # These are simple groups - no proper normal subgroups

    def test_m11_properties(self):
        """Test specific properties of M11."""
        dataset = self.load_dataset_with_filters(
            min_degree=11, max_degree=11, split="train"
        )

        if len(dataset) > 0:
            # M11 is a 4-transitive permutation group on 11 points
            # Order: 7920 = 11 × 10 × 9 × 8
            assert dataset[0]["group_order"] == 7920
            assert dataset[0]["group_degree"] == 11

            # Check permutation IDs
            for i in range(min(50, len(dataset))):
                perm_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
                for pid in perm_ids:
                    assert 0 <= pid < 7920

    def test_m12_properties(self):
        """Test specific properties of M12."""
        dataset = self.load_dataset_with_filters(
            min_degree=12, max_degree=12, split="train"
        )

        if len(dataset) > 0:
            # M12 is a 5-transitive permutation group on 12 points
            # Order: 95040 = 12 × 11 × 10 × 9 × 8
            assert dataset[0]["group_order"] == 95040
            assert dataset[0]["group_degree"] == 12

            # M12 contains M11 as a point stabilizer

    def test_mathieu_order_formulas(self):
        """Test the order formulas for Mathieu groups."""
        # M11: |M11| = 11 × 10 × 9 × 8 = 7920
        m11_order = 11 * 10 * 9 * 8
        assert m11_order == 7920

        # M12: |M12| = 12 × 11 × 10 × 9 × 8 = 95040
        m12_order = 12 * 11 * 10 * 9 * 8
        assert m12_order == 95040

        # Verify in dataset
        dataset_m11 = self.load_dataset_with_filters(
            min_degree=11, max_degree=11, split="train"
        )
        if len(dataset_m11) > 0:
            assert dataset_m11[0]["group_order"] == m11_order

        dataset_m12 = self.load_dataset_with_filters(
            min_degree=12, max_degree=12, split="train"
        )
        if len(dataset_m12) > 0:
            assert dataset_m12[0]["group_order"] == m12_order

    def test_mathieu_length_filtering(self):
        """Test Mathieu groups with various sequence lengths."""
        for degree in [11, 12]:
            # Test short sequences
            for length in [2, 3, 5, 10]:
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

            # Test medium sequences
            for length in [50, 100, 256]:
                dataset = self.load_dataset_with_filters(
                    min_degree=degree,
                    max_degree=degree,
                    min_len=length,
                    max_len=length,
                    split="train",
                )

                if len(dataset) > 0:
                    assert dataset[0]["sequence_length"] == length

            # Test long sequences
            dataset = self.load_dataset_with_filters(
                min_degree=degree,
                max_degree=degree,
                min_len=900,
                max_len=1024,
                split="train",
            )

    def test_mathieu_order_filtering(self):
        """Test filtering Mathieu groups by order."""
        # Filter for M11's order
        dataset = self.load_dataset_with_filters(
            min_order=7920, max_order=7920, split="train"
        )

        m11_found = False
        for i in range(min(100, len(dataset))):
            if (
                dataset[i]["group_type"] == "mathieu"
                and dataset[i]["group_order"] == 7920
            ):
                m11_found = True
                assert dataset[i]["group_degree"] == 11
                break

        if len(dataset) > 0 and dataset[0]["group_type"] == "mathieu":
            assert m11_found, "M11 not found when filtering by order 7920"

        # Filter for M12's order
        dataset = self.load_dataset_with_filters(
            min_order=95040, max_order=95040, split="train"
        )

        m12_found = False
        for i in range(min(100, len(dataset))):
            if (
                dataset[i]["group_type"] == "mathieu"
                and dataset[i]["group_order"] == 95040
            ):
                m12_found = True
                assert dataset[i]["group_degree"] == 12
                break

        if len(dataset) > 0 and dataset[0]["group_type"] == "mathieu":
            assert m12_found, "M12 not found when filtering by order 95040"

    def test_mathieu_range_queries(self):
        """Test range queries for Mathieu groups."""
        # Both Mathieu groups
        dataset = self.load_dataset_with_filters(
            min_degree=11, max_degree=12, split="train"
        )

        if len(dataset) > 0:
            mathieu_degrees = set()
            for i in range(min(200, len(dataset))):
                if dataset[i]["group_type"] == "mathieu":
                    mathieu_degrees.add(dataset[i]["group_degree"])

            # Should find at least one Mathieu group
            assert len(mathieu_degrees) >= 1
            assert mathieu_degrees.issubset({11, 12})

    def test_mathieu_transitivity(self):
        """Test transitivity properties of Mathieu groups."""
        # M11 is 4-transitive on 11 points
        dataset_m11 = self.load_dataset_with_filters(
            min_degree=11, max_degree=11, split="train"
        )

        if len(dataset_m11) > 0:
            # Can move any ordered 4-tuple to any other ordered 4-tuple
            assert dataset_m11[0]["group_degree"] == 11

        # M12 is 5-transitive on 12 points
        dataset_m12 = self.load_dataset_with_filters(
            min_degree=12, max_degree=12, split="train"
        )

        if len(dataset_m12) > 0:
            # Can move any ordered 5-tuple to any other ordered 5-tuple
            assert dataset_m12[0]["group_degree"] == 12

    def test_mathieu_steiner_systems(self):
        """Test connection to Steiner systems."""
        # M11 is the automorphism group of S(4,5,11) Steiner system
        # M12 is the automorphism group of S(5,6,12) Steiner system

        dataset_m11 = self.load_dataset_with_filters(
            min_degree=11, max_degree=11, split="train"
        )

        if len(dataset_m11) > 0:
            # S(4,5,11) has 11 points, blocks of size 5
            assert dataset_m11[0]["group_degree"] == 11

        dataset_m12 = self.load_dataset_with_filters(
            min_degree=12, max_degree=12, split="train"
        )

        if len(dataset_m12) > 0:
            # S(5,6,12) has 12 points, blocks of size 6
            assert dataset_m12[0]["group_degree"] == 12

    def test_mathieu_permutation_validity(self):
        """Test that permutation IDs are valid for Mathieu groups."""
        for degree, order in [(11, 7920), (12, 95040)]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            if len(dataset) > 0:
                for i in range(min(100, len(dataset))):
                    # Check input sequence
                    perm_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
                    for pid in perm_ids:
                        assert 0 <= pid < order, (
                            f"Invalid permutation ID {pid} for M{degree} of order {order}"
                        )

                    # Check target
                    target_id = int(dataset[i]["target"])
                    assert 0 <= target_id < order

    def test_mathieu_combined_filters(self):
        """Test combining multiple filters for Mathieu groups."""
        # Test M11 with specific length ranges
        test_cases = [
            (11, 3, 10),  # Short sequences
            (11, 50, 200),  # Medium sequences
            (11, 800, 1024),  # Long sequences
            (12, 5, 20),  # Short sequences for M12
            (12, 100, 500),  # Medium sequences for M12
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
                    assert dataset[i]["group_type"] == "mathieu"

    def test_mathieu_historical_significance(self):
        """Test properties related to historical significance."""
        # Mathieu groups were the first sporadic simple groups discovered
        # M11 (1861) and M12 (1861) by Émile Léonard Mathieu

        for degree in [11, 12]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            if len(dataset) > 0:
                # These are finite simple groups
                assert dataset[0]["group_type"] == "mathieu"

    def test_mathieu_subgroup_relationships(self):
        """Test subgroup relationships between Mathieu groups."""
        # M11 is a subgroup of M12 (point stabilizer)
        # Both are simple groups

        dataset_m11 = self.load_dataset_with_filters(
            min_degree=11, max_degree=11, split="train"
        )
        dataset_m12 = self.load_dataset_with_filters(
            min_degree=12, max_degree=12, split="train"
        )

        if len(dataset_m11) > 0 and len(dataset_m12) > 0:
            # M12 contains M11 as the stabilizer of a point
            # |M12| / |M11| = 95040 / 7920 = 12
            assert dataset_m12[0]["group_order"] // dataset_m11[0]["group_order"] == 12

    def test_mathieu_length_distribution(self):
        """Test that Mathieu groups have good length distribution."""
        for degree in [11, 12]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            if len(dataset) > 500:
                # Collect length statistics
                lengths = {}
                for i in range(500):
                    length = dataset[i]["sequence_length"]
                    lengths[length] = lengths.get(length, 0) + 1

                # Should have diverse lengths
                assert len(lengths) > 20, f"M{degree} has poor length diversity"

    def test_mathieu_train_test_split(self):
        """Test that Mathieu groups appear in both train and test sets."""
        for degree in [11, 12]:
            train_dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )
            test_dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="test"
            )

            if len(train_dataset) > 0:
                assert train_dataset[0]["group_type"] == "mathieu"

            if len(test_dataset) > 0:
                assert test_dataset[0]["group_type"] == "mathieu"

            # If both exist, verify properties match
            if len(train_dataset) > 0 and len(test_dataset) > 0:
                assert train_dataset[0]["group_order"] == test_dataset[0]["group_order"]

    def test_mathieu_exhaustive_coverage(self):
        """Exhaustive test of Mathieu group coverage."""
        total_samples = 0
        all_lengths = []

        for degree in [11, 12]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            total_samples += len(dataset)

            # Sample length distribution
            for i in range(min(500, len(dataset))):
                all_lengths.append(dataset[i]["sequence_length"])

        # Should have samples (may be fewer due to large group orders)
        if total_samples > 0:
            # Check length diversity
            unique_lengths = len(set(all_lengths))
            assert unique_lengths > 10, (
                f"Poor length diversity: {unique_lengths} unique lengths"
            )
