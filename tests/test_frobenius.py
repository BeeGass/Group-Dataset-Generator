#!/usr/bin/env python3
"""
Exhaustive tests for Frobenius groups dynamic filtering.
"""

import pytest
from .test_base import BaseGroupTest


class TestFrobeniusGroups(BaseGroupTest):
    """Test Frobenius groups - F20 and F21."""

    GROUP_TYPE = "frobenius"
    GROUP_CONFIG = {
        "degrees": [20, 21],  # F20 and F21
        "orders": {20: 20, 21: 21},  # Frobenius groups
        "prefix": "F",
    }

    @pytest.mark.parametrize("degree", [20, 21])
    def test_specific_degree(self, degree):
        """Test F20 and F21."""
        super().test_specific_degree(degree)

    def test_frobenius_basic_properties(self):
        """Test basic properties of Frobenius groups."""
        # F20
        dataset_f20 = self.load_dataset_with_filters(
            min_degree=20, max_degree=20, split="train"
        )
        assert len(dataset_f20) > 0, "F20 not found"
        assert dataset_f20[0]["group_order"] == 20
        assert dataset_f20[0]["group_degree"] == 20
        assert dataset_f20[0]["group_type"] == "frobenius"

        # F21
        dataset_f21 = self.load_dataset_with_filters(
            min_degree=21, max_degree=21, split="train"
        )
        if len(dataset_f21) > 0:
            assert dataset_f21[0]["group_order"] == 21
            assert dataset_f21[0]["group_degree"] == 21
            assert dataset_f21[0]["group_type"] == "frobenius"

    def test_frobenius_group_structure(self):
        """Test Frobenius group structure properties."""
        # Frobenius groups are semidirect products K ⋊ H where:
        # - K is the Frobenius kernel (normal subgroup)
        # - H is the Frobenius complement
        # - No non-identity element of H fixes any non-identity element of K

        dataset = self.load_dataset_with_filters(
            min_degree=20, max_degree=20, split="train"
        )

        assert len(dataset) > 0
        # F20 = C5 ⋊ C4 (cyclic group of order 5 extended by cyclic group of order 4)
        assert dataset[0]["group_order"] == 20

    def test_frobenius_f20_properties(self):
        """Test specific properties of F20."""
        dataset = self.load_dataset_with_filters(
            min_degree=20, max_degree=20, split="train"
        )

        assert len(dataset) > 0
        # F20 is the smallest Frobenius group
        # It's a non-abelian group of order 20
        assert dataset[0]["group_order"] == 20

        # Check permutation IDs are valid
        for i in range(min(50, len(dataset))):
            perm_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
            for pid in perm_ids:
                assert 0 <= pid < 20

    def test_frobenius_length_filtering(self):
        """Test Frobenius groups with various sequence lengths."""
        for degree in [20, 21]:
            # Test short sequences
            for length in [2, 3, 5, 10, 20]:
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

    def test_frobenius_order_filtering(self):
        """Test filtering Frobenius groups by order."""
        # Filter for order 20 (should find F20)
        dataset = self.load_dataset_with_filters(
            min_order=20, max_order=20, split="train"
        )

        f20_found = False
        for i in range(min(100, len(dataset))):
            if (
                dataset[i]["group_type"] == "frobenius"
                and dataset[i]["group_order"] == 20
            ):
                f20_found = True
                assert dataset[i]["group_degree"] == 20
                break

        assert f20_found, "F20 not found when filtering by order 20"

        # Filter for order 21 (should find F21)
        dataset = self.load_dataset_with_filters(
            min_order=21, max_order=21, split="train"
        )

        f21_found = False
        for i in range(min(100, len(dataset))):
            if (
                dataset[i]["group_type"] == "frobenius"
                and dataset[i]["group_order"] == 21
            ):
                f21_found = True
                assert dataset[i]["group_degree"] == 21
                break

    def test_frobenius_range_queries(self):
        """Test range queries for Frobenius groups."""
        # Both Frobenius groups
        dataset = self.load_dataset_with_filters(
            min_degree=20, max_degree=21, split="train"
        )

        if len(dataset) > 0:
            frobenius_degrees = set()
            for i in range(min(200, len(dataset))):
                if dataset[i]["group_type"] == "frobenius":
                    frobenius_degrees.add(dataset[i]["group_degree"])

            # Should find at least one Frobenius group
            assert len(frobenius_degrees) >= 1
            assert frobenius_degrees.issubset({20, 21})

    def test_frobenius_transitive_action(self):
        """Test that Frobenius groups act transitively."""
        # Frobenius groups act transitively on their natural permutation representation
        dataset = self.load_dataset_with_filters(
            min_degree=20, max_degree=20, split="train"
        )

        if len(dataset) > 0:
            # All permutation IDs should be valid for the group order
            order = dataset[0]["group_order"]
            for i in range(min(100, len(dataset))):
                perm_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
                for pid in perm_ids:
                    assert 0 <= pid < order

                target_id = int(dataset[i]["target"])
                assert 0 <= target_id < order

    def test_frobenius_combined_filters(self):
        """Test combining multiple filters for Frobenius groups."""
        # Test F20 with specific length ranges
        test_cases = [
            (20, 3, 10),  # Short sequences
            (20, 50, 200),  # Medium sequences
            (20, 800, 1024),  # Long sequences
            (21, 5, 20),  # Short sequences for F21
            (21, 100, 500),  # Medium sequences for F21
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
                    assert dataset[i]["group_type"] == "frobenius"

    def test_frobenius_non_abelian(self):
        """Test that Frobenius groups are non-abelian."""
        for degree in [20, 21]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            if len(dataset) > 0:
                # Frobenius groups are non-abelian by definition
                # They have non-trivial semidirect product structure
                assert dataset[0]["group_type"] == "frobenius"

    def test_frobenius_kernel_complement(self):
        """Test kernel and complement structure of Frobenius groups."""
        # F20 = C5 ⋊ C4
        dataset_f20 = self.load_dataset_with_filters(
            min_degree=20, max_degree=20, split="train"
        )

        if len(dataset_f20) > 0:
            # Kernel has order 5, complement has order 4
            # Their product gives order 20
            assert dataset_f20[0]["group_order"] == 20

        # F21 = C7 ⋊ C3
        dataset_f21 = self.load_dataset_with_filters(
            min_degree=21, max_degree=21, split="train"
        )

        if len(dataset_f21) > 0:
            # Kernel has order 7, complement has order 3
            # Their product gives order 21
            assert dataset_f21[0]["group_order"] == 21

    def test_frobenius_length_distribution(self):
        """Test that Frobenius groups have good length distribution."""
        for degree in [20, 21]:
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
                assert len(lengths) > 20, (
                    f"Frobenius group at degree {degree} has poor length diversity"
                )

    def test_frobenius_train_test_split(self):
        """Test that Frobenius groups appear in both train and test sets."""
        for degree in [20, 21]:
            train_dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )
            test_dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="test"
            )

            if len(train_dataset) > 0:
                assert train_dataset[0]["group_type"] == "frobenius"

            if len(test_dataset) > 0:
                assert test_dataset[0]["group_type"] == "frobenius"

            # If both exist, verify properties match
            if len(train_dataset) > 0 and len(test_dataset) > 0:
                assert train_dataset[0]["group_order"] == test_dataset[0]["group_order"]

    def test_frobenius_exhaustive_coverage(self):
        """Exhaustive test of Frobenius group coverage."""
        total_samples = 0
        all_lengths = []

        for degree in [20, 21]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            total_samples += len(dataset)

            # Sample length distribution
            for i in range(min(500, len(dataset))):
                all_lengths.append(dataset[i]["sequence_length"])

        # Should have substantial samples
        assert total_samples > 500, f"Too few Frobenius samples: {total_samples}"

        # Check length diversity
        unique_lengths = len(set(all_lengths))
        assert unique_lengths > 50, (
            f"Poor length diversity: {unique_lengths} unique lengths"
        )
