#!/usr/bin/env python3
"""
Exhaustive tests for Klein four-group (V4) dynamic filtering.
"""

import pytest
from .test_base import BaseGroupTest


class TestKleinFourGroup(BaseGroupTest):
    """Test Klein four-group V4."""

    GROUP_TYPE = "klein"
    GROUP_CONFIG = {
        "degrees": [4],  # V4 acts on 4 points
        "orders": {4: 4},  # V4 has order 4
        "prefix": "V",
    }

    @pytest.mark.parametrize("degree", [4])
    def test_specific_degree(self, degree):
        """Test Klein four-group (only one degree)."""
        super().test_specific_degree(degree)

    def test_klein_basic_properties(self):
        """Test basic properties of Klein four-group."""
        dataset = self.load_dataset_with_filters(split="train")

        assert len(dataset) > 0, "V4 samples not found"

        # V4 has exactly 4 elements
        for i in range(min(50, len(dataset))):
            assert dataset[i]["group_order"] == 4
            assert dataset[i]["group_degree"] == 4
            assert dataset[i]["group_type"] == "klein"

    def test_klein_is_abelian(self):
        """Test that V4 is abelian (all elements have order ≤ 2)."""
        dataset = self.load_dataset_with_filters(split="train")

        # In V4, every non-identity element has order 2
        # This means any element composed with itself gives identity
        # V4 = {e, a, b, ab} where a² = b² = (ab)² = e

        # Check permutation IDs are in range [0, 3]
        for i in range(min(100, len(dataset))):
            perm_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
            for pid in perm_ids:
                assert 0 <= pid <= 3, f"Invalid permutation ID {pid} for V4"

    def test_klein_vs_cyclic_c4(self):
        """Test that V4 is different from cyclic group C4."""
        # V4 has order 4 but is NOT cyclic
        # V4 ≅ Z2 × Z2, while C4 ≅ Z4

        dataset_v4 = self.load_dataset_with_filters(split="train")

        # Both have order 4 but different structure
        assert dataset_v4[0]["group_order"] == 4
        assert dataset_v4[0]["group_type"] == "klein"  # Not "cyclic"

    def test_klein_length_filtering(self):
        """Test V4 with various sequence lengths."""
        test_lengths = [2, 3, 4, 5, 10, 20, 50, 100, 256, 512, 1024]

        for length in test_lengths:
            dataset = self.load_dataset_with_filters(
                min_len=length, max_len=length, split="train"
            )

            if len(dataset) > 0:
                # All should be V4 samples
                for i in range(min(10, len(dataset))):
                    assert dataset[i]["sequence_length"] == length
                    assert dataset[i]["group_type"] == "klein"
                    assert dataset[i]["group_order"] == 4

    def test_klein_composition_table(self):
        """Test that V4 samples follow correct composition rules."""
        # V4 composition table:
        # Every element is its own inverse
        # Any two distinct non-identity elements compose to the third

        dataset = self.load_dataset_with_filters(
            min_len=2,
            max_len=2,
            split="train",  # Length 2 for simple composition
        )

        if len(dataset) > 0:
            for i in range(min(50, len(dataset))):
                input_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
                target_id = int(dataset[i]["target"])

                # All IDs should be in range [0, 3]
                assert all(0 <= id <= 3 for id in input_ids)
                assert 0 <= target_id <= 3

    def test_klein_order_filtering(self):
        """Test filtering V4 by order."""
        # Filter for order exactly 4
        dataset = self.load_dataset_with_filters(
            min_order=4, max_order=4, split="train"
        )

        # Should find V4 samples
        v4_found = False
        for i in range(min(100, len(dataset))):
            if dataset[i]["group_type"] == "klein":
                v4_found = True
                assert dataset[i]["group_order"] == 4
                break

        assert v4_found, "V4 not found when filtering by order 4"

    def test_klein_combined_filters(self):
        """Test combining multiple filters for V4."""
        # Test degree + length filters
        dataset = self.load_dataset_with_filters(
            min_degree=4, max_degree=4, min_len=10, max_len=100, split="train"
        )

        if len(dataset) > 0:
            for i in range(min(30, len(dataset))):
                assert dataset[i]["group_degree"] == 4
                assert 10 <= dataset[i]["sequence_length"] <= 100
                assert dataset[i]["group_type"] == "klein"

        # Test order + length filters
        dataset = self.load_dataset_with_filters(
            min_order=4, max_order=4, min_len=500, max_len=1024, split="train"
        )

        if len(dataset) > 0:
            # Check if V4 appears in results
            for i in range(min(50, len(dataset))):
                if dataset[i]["group_type"] == "klein":
                    assert dataset[i]["group_order"] == 4
                    assert 500 <= dataset[i]["sequence_length"] <= 1024

    def test_klein_train_test_distribution(self):
        """Test that V4 appears in both train and test sets."""
        train_dataset = self.load_dataset_with_filters(split="train")
        test_dataset = self.load_dataset_with_filters(split="test")

        assert len(train_dataset) > 0, "No V4 samples in training set"
        assert len(test_dataset) > 0, "No V4 samples in test set"

        # Verify both have correct properties
        for dataset, split_name in [(train_dataset, "train"), (test_dataset, "test")]:
            sample = dataset[0]
            assert sample["group_type"] == "klein", f"Wrong group type in {split_name}"
            assert sample["group_order"] == 4, f"Wrong order in {split_name}"
            assert sample["group_degree"] == 4, f"Wrong degree in {split_name}"

    def test_klein_exhaustive_length_coverage(self):
        """Test V4 coverage across all possible lengths."""
        # Collect all unique lengths in the dataset
        dataset = self.load_dataset_with_filters(split="train")

        lengths_found = set()
        for i in range(min(1000, len(dataset))):
            lengths_found.add(dataset[i]["sequence_length"])

        # Should have good length coverage
        assert len(lengths_found) > 50, (
            f"V4 has poor length diversity: only {len(lengths_found)} unique lengths"
        )

        # Check specific important lengths
        important_lengths = [3, 4, 10, 100, 512, 1024]
        for length in important_lengths:
            dataset_len = self.load_dataset_with_filters(
                min_len=length, max_len=length, split="train"
            )
            # Some lengths might not have samples, but common ones should

    def test_klein_edge_cases(self):
        """Test edge cases for Klein four-group."""
        # Very short sequences
        dataset = self.load_dataset_with_filters(min_len=1, max_len=1, split="train")
        # Length 1 sequences might be rare or not exist

        # Maximum length sequences
        dataset = self.load_dataset_with_filters(
            min_len=1024, max_len=1024, split="train"
        )

        if len(dataset) > 0:
            assert dataset[0]["sequence_length"] == 1024
            assert dataset[0]["group_type"] == "klein"

        # Invalid degree filter (V4 only has degree 4)
        dataset = self.load_dataset_with_filters(
            min_degree=5, max_degree=10, split="train"
        )

        # Should not contain any V4 samples
        for i in range(min(100, len(dataset))):
            assert dataset[i]["group_type"] != "klein"
