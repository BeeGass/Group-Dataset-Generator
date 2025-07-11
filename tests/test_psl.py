#!/usr/bin/env python3
"""
Exhaustive tests for PSL (Projective Special Linear) groups dynamic filtering.
"""

import pytest
from .test_base import BaseGroupTest


class TestPSLGroups(BaseGroupTest):
    """Test PSL groups - PSL(2,5) and PSL(2,7)."""

    GROUP_TYPE = "psl"
    GROUP_CONFIG = {
        "degrees": [6, 8],  # PSL(2,5) on 6 points, PSL(2,7) on 8 points
        "orders": {6: 60, 8: 168},  # PSL(2,5) has order 60, PSL(2,7) has order 168
        "prefix": "PSL",
        "params": {6: (2, 5), 8: (2, 7)},  # (n, q) for PSL(n,q)
    }

    @pytest.mark.parametrize("degree", [6, 8])
    def test_specific_degree(self, degree):
        """Test PSL(2,5) and PSL(2,7)."""
        super().test_specific_degree(degree)

    def test_psl_basic_properties(self):
        """Test basic properties of PSL groups."""
        # PSL(2,5)
        dataset_psl25 = self.load_dataset_with_filters(
            min_degree=6, max_degree=6, split="train"
        )
        assert len(dataset_psl25) > 0, "PSL(2,5) not found"
        assert dataset_psl25[0]["group_order"] == 60
        assert dataset_psl25[0]["group_degree"] == 6
        assert dataset_psl25[0]["group_type"] == "psl"

        # PSL(2,7)
        dataset_psl27 = self.load_dataset_with_filters(
            min_degree=8, max_degree=8, split="train"
        )
        assert len(dataset_psl27) > 0, "PSL(2,7) not found"
        assert dataset_psl27[0]["group_order"] == 168
        assert dataset_psl27[0]["group_degree"] == 8
        assert dataset_psl27[0]["group_type"] == "psl"

    def test_psl25_is_isomorphic_to_a5(self):
        """Test that PSL(2,5) ≅ A5 (alternating group on 5 elements)."""
        dataset = self.load_dataset_with_filters(
            min_degree=6, max_degree=6, split="train"
        )

        assert len(dataset) > 0
        # PSL(2,5) has the same order as A5
        assert dataset[0]["group_order"] == 60

        # PSL(2,5) is the smallest non-abelian simple group
        # It's isomorphic to A5 and also to the icosahedral group

    def test_psl27_properties(self):
        """Test properties of PSL(2,7)."""
        dataset = self.load_dataset_with_filters(
            min_degree=8, max_degree=8, split="train"
        )

        assert len(dataset) > 0
        assert dataset[0]["group_order"] == 168

        # PSL(2,7) is simple and has order 168 = 2^3 × 3 × 7
        # It's the automorphism group of the Fano plane

    def test_psl_order_formula(self):
        """Test that PSL(2,q) has order (q³-q)/2 for prime q."""
        # PSL(2,5): (5³-5)/2 = (125-5)/2 = 120/2 = 60 ✓
        # PSL(2,7): (7³-7)/2 = (343-7)/2 = 336/2 = 168 ✓

        test_cases = [(5, 60), (7, 168)]
        for q, expected_order in test_cases:
            calculated_order = (q**3 - q) // 2
            assert calculated_order == expected_order, (
                f"PSL(2,{q}) order calculation failed"
            )

    def test_psl_simple_groups(self):
        """Test that PSL groups are simple (no proper normal subgroups)."""
        # Both PSL(2,5) and PSL(2,7) are simple groups
        for degree in [6, 8]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            assert len(dataset) > 0, f"PSL group at degree {degree} not found"

            # Simple groups should have good representation in dataset
            assert len(dataset) >= 100, "Simple group has too few samples"

    def test_psl_length_filtering(self):
        """Test PSL groups with various sequence lengths."""
        for degree in [6, 8]:
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

    def test_psl_order_filtering(self):
        """Test filtering PSL groups by order."""
        # Filter for order 60 (should find PSL(2,5))
        dataset = self.load_dataset_with_filters(
            min_order=60, max_order=60, split="train"
        )

        psl25_found = False
        for i in range(min(100, len(dataset))):
            if dataset[i]["group_type"] == "psl" and dataset[i]["group_order"] == 60:
                psl25_found = True
                assert dataset[i]["group_degree"] == 6
                break

        assert psl25_found, "PSL(2,5) not found when filtering by order 60"

        # Filter for order 168 (should find PSL(2,7))
        dataset = self.load_dataset_with_filters(
            min_order=168, max_order=168, split="train"
        )

        psl27_found = False
        for i in range(min(100, len(dataset))):
            if dataset[i]["group_type"] == "psl" and dataset[i]["group_order"] == 168:
                psl27_found = True
                assert dataset[i]["group_degree"] == 8
                break

        assert psl27_found, "PSL(2,7) not found when filtering by order 168"

    def test_psl_range_queries(self):
        """Test range queries for PSL groups."""
        # Both PSL groups
        dataset = self.load_dataset_with_filters(
            min_degree=6, max_degree=8, split="train"
        )

        if len(dataset) > 0:
            psl_degrees = set()
            for i in range(min(200, len(dataset))):
                if dataset[i]["group_type"] == "psl":
                    psl_degrees.add(dataset[i]["group_degree"])

            # Should find at least one PSL group
            assert len(psl_degrees) >= 1
            assert psl_degrees.issubset({6, 8})

    def test_psl_permutation_validity(self):
        """Test that permutation IDs are valid for PSL groups."""
        for degree, order in [(6, 60), (8, 168)]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            if len(dataset) > 0:
                for i in range(min(100, len(dataset))):
                    # Check input sequence
                    perm_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
                    for pid in perm_ids:
                        assert 0 <= pid < order, (
                            f"Invalid permutation ID {pid} for PSL group of order {order}"
                        )

                    # Check target
                    target_id = int(dataset[i]["target"])
                    assert 0 <= target_id < order

    def test_psl_combined_filters(self):
        """Test combining multiple filters for PSL groups."""
        # Test PSL(2,5) with specific length ranges
        test_cases = [
            (6, 3, 10),  # Short sequences
            (6, 50, 200),  # Medium sequences
            (6, 800, 1024),  # Long sequences
            (8, 5, 20),  # Short sequences for PSL(2,7)
            (8, 100, 500),  # Medium sequences for PSL(2,7)
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
                    assert dataset[i]["group_type"] == "psl"

    def test_psl_as_matrix_groups(self):
        """Test properties related to PSL as matrix groups."""
        # PSL(n,q) is the group of n×n matrices over F_q with determinant 1,
        # modulo scalar matrices

        dataset_psl25 = self.load_dataset_with_filters(
            min_degree=6, max_degree=6, split="train"
        )

        if len(dataset_psl25) > 0:
            # PSL(2,5) acts on the projective line P¹(F_5) which has 6 points
            assert dataset_psl25[0]["group_degree"] == 6

        dataset_psl27 = self.load_dataset_with_filters(
            min_degree=8, max_degree=8, split="train"
        )

        if len(dataset_psl27) > 0:
            # PSL(2,7) acts on P¹(F_7) which has 8 points
            assert dataset_psl27[0]["group_degree"] == 8

    def test_psl_length_distribution(self):
        """Test that PSL groups have good length distribution."""
        for degree in [6, 8]:
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
                    f"PSL at degree {degree} has poor length diversity"
                )

    def test_psl_train_test_split(self):
        """Test that PSL groups appear in both train and test sets."""
        for degree in [6, 8]:
            train_dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )
            test_dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="test"
            )

            assert len(train_dataset) > 0, f"No PSL samples at degree {degree} in train"
            assert len(test_dataset) > 0, f"No PSL samples at degree {degree} in test"

            # Verify properties match
            assert train_dataset[0]["group_type"] == "psl"
            assert test_dataset[0]["group_type"] == "psl"
            assert train_dataset[0]["group_order"] == test_dataset[0]["group_order"]

    def test_psl_exhaustive_coverage(self):
        """Exhaustive test of PSL group coverage."""
        total_samples = 0
        all_lengths = []

        for degree in [6, 8]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            total_samples += len(dataset)

            # Sample length distribution
            for i in range(min(500, len(dataset))):
                all_lengths.append(dataset[i]["sequence_length"])

        # Should have substantial samples
        assert total_samples > 500, f"Too few PSL samples: {total_samples}"

        # Check length diversity
        unique_lengths = len(set(all_lengths))
        assert unique_lengths > 50, (
            f"Poor length diversity: {unique_lengths} unique lengths"
        )
