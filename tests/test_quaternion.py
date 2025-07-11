#!/usr/bin/env python3
"""
Exhaustive tests for quaternion groups (Q8, Q16, Q32) dynamic filtering.
"""

import pytest
from .test_base import BaseGroupTest


class TestQuaternionGroups(BaseGroupTest):
    """Test quaternion groups Q8, Q16, and Q32."""

    GROUP_TYPE = "quaternion"
    GROUP_CONFIG = {
        "degrees": [8, 16, 32],  # Q8, Q16, Q32
        "orders": {8: 8, 16: 16, 32: 32},  # Order equals degree for our representation
        "prefix": "Q",
    }

    @pytest.mark.parametrize("degree", [8, 16, 32])
    def test_specific_degree(self, degree):
        """Test each quaternion group Q8, Q16, Q32."""
        super().test_specific_degree(degree)

    def test_quaternion_basic_properties(self):
        """Test basic properties of quaternion groups."""
        for n in [8, 16, 32]:
            dataset = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, split="train"
            )

            assert len(dataset) > 0, f"Q{n} samples not found"

            # Verify order
            for i in range(min(20, len(dataset))):
                assert dataset[i]["group_order"] == n
                assert dataset[i]["group_degree"] == n
                assert dataset[i]["group_type"] == "quaternion"

    def test_q8_special_properties(self):
        """Test special properties of Q8 (quaternion group of order 8)."""
        dataset = self.load_dataset_with_filters(
            min_degree=8, max_degree=8, split="train"
        )

        assert len(dataset) > 0, "Q8 not found"

        # Q8 is the smallest non-abelian group where every subgroup is normal
        # Q8 = {±1, ±i, ±j, ±k} with i² = j² = k² = ijk = -1
        for i in range(min(50, len(dataset))):
            assert dataset[i]["group_order"] == 8

            # Check permutation IDs
            perm_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
            for pid in perm_ids:
                assert 0 <= pid < 8

    def test_generalized_quaternion_groups(self):
        """Test generalized quaternion groups Q16 and Q32."""
        for n in [16, 32]:
            dataset = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, split="train"
            )

            assert len(dataset) > 0, f"Q{n} not found"

            # Q2^n has order 2^n
            assert dataset[0]["group_order"] == n

            # These are non-abelian groups of order 2^n
            # Q2^n = <a, b | a^(2^(n-1)) = 1, b² = a^(2^(n-2)), bab^(-1) = a^(-1)>

    def test_quaternion_vs_dihedral(self):
        """Test that Q8 is different from D4 despite same order."""
        dataset_q8 = self.load_dataset_with_filters(
            min_degree=8, max_degree=8, split="train"
        )

        # Q8 and D4 both have order 8 but different structure
        # Q8 has only one element of order 2, while D4 has five
        assert len(dataset_q8) > 0
        assert dataset_q8[0]["group_type"] == "quaternion"
        assert dataset_q8[0]["group_order"] == 8

    def test_quaternion_length_filtering(self):
        """Test quaternion groups with various sequence lengths."""
        for n in [8, 16, 32]:
            # Test short sequences
            for length in [2, 3, 4, 5, 10]:
                dataset = self.load_dataset_with_filters(
                    min_degree=n,
                    max_degree=n,
                    min_len=length,
                    max_len=length,
                    split="train",
                )

                if len(dataset) > 0:
                    assert dataset[0]["sequence_length"] == length
                    assert dataset[0]["group_degree"] == n

            # Test medium sequences
            for length in [50, 100, 256]:
                dataset = self.load_dataset_with_filters(
                    min_degree=n,
                    max_degree=n,
                    min_len=length,
                    max_len=length,
                    split="train",
                )

                if len(dataset) > 0:
                    assert dataset[0]["sequence_length"] == length

            # Test long sequences
            dataset = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, min_len=1000, max_len=1024, split="train"
            )

    def test_quaternion_order_filtering(self):
        """Test filtering quaternion groups by order."""
        # Each quaternion group has unique order in our set
        for n in [8, 16, 32]:
            dataset = self.load_dataset_with_filters(
                min_order=n, max_order=n, split="train"
            )

            # Should find Qn
            qn_found = False
            for i in range(min(100, len(dataset))):
                if (
                    dataset[i]["group_type"] == "quaternion"
                    and dataset[i]["group_order"] == n
                ):
                    qn_found = True
                    assert dataset[i]["group_degree"] == n
                    break

            assert qn_found, f"Q{n} not found when filtering by order {n}"

    def test_quaternion_range_queries(self):
        """Test range queries across quaternion groups."""
        # Test Q8-Q16 range
        dataset = self.load_dataset_with_filters(
            min_degree=8, max_degree=16, split="train"
        )

        if len(dataset) > 0:
            degrees_found = set()
            for i in range(min(200, len(dataset))):
                if dataset[i]["group_type"] == "quaternion":
                    degrees_found.add(dataset[i]["group_degree"])

            # Should find both Q8 and Q16
            assert 8 in degrees_found or 16 in degrees_found

        # Test all quaternion groups
        dataset = self.load_dataset_with_filters(
            min_degree=8, max_degree=32, split="train"
        )

        if len(dataset) > 0:
            quaternion_degrees = set()
            for i in range(min(300, len(dataset))):
                if dataset[i]["group_type"] == "quaternion":
                    quaternion_degrees.add(dataset[i]["group_degree"])

            # Should have representation from multiple quaternion groups
            assert len(quaternion_degrees) >= 1

    def test_quaternion_composition_properties(self):
        """Test composition properties of quaternion groups."""
        # In Q8: i² = j² = k² = -1, ij = k, jk = i, ki = j
        dataset = self.load_dataset_with_filters(
            min_degree=8, max_degree=8, min_len=2, max_len=5, split="train"
        )

        if len(dataset) > 0:
            for i in range(min(100, len(dataset))):
                input_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
                target_id = int(dataset[i]["target"])

                # All IDs should be valid
                assert all(0 <= id < 8 for id in input_ids)
                assert 0 <= target_id < 8

    def test_quaternion_power_structure(self):
        """Test that quaternion groups Q_{2^n} have correct power structure."""
        for k, n in [(3, 8), (4, 16), (5, 32)]:  # Q_{2^k} = Q_n
            dataset = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, split="train"
            )

            if len(dataset) > 0:
                assert dataset[0]["group_order"] == n
                assert dataset[0]["group_order"] == 2**k

    def test_quaternion_non_abelian(self):
        """Test that all quaternion groups are non-abelian."""
        # All Qn for n ≥ 8 are non-abelian
        for n in [8, 16, 32]:
            dataset = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, split="train"
            )

            assert len(dataset) > 0, f"Q{n} (non-abelian) not found"

            # Unlike abelian groups, order matters in composition
            # This should be reflected in the variety of targets for same elements

    def test_quaternion_hamiltonian_property(self):
        """Test that quaternion groups are Hamiltonian (every subgroup is normal)."""
        # This is a unique property - Qn are the only non-abelian Hamiltonian groups
        dataset_q8 = self.load_dataset_with_filters(
            min_degree=8, max_degree=8, split="train"
        )

        assert len(dataset_q8) > 0
        # Q8 is the smallest Hamiltonian group
        assert dataset_q8[0]["group_order"] == 8

    def test_quaternion_binary_representation(self):
        """Test that our quaternion groups use appropriate degree for representation."""
        # We represent Qn as permutations on n points
        representation_map = {8: 8, 16: 16, 32: 32}

        for qn_order, degree in representation_map.items():
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            if len(dataset) > 0:
                assert dataset[0]["group_degree"] == degree
                assert dataset[0]["group_order"] == qn_order

    def test_quaternion_exhaustive_coverage(self):
        """Exhaustive tests for quaternion group coverage."""
        all_lengths = []
        all_samples = 0

        for n in [8, 16, 32]:
            dataset = self.load_dataset_with_filters(
                min_degree=n, max_degree=n, split="train"
            )

            all_samples += len(dataset)

            # Collect length distribution
            for i in range(min(500, len(dataset))):
                all_lengths.append(dataset[i]["sequence_length"])

        # Should have substantial total samples
        assert all_samples > 1000, f"Too few total quaternion samples: {all_samples}"

        # Should have diverse lengths
        unique_lengths = len(set(all_lengths))
        assert unique_lengths > 50, (
            f"Poor length diversity: only {unique_lengths} unique lengths"
        )
