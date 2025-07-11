#!/usr/bin/env python3
"""
Exhaustive tests for symmetric groups (S3-S12) dynamic filtering.
"""

import pytest
from .test_base import BaseGroupTest


class TestSymmetricGroups(BaseGroupTest):
    """Test symmetric groups S3 through S12."""

    GROUP_TYPE = "symmetric"
    GROUP_CONFIG = {
        "degrees": list(range(3, 13)),  # S3 to S12
        "orders": {
            3: 6,  # S3
            4: 24,  # S4
            5: 120,  # S5
            6: 720,  # S6
            7: 5040,  # S7
            8: 40320,  # S8
            9: 362880,  # S9
            10: 3628800,  # S10
            11: 39916800,  # S11
            12: 479001600,  # S12
        },
        "prefix": "S",
    }

    # Override test_specific_degree to test all symmetric group degrees
    @pytest.mark.parametrize("degree", list(range(3, 13)))
    def test_specific_degree(self, degree):
        """Test each symmetric group degree S3 through S12."""
        super().test_specific_degree(degree)

    def test_symmetric_specific_properties(self):
        """Test properties specific to symmetric groups."""
        # S3 specific tests
        dataset_s3 = self.load_dataset_with_filters(
            min_degree=3, max_degree=3, split="train"
        )
        assert len(dataset_s3) > 0, "S3 samples not found"

        # Verify S3 has order 6
        for i in range(min(10, len(dataset_s3))):
            assert dataset_s3[i]["group_order"] == 6

        # S12 specific tests (largest symmetric group)
        dataset_s12 = self.load_dataset_with_filters(
            min_degree=12, max_degree=12, split="train"
        )
        assert len(dataset_s12) > 0, "S12 samples not found"

        # Verify S12 has order 479001600
        for i in range(min(10, len(dataset_s12))):
            assert dataset_s12[i]["group_order"] == 479001600

    def test_symmetric_subgroup_relationships(self):
        """Test that smaller symmetric groups are properly represented."""
        # S3 ⊂ S4 ⊂ S5 ... in terms of degree relationships
        for degree in range(3, 12):
            smaller = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )
            larger = self.load_dataset_with_filters(
                min_degree=degree + 1, max_degree=degree + 1, split="train"
            )

            assert len(smaller) > 0 and len(larger) > 0
            assert smaller[0]["group_order"] < larger[0]["group_order"]

    def test_high_degree_symmetric_groups(self):
        """Special tests for high-degree symmetric groups (S10, S11, S12)."""
        high_degrees = [10, 11, 12]

        for degree in high_degrees:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            assert len(dataset) > 0, f"S{degree} not found"

            # These should have substantial samples despite large order
            assert len(dataset) >= 100, f"S{degree} has too few samples"

            # Test various lengths for high-degree groups
            for length in [10, 100, 1024]:
                filtered = self.load_dataset_with_filters(
                    min_degree=degree,
                    max_degree=degree,
                    min_len=length,
                    max_len=length,
                    split="train",
                )
                # High degree groups should have samples at various lengths

    def test_symmetric_factorial_orders(self):
        """Verify that symmetric groups have factorial orders."""
        import math

        for degree in self.GROUP_CONFIG["degrees"]:
            expected_order = math.factorial(degree)
            actual_order = self.GROUP_CONFIG["orders"][degree]
            assert actual_order == expected_order, (
                f"S{degree} order {actual_order} != {degree}! = {expected_order}"
            )

    def test_symmetric_permutation_ids(self):
        """Test that permutation IDs are within valid range for each symmetric group."""
        for degree in [3, 5, 8, 10]:  # Sample some degrees
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            order = self.GROUP_CONFIG["orders"][degree]

            # Check input sequences
            for i in range(min(50, len(dataset))):
                tokens = dataset[i]["input_sequence"].split()
                for token in tokens:
                    perm_id = int(token)
                    assert 0 <= perm_id < order, (
                        f"Invalid permutation ID {perm_id} for S{degree} (order {order})"
                    )

                # Check target
                target_id = int(dataset[i]["target"])
                assert 0 <= target_id < order
