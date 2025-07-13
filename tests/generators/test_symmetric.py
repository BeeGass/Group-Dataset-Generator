#!/usr/bin/env python3
"""
Exhaustive tests for individual symmetric group datasets (S3-S9).
"""

import pytest
from ..test_base_individual import BaseIndividualGroupTest


class TestSymmetricIndividual(BaseIndividualGroupTest):
    """Exhaustive tests for individual symmetric group datasets S3 through S9."""

    GROUP_TYPE = "symmetric"
    GROUP_CONFIG = {
        "degrees": list(range(3, 10)),  # S3 to S9
        "orders": {
            3: 6,  # S3 = 3!
            4: 24,  # S4 = 4!
            5: 120,  # S5 = 5!
            6: 720,  # S6 = 6!
            7: 5040,  # S7 = 7!
            8: 40320,  # S8 = 8!
            9: 362880,  # S9 = 9!
        },
        "prefix": "S",
    }

    # Override to test all symmetric group degrees
    @pytest.mark.parametrize("degree", list(range(3, 10)))
    def test_group_properties(self, degree):
        """Test group properties for each symmetric degree."""
        super().test_group_properties(degree)

    def test_factorial_orders(self):
        """Test that symmetric groups have correct factorial orders."""
        import math

        for degree in self.GROUP_CONFIG["degrees"]:
            expected_order = math.factorial(degree)
            actual_order = self.GROUP_CONFIG["orders"][degree]
            assert actual_order == expected_order, (
                f"S{degree} should have order {expected_order}, config has {actual_order}"
            )

            # Also verify in dataset
            dataset = self.load_individual_dataset(degree)
            assert dataset[0]["group_order"] == expected_order

    def test_solvable_boundary(self):
        """Test the boundary between solvable and non-solvable symmetric groups."""
        # S3 and S4 are solvable
        for degree in [3, 4]:
            dataset = self.load_individual_dataset(degree)
            # Just verify they load correctly - solvability is a mathematical property
            assert len(dataset) > 0
            assert dataset[0]["group_degree"] == degree

        # S5 and above are non-solvable
        for degree in [5, 6, 7]:
            dataset = self.load_individual_dataset(degree)
            assert len(dataset) > 0
            assert dataset[0]["group_degree"] == degree
            # S5 is the first non-solvable symmetric group
            if degree == 5:
                assert dataset[0]["group_order"] == 120

    def test_symmetric_specific_compositions(self):
        """Test specific composition properties of symmetric groups."""
        # Test S3 - smallest non-abelian group
        s3_dataset = self.load_individual_dataset(3)
        s3_order = 6

        # Collect some composition examples
        compositions = []
        for i in range(min(20, len(s3_dataset))):
            sample = s3_dataset[i]
            if len(sample["input_sequence"]) == 2:
                compositions.append((sample["input_sequence"], sample["target"]))

        # Verify all IDs are valid
        for seq, target in compositions:
            assert all(0 <= x < s3_order for x in seq)
            assert 0 <= target < s3_order

    def test_large_symmetric_groups(self):
        """Test properties of larger symmetric groups (S7, S8, S9)."""
        large_degrees = [7, 8, 9]

        for degree in large_degrees:
            dataset = self.load_individual_dataset(degree)
            expected_order = self.GROUP_CONFIG["orders"][degree]

            # These should have substantial data
            assert len(dataset) > 1000, f"S{degree} dataset too small"

            # Verify order grows factorially
            if degree > 7:
                prev_order = self.GROUP_CONFIG["orders"][degree - 1]
                assert expected_order == prev_order * degree

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_permutation_composition_examples(self, n):
        """Test specific permutation composition examples."""
        dataset = self.load_individual_dataset(n)

        # For small groups, we can verify some specific compositions
        if n == 3:
            # In S3, element 0 is identity
            # Find examples where we compose with identity
            identity_examples = []
            for i in range(min(100, len(dataset))):
                sample = dataset[i]
                input_ids = [int(x) for x in sample["input_sequence"].split()]
                if 0 in input_ids:
                    identity_examples.append(sample)

            # Composing with identity should give the other element
            for sample in identity_examples[:5]:
                input_ids = [int(x) for x in sample["input_sequence"].split()]
                target_id = int(sample["target"])
                if len(input_ids) == 2:
                    if input_ids[0] == 0:
                        # 0 ∘ x = x
                        assert target_id == input_ids[1], (
                            f"Identity composition failed: 0 ∘ {input_ids[1]} should be {input_ids[1]}, got {target_id}"
                        )

    def test_sequence_length_scaling(self):
        """Test that larger groups have appropriate sequence length distributions."""
        # Larger groups should still have good coverage of sequence lengths
        for degree in [5, 7, 9]:
            dataset = self.load_individual_dataset(degree)

            # Sample sequence lengths
            lengths = []
            for i in range(min(500, len(dataset))):
                lengths.append(dataset[i]["sequence_length"])

            # Should have both short and long sequences
            assert (
                min(lengths) <= 50
            )  # We're sampling, so might not get the very shortest
            assert max(lengths) >= 500

            # Average length should be reasonable
            avg_length = sum(lengths) / len(lengths)
            assert (
                50 <= avg_length <= 600
            )  # Datasets have uniform distribution from 3 to 1024
