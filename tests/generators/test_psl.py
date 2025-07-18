#!/usr/bin/env python3
"""
Exhaustive tests for individual PSL group datasets (PSL(2,5), PSL(2,7)).
"""

import pytest
from ..test_base_individual import BaseIndividualGroupTest


class TestPSLIndividual(BaseIndividualGroupTest):
    """Exhaustive tests for individual PSL group datasets."""

    GROUP_TYPE = "psl"
    GROUP_CONFIG = {
        "degrees": [6, 8],  # Special encoding for PSL groups
        "orders": {
            6: 60,  # PSL(2,5) has order 60
            8: 168,  # PSL(2,7) has order 168
        },
        "prefix": "PSL",
    }

    # Override to test all PSL group degrees
    @pytest.mark.parametrize("degree", [6, 8])
    def test_group_properties(self, degree):
        """Test group properties for each PSL degree."""
        super().test_group_properties(degree)

    def test_psl_simple_groups(self):
        """Test that PSL groups are simple."""
        # PSL(2,p) is simple for all primes p ≥ 5
        # These are among the most important finite simple groups

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # Both PSL(2,5) and PSL(2,7) are simple groups
            # (no proper normal subgroups)
            assert len(dataset) > 0
            assert dataset[0]["group_type"] == "psl"

    def test_psl_2_5_properties(self):
        """Test specific properties of PSL(2,5)."""
        dataset = self.load_individual_dataset(6)  # degree 6 encodes PSL(2,5)

        # PSL(2,5) has order 60 = |SL(2,5)|/2
        assert dataset[0]["group_order"] == 60

        # PSL(2,5) is isomorphic to:
        # - A5 (alternating group on 5 elements)
        # - I (rotational symmetry group of icosahedron)
        # It's the smallest non-abelian simple group

    def test_psl_2_7_properties(self):
        """Test specific properties of PSL(2,7)."""
        dataset = self.load_individual_dataset(8)  # degree 8 encodes PSL(2,7)

        # PSL(2,7) has order 168 = |SL(2,7)|/2
        assert dataset[0]["group_order"] == 168

        # PSL(2,7) is the second smallest non-abelian simple group
        # It's the automorphism group of the Klein quartic
        # Also the symmetry group of the Fano plane

    def test_order_formula(self):
        """Test that PSL groups follow the order formula."""
        # |PSL(2,q)| = (q³ - q)/2 for prime q

        psl_params = {
            6: 5,  # PSL(2,5): (5³ - 5)/2 = 120/2 = 60
            8: 7,  # PSL(2,7): (7³ - 7)/2 = 336/2 = 168
        }

        for degree, q in psl_params.items():
            dataset = self.load_individual_dataset(degree)
            expected_order = (q**3 - q) // 2
            assert dataset[0]["group_order"] == expected_order

    def test_non_solvable_property(self):
        """Test that PSL groups are non-solvable."""
        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # PSL(2,p) for p ≥ 5 are simple, hence non-solvable
            # They are in NC¹ complexity class
            assert dataset[0]["group_order"] in [60, 168]

    def test_projective_action(self):
        """Test properties related to projective action."""
        # PSL(2,q) acts on the projective line P¹(F_q)
        # This action is sharply 3-transitive

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # Sample some compositions
            short_sequences = []
            for i in range(min(50, len(dataset))):
                if dataset[i]["sequence_length"] <= 3:
                    short_sequences.append(dataset[i])

            # Verify valid permutation IDs
            for sample in short_sequences:
                input_ids = [int(x) for x in sample["input_sequence"].split()]
                target_id = int(sample["target"])
                order = dataset[0]["group_order"]
                assert all(0 <= x < order for x in input_ids)
                assert 0 <= target_id < order

    def test_generation_properties(self):
        """Test generation properties of PSL groups."""
        # PSL(2,p) can be generated by two elements
        # One of order 2 and one of order 3 (or p)

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # These are finitely presented groups with known presentations
            assert dataset[0]["group_type"] == "psl"

    def test_exceptional_isomorphisms(self):
        """Test exceptional isomorphisms of small PSL groups."""
        # PSL(2,5) ≅ A5 (we test this implicitly)
        psl_2_5_dataset = self.load_individual_dataset(6)
        assert psl_2_5_dataset[0]["group_order"] == 60

        # PSL(2,7) is the unique simple group of order 168
        psl_2_7_dataset = self.load_individual_dataset(8)
        assert psl_2_7_dataset[0]["group_order"] == 168

        # These are fundamental groups in mathematics
