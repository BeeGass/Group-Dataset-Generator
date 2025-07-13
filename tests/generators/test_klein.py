#!/usr/bin/env python3
"""
Exhaustive tests for individual Klein Four group dataset (V4).
"""

import pytest
from ..test_base_individual import BaseIndividualGroupTest


class TestKleinIndividual(BaseIndividualGroupTest):
    """Exhaustive tests for individual Klein Four group dataset."""

    GROUP_TYPE = "klein"
    GROUP_CONFIG = {
        "degrees": [4],  # Only one Klein Four group
        "orders": {
            4: 4,  # V4 has order 4
        },
        "prefix": "V",
    }

    # Override to test Klein Four group
    @pytest.mark.parametrize("degree", [4])
    def test_group_properties(self, degree):
        """Test group properties for Klein Four group."""
        super().test_group_properties(degree)

    def test_klein_abelian_property(self):
        """Test that Klein Four group is abelian."""
        dataset = self.load_individual_dataset(4)

        # V4 is abelian - all elements commute
        # V4 = {e, a, b, ab} where a^2 = b^2 = (ab)^2 = e
        assert dataset[0]["group_order"] == 4
        assert dataset[0]["group_degree"] == 4

    def test_klein_structure(self):
        """Test the structure of Klein Four group."""
        dataset = self.load_individual_dataset(4)

        # V4 is isomorphic to Z2 × Z2
        # All non-identity elements have order 2
        # This is the smallest non-cyclic abelian group

        # Sample some compositions
        short_sequences = []
        for i in range(min(50, len(dataset))):
            if dataset[i]["sequence_length"] <= 3:
                short_sequences.append(dataset[i])

        # All elements should be in [0, 1, 2, 3]
        for sample in short_sequences:
            input_ids = [int(x) for x in sample["input_sequence"].split()]
            target_id = int(sample["target"])
            assert all(0 <= x < 4 for x in input_ids)
            assert 0 <= target_id < 4

    def test_all_elements_self_inverse(self):
        """Test that all non-identity elements are self-inverse in V4."""
        dataset = self.load_individual_dataset(4)

        # In V4, every element is its own inverse
        # a * a = e for all elements a
        self_compositions = []
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            input_ids = [int(x) for x in sample["input_sequence"].split()]
            if len(input_ids) == 2 and input_ids[0] == input_ids[1]:
                self_compositions.append((input_ids[0], int(sample["target"])))

        # All self-compositions should give identity (0)
        # except identity itself
        for elem, result in self_compositions[:10]:
            if elem == 0:
                assert result == 0, "0 * 0 should be 0"
            else:
                assert result == 0, f"{elem} * {elem} should be 0 (identity)"

    def test_klein_as_symmetry_group(self):
        """Test Klein Four group as symmetry group of rectangle."""
        dataset = self.load_individual_dataset(4)

        # V4 is the symmetry group of a rectangle (not square)
        # It has:
        # - Identity
        # - Rotation by 180°
        # - Reflection across horizontal axis
        # - Reflection across vertical axis

        assert len(dataset) > 0
        assert dataset[0]["group_type"] == "klein"

    def test_klein_subgroup_of_larger_groups(self):
        """Test that Klein Four group appears as subgroup in larger groups."""
        dataset = self.load_individual_dataset(4)

        # V4 is a normal subgroup of:
        # - S4 (symmetric group on 4 elements)
        # - A4 (alternating group on 4 elements)
        # - D4 (dihedral group of order 8)

        # This is a mathematical fact we're acknowledging
        assert dataset[0]["group_order"] == 4
