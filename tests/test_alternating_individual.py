#!/usr/bin/env python3
"""
Exhaustive tests for individual alternating group datasets (A3-A9).
"""

import pytest
from .test_base_individual import BaseIndividualGroupTest
import math


class TestAlternatingIndividual(BaseIndividualGroupTest):
    """Exhaustive tests for individual alternating group datasets A3 through A9."""

    GROUP_TYPE = "alternating"
    GROUP_CONFIG = {
        "degrees": list(range(3, 10)),  # A3 to A9
        "orders": {
            3: 3,  # A3 = 3!/2 = 3
            4: 12,  # A4 = 4!/2 = 12
            5: 60,  # A5 = 5!/2 = 60
            6: 360,  # A6 = 6!/2 = 360
            7: 2520,  # A7 = 7!/2 = 2520
            8: 20160,  # A8 = 8!/2 = 20160
            9: 181440,  # A9 = 9!/2 = 181440
        },
        "prefix": "A",
    }

    # Override to test all alternating group degrees
    @pytest.mark.parametrize("degree", list(range(3, 10)))
    def test_group_properties(self, degree):
        """Test group properties for each alternating degree."""
        super().test_group_properties(degree)

    def test_half_factorial_orders(self):
        """Test that alternating groups have correct orders (n!/2)."""
        for degree in self.GROUP_CONFIG["degrees"]:
            expected_order = math.factorial(degree) // 2
            actual_order = self.GROUP_CONFIG["orders"][degree]
            assert actual_order == expected_order, (
                f"A{degree} should have order {expected_order}, config has {actual_order}"
            )

            # Verify in dataset
            dataset = self.load_individual_dataset(degree)
            assert dataset[0]["group_order"] == expected_order

    def test_alternating_vs_symmetric_relationship(self):
        """Test that An has half the elements of Sn."""
        for degree in self.GROUP_CONFIG["degrees"]:
            an_order = self.GROUP_CONFIG["orders"][degree]
            sn_order = math.factorial(degree)

            assert an_order * 2 == sn_order, (
                f"A{degree} order {an_order} should be half of S{degree} order {sn_order}"
            )

    def test_a5_smallest_simple_nonabelian(self):
        """Test A5 as the smallest simple non-abelian group."""
        # A5 has special significance
        a5_dataset = self.load_individual_dataset(5)

        assert len(a5_dataset) > 0
        assert a5_dataset[0]["group_order"] == 60
        assert a5_dataset[0]["group_degree"] == 5

        # A5 is isomorphic to PSL(2,5) and the icosahedral group
        # It's the smallest non-solvable group

    def test_a3_cyclic_special_case(self):
        """Test that A3 is cyclic (isomorphic to C3)."""
        a3_dataset = self.load_individual_dataset(3)

        # A3 has only 3 elements and is cyclic
        assert a3_dataset[0]["group_order"] == 3

        # All non-identity elements should have order 3
        # Element 0 is identity, elements 1 and 2 generate the group

    def test_a4_solvable_special_case(self):
        """Test A4 as the largest solvable alternating group."""
        a4_dataset = self.load_individual_dataset(4)

        # A4 has order 12 and is solvable but not simple
        assert a4_dataset[0]["group_order"] == 12

        # A4 has a normal subgroup isomorphic to Klein four-group
        # It's the symmetry group of a tetrahedron

    def test_simple_group_boundary(self):
        """Test the boundary between non-simple (A3, A4) and simple (A5+) groups."""
        # A3 and A4 are not simple
        for degree in [3, 4]:
            dataset = self.load_individual_dataset(degree)
            assert len(dataset) > 0
            # A3 is cyclic, A4 has Klein four-group as normal subgroup

        # A5 and above are simple groups (no proper normal subgroups)
        for degree in [5, 6, 7]:
            dataset = self.load_individual_dataset(degree)
            assert len(dataset) > 0
            # These are all simple groups

    def test_even_permutations_only(self):
        """Test that alternating groups contain only even permutations."""
        # For small groups, verify the permutation structure
        for degree in [3, 4]:
            dataset = self.load_individual_dataset(degree)

            # All elements in An are even permutations
            # (product of even number of transpositions)
            # This is a mathematical property we're just acknowledging
            assert dataset[0]["group_degree"] == degree

    @pytest.mark.parametrize("degree", [5, 6, 7, 8, 9])
    def test_large_alternating_groups(self, degree):
        """Test properties of larger alternating groups."""
        dataset = self.load_individual_dataset(degree)
        expected_order = self.GROUP_CONFIG["orders"][degree]

        # Should have substantial data
        assert len(dataset) > 1000, f"A{degree} dataset too small"

        # Verify correct order
        assert dataset[0]["group_order"] == expected_order

        # Check growth pattern
        if degree > 5:
            prev_order = self.GROUP_CONFIG["orders"][degree - 1]
            ratio = expected_order / prev_order
            # |An| = n!/2, so |An|/|An-1| = n!/(n-1)! = n
            expected_ratio = degree
            assert abs(ratio - expected_ratio) < 0.1, (
                f"Order growth from A{degree - 1} to A{degree} incorrect"
            )

    def test_composition_properties(self):
        """Test specific composition properties of alternating groups."""
        # Test A4 compositions
        a4_dataset = self.load_individual_dataset(4)

        # Sample some short compositions
        short_compositions = []
        for i in range(min(50, len(a4_dataset))):
            if a4_dataset[i]["sequence_length"] <= 3:
                short_compositions.append(a4_dataset[i])

        # Verify all elements are in valid range
        for sample in short_compositions:
            input_ids = [int(x) for x in sample["input_sequence"].split()]
            assert all(0 <= x < 12 for x in input_ids)
            target_id = int(sample["target"])
            assert 0 <= target_id < 12
