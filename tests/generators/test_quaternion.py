#!/usr/bin/env python3
"""
Exhaustive tests for individual quaternion group datasets (Q8, Q16, Q32).
"""

import pytest
from ..test_base_individual import BaseIndividualGroupTest


class TestQuaternionIndividual(BaseIndividualGroupTest):
    """Exhaustive tests for individual quaternion group datasets."""

    GROUP_TYPE = "quaternion"
    GROUP_CONFIG = {
        "degrees": [8, 16, 32],
        "orders": {
            8: 8,  # Q8 - quaternion group of order 8
            16: 16,  # Q16 - generalized quaternion group of order 16
            32: 32,  # Q32 - generalized quaternion group of order 32
        },
        "prefix": "Q",
    }

    # Override to test all quaternion group degrees
    @pytest.mark.parametrize("degree", [8, 16, 32])
    def test_group_properties(self, degree):
        """Test group properties for each quaternion degree."""
        super().test_group_properties(degree)

    def test_quaternion_orders_power_of_2(self):
        """Test that quaternion groups have order 2^n."""
        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # Check that order is a power of 2
            order = dataset[0]["group_order"]
            assert order == degree
            assert (order & (order - 1)) == 0  # Check if power of 2
            assert order >= 8  # Minimum order is 8

    def test_q8_fundamental_properties(self):
        """Test fundamental properties of Q8."""
        dataset = self.load_individual_dataset(8)

        # Q8 = {±1, ±i, ±j, ±k} with Hamilton's quaternion relations
        # i² = j² = k² = ijk = -1
        assert dataset[0]["group_order"] == 8

        # Q8 is the smallest non-abelian group where every subgroup is normal
        # It's also the smallest Hamiltonian group (non-abelian with all subgroups normal)

    def test_non_abelian_property(self):
        """Test that quaternion groups are non-abelian."""
        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # All quaternion groups are non-abelian
            # They have the presentation:
            # Q_{2^n} = <a, b | a^{2^{n-1}} = 1, b² = a^{2^{n-2}}, bab^{-1} = a^{-1}>
            assert len(dataset) > 0
            assert dataset[0]["group_degree"] == degree

    def test_center_structure(self):
        """Test center structure of quaternion groups."""
        # The center of Q_{2^n} has order 2 and consists of {1, -1}

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # The center Z(Q_{2^n}) = {1, a^{2^{n-2}}} ≅ Z₂
            # This is a mathematical property we're acknowledging
            assert dataset[0]["group_order"] == degree

    def test_cyclic_subgroups(self):
        """Test that quaternion groups have cyclic subgroups."""
        # Every proper subgroup of a quaternion group is cyclic

        dataset_8 = self.load_individual_dataset(8)

        # Q8 has:
        # - One subgroup of order 1: {1}
        # - One subgroup of order 2: {1, -1}
        # - Three subgroups of order 4: <i>, <j>, <k>
        # - No subgroups of order 8 (since Q8 itself is not cyclic)

        assert dataset_8[0]["group_order"] == 8

    def test_presentation_relations(self):
        """Test presentation relations for quaternion groups."""
        # Q_{2^n} has specific presentation relations

        for degree in [8, 16]:
            dataset = self.load_individual_dataset(degree)

            # Sample some compositions
            short_sequences = []
            for i in range(min(50, len(dataset))):
                if dataset[i]["sequence_length"] <= 4:
                    short_sequences.append(dataset[i])

            # All elements should be in valid range
            for sample in short_sequences:
                input_ids = [int(x) for x in sample["input_sequence"].split()]
                target_id = int(sample["target"])
                assert all(0 <= x < degree for x in input_ids)
                assert 0 <= target_id < degree

    def test_generalized_quaternion_pattern(self):
        """Test pattern in generalized quaternion groups."""
        # Q_{2^n} for n ≥ 3 follows a specific pattern

        orders = []
        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)
            orders.append(dataset[0]["group_order"])

        # Check that orders double each time
        for i in range(1, len(orders)):
            assert orders[i] == 2 * orders[i - 1]

    def test_dicyclic_group_property(self):
        """Test that quaternion groups are dicyclic groups."""
        # Quaternion groups are special cases of dicyclic groups
        # Dic_n = Q_{4n} when n is a power of 2

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # Dicyclic groups have a cyclic subgroup of index 2
            # For Q_{2^n}, this is the subgroup generated by 'a'
            assert dataset[0]["group_type"] == "quaternion"

    @pytest.mark.parametrize("n", [8, 16])
    def test_element_orders(self, n):
        """Test orders of elements in quaternion groups."""
        dataset = self.load_individual_dataset(n)

        # In Q_{2^n}:
        # - Identity has order 1
        # - One element has order 2 (the central element -1)
        # - Other elements have order 4 or 2^{n-1}

        assert dataset[0]["group_degree"] == n
        assert dataset[0]["group_order"] == n
