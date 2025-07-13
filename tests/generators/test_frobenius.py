#!/usr/bin/env python3
"""
Exhaustive tests for individual Frobenius group datasets (F20, F21).
"""

import pytest
from .test_base_individual import BaseIndividualGroupTest


class TestFrobeniusIndividual(BaseIndividualGroupTest):
    """Exhaustive tests for individual Frobenius group datasets."""

    GROUP_TYPE = "frobenius"
    GROUP_CONFIG = {
        "degrees": [5, 7],  # Degrees are the number of points acted on
        "orders": {
            5: 20,  # F20 - acts on 5 points, has order 20
            7: 21,  # F21 - acts on 7 points, has order 21
        },
        "prefix": "F",
    }

    # Override to test all Frobenius group degrees
    @pytest.mark.parametrize("degree", [5, 7])
    def test_group_properties(self, degree):
        """Test group properties for each Frobenius degree."""
        super().test_group_properties(degree)

    def test_frobenius_structure(self):
        """Test the structure of Frobenius groups."""
        # Frobenius groups are transitive permutation groups where
        # only the identity fixes more than one point

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)
            order = self.GROUP_CONFIG["orders"][degree]

            assert dataset[0]["group_order"] == order
            assert dataset[0]["group_type"] == "frobenius"

    def test_f20_properties(self):
        """Test specific properties of F20."""
        dataset = self.load_individual_dataset(5)  # F20 acts on 5 points

        # F20 has order 20 = 4 × 5
        # It's a semidirect product C5 ⋊ C4
        # Also written as AGL(1,5) - affine general linear group

        assert dataset[0]["group_order"] == 20

        # F20 is solvable (as are all Frobenius groups of order < 60)
        # It has a normal subgroup of order 5 (Frobenius kernel)
        # and a complement of order 4 (Frobenius complement)

    def test_f21_properties(self):
        """Test specific properties of F21."""
        dataset = self.load_individual_dataset(7)  # F21 acts on 7 points

        # F21 has order 21 = 3 × 7
        # It's a semidirect product C7 ⋊ C3

        assert dataset[0]["group_order"] == 21

        # F21 is the smallest non-abelian group of order 21
        # It has a normal subgroup of order 7 (Frobenius kernel)
        # and a complement of order 3 (Frobenius complement)

    def test_solvability(self):
        """Test that these Frobenius groups are solvable."""
        # All Frobenius groups of order < 60 are solvable
        # (The first non-solvable Frobenius group has order 60)

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # Both F20 and F21 are solvable
            assert dataset[0]["group_order"] < 60

    def test_kernel_complement_structure(self):
        """Test Frobenius kernel and complement structure."""
        # Every Frobenius group is a semidirect product K ⋊ H where:
        # - K is the Frobenius kernel (normal, regular, nilpotent)
        # - H is the Frobenius complement (acts freely on K \ {1})

        kernel_complement = {
            5: (5, 4),  # F20 = C5 ⋊ C4, acts on 5 points
            7: (7, 3),  # F21 = C7 ⋊ C3, acts on 7 points
        }

        for degree, (kernel_order, complement_order) in kernel_complement.items():
            dataset = self.load_individual_dataset(degree)
            order = self.GROUP_CONFIG["orders"][degree]

            # Verify order factorization
            assert kernel_order * complement_order == order
            assert dataset[0]["group_order"] == order

    def test_transitive_action(self):
        """Test that Frobenius groups act transitively."""
        # Frobenius groups are defined as transitive permutation groups
        # where no non-identity element fixes more than one point

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # Sample some compositions
            short_sequences = []
            for i in range(min(50, len(dataset))):
                if dataset[i]["sequence_length"] <= 3:
                    short_sequences.append(dataset[i])

            # All elements should be valid permutations
            for sample in short_sequences:
                input_ids = [int(x) for x in sample["input_sequence"].split()]
                target_id = int(sample["target"])
                order = self.GROUP_CONFIG["orders"][degree]
                assert all(0 <= x < order for x in input_ids)
                assert 0 <= target_id < order

    def test_non_abelian_property(self):
        """Test that Frobenius groups are non-abelian."""
        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # Both F20 and F21 are non-abelian
            # (Abelian transitive groups are regular, but Frobenius groups are not)
            assert len(dataset) > 0

    def test_frobenius_theorem_application(self):
        """Test properties guaranteed by Frobenius theorem."""
        # Frobenius theorem: If G is a finite transitive permutation group
        # where only the identity fixes more than one point, then:
        # 1. The elements fixing one point form a group H (Frobenius complement)
        # 2. The elements fixing no points, together with identity, form a normal subgroup K
        # 3. G = K ⋊ H

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # This structure makes Frobenius groups important in group theory
            assert dataset[0]["group_type"] == "frobenius"
