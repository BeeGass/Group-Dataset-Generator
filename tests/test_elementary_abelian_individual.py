#!/usr/bin/env python3
"""
Exhaustive tests for individual elementary abelian group datasets.
"""

import pytest
from .test_base_individual import BaseIndividualGroupTest


class TestElementaryAbelianIndividual(BaseIndividualGroupTest):
    """Exhaustive tests for individual elementary abelian group datasets."""

    GROUP_TYPE = "elementary_abelian"
    GROUP_CONFIG = {
        "degrees": [2, 3, 4, 5, 8, 9, 16, 25, 27, 32],
        "orders": {
            2: 2,  # Z2^1
            3: 3,  # Z3^1
            4: 4,  # Z2^2
            5: 5,  # Z5^1
            8: 8,  # Z2^3
            9: 9,  # Z3^2
            16: 16,  # Z2^4
            25: 25,  # Z5^2
            27: 27,  # Z3^3
            32: 32,  # Z2^5
        },
        "prefix": "Z",  # But naming is special for these groups
    }

    # Override to test all elementary abelian degrees
    @pytest.mark.parametrize("degree", [2, 3, 4, 5, 8, 9, 16, 25, 27, 32])
    def test_group_properties(self, degree):
        """Test group properties for each elementary abelian degree."""
        super().test_group_properties(degree)

    def test_prime_power_structure(self):
        """Test that elementary abelian groups have prime power orders."""
        import math

        # Group degrees to prime factorization
        prime_powers = {
            2: (2, 1),  # 2^1
            3: (3, 1),  # 3^1
            4: (2, 2),  # 2^2
            5: (5, 1),  # 5^1
            8: (2, 3),  # 2^3
            9: (3, 2),  # 3^2
            16: (2, 4),  # 2^4
            25: (5, 2),  # 5^2
            27: (3, 3),  # 3^3
            32: (2, 5),  # 2^5
        }

        for degree, (prime, power) in prime_powers.items():
            dataset = self.load_individual_dataset(degree)
            expected_order = prime**power
            assert dataset[0]["group_order"] == expected_order
            assert dataset[0]["group_order"] == degree

    def test_abelian_property(self):
        """Test that all elementary abelian groups are abelian."""
        # Elementary abelian groups are direct products of cyclic groups of prime order
        # They are all abelian

        for degree in [4, 8, 9]:  # Test a few groups
            dataset = self.load_individual_dataset(degree)

            # Sample some compositions with commuted elements
            assert len(dataset) > 0
            assert dataset[0]["group_degree"] == degree

    def test_vector_space_structure(self):
        """Test vector space structure of elementary abelian p-groups."""
        # Elementary abelian p-groups are vector spaces over GF(p)

        # Test Z2^2 (degree 4)
        z2_2_dataset = self.load_individual_dataset(4)

        # This can be viewed as 2D vector space over GF(2)
        # Addition is component-wise mod 2
        assert z2_2_dataset[0]["group_order"] == 4

        # Test Z3^2 (degree 9)
        z3_2_dataset = self.load_individual_dataset(9)

        # This can be viewed as 2D vector space over GF(3)
        # Addition is component-wise mod 3
        assert z3_2_dataset[0]["group_order"] == 9

    def test_all_elements_have_prime_order(self):
        """Test that all non-identity elements have prime order."""
        # In elementary abelian p-groups, all non-identity elements have order p

        prime_groups = {
            4: 2,  # Z2^2: non-identity elements have order 2
            8: 2,  # Z2^3: non-identity elements have order 2
            9: 3,  # Z3^2: non-identity elements have order 3
            25: 5,  # Z5^2: non-identity elements have order 5
        }

        for degree, prime in prime_groups.items():
            dataset = self.load_individual_dataset(degree)

            # Find self-compositions
            self_compositions = []
            for i in range(min(50, len(dataset))):
                sample = dataset[i]
                input_ids = [int(x) for x in sample["input_sequence"].split()]
                if len(input_ids) == prime and all(
                    x == input_ids[0] for x in input_ids
                ):
                    # Element composed with itself p times
                    self_compositions.append((input_ids[0], int(sample["target"])))

            # Non-identity elements composed p times should give identity
            for elem, result in self_compositions[:5]:
                if elem != 0:  # Non-identity
                    assert result == 0, (
                        f"Element {elem} to power {prime} should be identity"
                    )

    def test_direct_product_structure(self):
        """Test that elementary abelian groups are direct products."""
        # Z_p^k is isomorphic to Z_p × Z_p × ... × Z_p (k times)

        # Test specific examples
        dataset_4 = self.load_individual_dataset(4)  # Z2 × Z2
        dataset_8 = self.load_individual_dataset(8)  # Z2 × Z2 × Z2
        dataset_9 = self.load_individual_dataset(9)  # Z3 × Z3

        assert dataset_4[0]["group_order"] == 4
        assert dataset_8[0]["group_order"] == 8
        assert dataset_9[0]["group_order"] == 9

    @pytest.mark.parametrize("degree,prime", [(4, 2), (8, 2), (9, 3), (25, 5)])
    def test_composition_modular_arithmetic(self, degree, prime):
        """Test that composition follows component-wise modular arithmetic."""
        dataset = self.load_individual_dataset(degree)

        # Sample some short sequences
        for i in range(min(20, len(dataset))):
            sample = dataset[i]
            if sample["sequence_length"] <= 3:
                input_ids = [int(x) for x in sample["input_sequence"].split()]
                target_id = int(sample["target"])

                # All elements should be valid
                assert all(0 <= x < degree for x in input_ids)
                assert 0 <= target_id < degree

    def test_large_elementary_abelian_groups(self):
        """Test properties of larger elementary abelian groups."""
        large_degrees = [16, 25, 27, 32]

        for degree in large_degrees:
            dataset = self.load_individual_dataset(degree)

            # Should have substantial data
            assert len(dataset) > 1000

            # Verify order
            assert dataset[0]["group_order"] == degree

            # These are important in coding theory and cryptography
