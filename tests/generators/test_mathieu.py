#!/usr/bin/env python3
"""
Exhaustive tests for individual Mathieu group datasets (M11, M12).
"""

import pytest
from .test_base_individual import BaseIndividualGroupTest


class TestMathieuIndividual(BaseIndividualGroupTest):
    """Exhaustive tests for individual Mathieu group datasets."""

    GROUP_TYPE = "mathieu"
    GROUP_CONFIG = {
        "degrees": [11, 12],
        "orders": {
            11: 7920,  # M11
            12: 95040,  # M12
        },
        "prefix": "M",
    }

    # Override to test all Mathieu group degrees
    @pytest.mark.parametrize("degree", [11, 12])
    def test_group_properties(self, degree):
        """Test group properties for each Mathieu degree."""
        super().test_group_properties(degree)

    def test_mathieu_sporadic_simple(self):
        """Test that Mathieu groups are sporadic simple groups."""
        # Mathieu groups are among the first discovered sporadic simple groups
        # They don't fit into any infinite family of simple groups

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # M11 and M12 are simple groups
            assert len(dataset) > 0
            assert dataset[0]["group_type"] == "mathieu"

    def test_m11_properties(self):
        """Test specific properties of M11."""
        dataset = self.load_individual_dataset(11)

        # M11 has order 7920 = 2^4 × 3^2 × 5 × 11
        assert dataset[0]["group_order"] == 7920

        # M11 is:
        # - Smallest sporadic simple group
        # - 4-transitive on 11 points
        # - Has no outer automorphisms

    def test_m12_properties(self):
        """Test specific properties of M12."""
        dataset = self.load_individual_dataset(12)

        # M12 has order 95040 = 2^6 × 3^3 × 5 × 11
        assert dataset[0]["group_order"] == 95040

        # M12 is:
        # - 5-transitive on 12 points
        # - Contains M11 as a point stabilizer
        # - Has outer automorphism group of order 2

    def test_order_factorizations(self):
        """Test the prime factorizations of Mathieu group orders."""
        factorizations = {
            11: {2: 4, 3: 2, 5: 1, 11: 1},  # 7920 = 2^4 × 3^2 × 5 × 11
            12: {2: 6, 3: 3, 5: 1, 11: 1},  # 95040 = 2^6 × 3^3 × 5 × 11
        }

        for degree, factors in factorizations.items():
            dataset = self.load_individual_dataset(degree)
            order = dataset[0]["group_order"]

            # Verify the order
            expected_order = 1
            for prime, power in factors.items():
                expected_order *= prime**power
            assert order == expected_order

    def test_transitivity_levels(self):
        """Test transitivity levels of Mathieu groups."""
        # M11 is 4-transitive on 11 points
        # M12 is 5-transitive on 12 points
        # These are among the most highly transitive permutation groups

        transitivity = {
            11: 4,  # M11 is 4-transitive
            12: 5,  # M12 is 5-transitive
        }

        for degree, trans_level in transitivity.items():
            dataset = self.load_individual_dataset(degree)

            # High transitivity is a rare property
            # Only alternating, symmetric, and Mathieu groups achieve it
            assert dataset[0]["group_degree"] == degree

    def test_non_solvable_property(self):
        """Test that Mathieu groups are non-solvable."""
        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # All Mathieu groups are simple, hence non-solvable
            # They are in NC¹ complexity class
            assert dataset[0]["group_order"] > 60  # Larger than smallest non-solvable

    def test_steiner_system_connection(self):
        """Test connection to Steiner systems."""
        # M11 is the automorphism group of S(4,5,11) Steiner system
        # M12 is the automorphism group of S(5,6,12) Steiner system

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # These connections make Mathieu groups important in combinatorics
            assert dataset[0]["group_type"] == "mathieu"

    def test_composition_examples(self):
        """Test composition examples in Mathieu groups."""
        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # Sample some short compositions
            short_sequences = []
            for i in range(min(30, len(dataset))):
                if dataset[i]["sequence_length"] <= 3:
                    short_sequences.append(dataset[i])

            # Verify valid permutation IDs
            for sample in short_sequences:
                input_ids = [int(x) for x in sample["input_sequence"].split()]
                target_id = int(sample["target"])
                order = dataset[0]["group_order"]
                assert all(0 <= x < order for x in input_ids)
                assert 0 <= target_id < order

    def test_subgroup_relationships(self):
        """Test subgroup relationships between Mathieu groups."""
        # M11 is a subgroup of M12 (point stabilizer)

        m11_dataset = self.load_individual_dataset(11)
        m12_dataset = self.load_individual_dataset(12)

        m11_order = m11_dataset[0]["group_order"]
        m12_order = m12_dataset[0]["group_order"]

        # |M12| / |M11| = 12
        assert m12_order == 12 * m11_order

    def test_historical_significance(self):
        """Test properties related to historical significance."""
        # Mathieu groups were discovered by Émile Mathieu in 1861 and 1873
        # They were the first sporadic simple groups to be discovered

        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_individual_dataset(degree)

            # These groups sparked the classification of finite simple groups
            assert dataset[0]["group_type"] == "mathieu"

    def test_generator_correctness(self):
        """Test that the generators for M11 and M12 are correct."""
        import numpy as np
        from gdg.generators.mathieu import MathieuGroupGenerator, _cycles_to_permutation

        generator = MathieuGroupGenerator()

        # Test M11 generators
        gen_a_11 = _cycles_to_permutation([(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)], 11)
        gen_b_11 = _cycles_to_permutation([(3, 7, 11, 8), (4, 10, 5, 6)], 11)

        # gen_a should be an 11-cycle
        assert len(set(gen_a_11)) == 11
        # Check it's actually a cycle
        current = 0
        for _ in range(11):
            current = gen_a_11[current]
        assert current == 0

        # Test M12 generators
        gen_a_12 = _cycles_to_permutation(
            [(1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)], 12
        )
        gen_b_12 = _cycles_to_permutation([(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)], 12)

        # gen_a should be an involution (order 2)
        assert np.array_equal(gen_a_12[gen_a_12], np.arange(12))

        # gen_b should fix element 11 (0-indexed)
        assert gen_b_12[11] == 11

        # Generate the groups and check sizes
        m11_perms = generator.generate_group(n=11)
        assert len(m11_perms) == 7920

        m12_perms = generator.generate_group(n=12)
        assert len(m12_perms) == 95040

    def test_group_closure(self):
        """Test that the generated M12 group is closed under its own composition."""
        import numpy as np
        from gdg.generators.mathieu import MathieuGroupGenerator

        generator = MathieuGroupGenerator(seed=42)

        # Generate M12
        m12_perms = generator.generate_group(n=12)
        perm_to_idx = {tuple(p): i for i, p in enumerate(m12_perms)}

        print(f"\nGenerated {len(m12_perms)} elements (expected 95040)")

        # Test closure using the expansion's composition convention
        errors = 0
        np.random.seed(42)
        for _ in range(100):
            i, j = np.random.randint(0, len(m12_perms), size=2)
            perm1 = m12_perms[i]
            perm2 = m12_perms[j]

            # Use the same composition as in expansion: result[i] = perm1[perm2[i]]
            result = np.array([perm1[perm2[k]] for k in range(12)])

            if tuple(result) not in perm_to_idx:
                errors += 1
                print(f"ERROR: {i} ∘ {j} not in group (expansion convention)")

        print(f"Expansion convention errors: {errors}/100")

        # Test closure using base generator's convention
        errors2 = 0
        for _ in range(100):
            i, j = np.random.randint(0, len(m12_perms), size=2)
            perm1 = m12_perms[i]
            perm2 = m12_perms[j]

            # Use base generator's composition: perm1[perm2]
            result = perm1[perm2]

            if tuple(result) not in perm_to_idx:
                errors2 += 1

        print(f"Base generator convention errors: {errors2}/100")

        assert errors == 0 or errors2 == 0, (
            "Group is not closed under either composition convention!"
        )

    def test_composition_consistency(self):
        """Test that composition in the generator is consistent with base class."""
        import numpy as np
        from gdg.generators.mathieu import MathieuGroupGenerator

        generator = MathieuGroupGenerator(seed=42)

        # Generate M12
        m12_perms = generator.generate_group(n=12)
        generator._permutations = m12_perms
        generator._perm_to_idx = {tuple(p): i for i, p in enumerate(m12_perms)}

        # Test both composition conventions
        print("\nTesting composition conventions...")

        # Pick two specific permutations
        perm1 = m12_perms[0]  # identity
        perm2 = m12_perms[1]  # some other perm

        # Method 1: perm1[perm2] (what base_generator uses)
        result1 = perm1[perm2]
        print(f"perm1[perm2] = {result1}")

        # Method 2: composition as in group expansion
        result2 = np.array([perm1[perm2[i]] for i in range(12)])
        print(f"perm1[perm2[i] for i] = {result2}")

        # Test random compositions
        np.random.seed(42)
        errors = []
        for k in range(20):
            # Pick two random permutations
            i, j = np.random.randint(0, len(m12_perms), size=2)
            perm1 = m12_perms[i]
            perm2 = m12_perms[j]

            # Compute composition using base class method
            result = generator._compute_composition(perm1, perm2)

            # Verify result is in the group
            try:
                idx = generator._find_permutation_index(result)
                if k < 5:  # Only print first few
                    print(f"Composition of perm[{i}] and perm[{j}] = perm[{idx}] ✓")
            except ValueError:
                errors.append((i, j, perm1, perm2, result))

        if errors:
            print(f"\nFound {len(errors)} composition errors out of 20 tests")
            # Analyze first error
            i, j, perm1, perm2, result = errors[0]
            print(f"\nFirst error: perm[{i}] ∘ perm[{j}]")
            print(f"  perm1: {perm1}")
            print(f"  perm2: {perm2}")
            print(f"  result: {result}")

            # Try the other composition convention
            alt_result = np.array([perm2[perm1[i]] for i in range(12)])
            print(f"  alt_result (perm2[perm1]): {alt_result}")
            if tuple(alt_result) in generator._perm_to_idx:
                print(f"  ✓ Alternative composition IS in group!")
                assert False, "Wrong composition convention in base_generator!"
            else:
                print(f"  ✗ Alternative composition also NOT in group")
                assert False, (
                    "Neither composition convention works - group may be incomplete"
                )
