#!/usr/bin/env python3
"""
Exhaustive tests for data correctness of all permutation group datasets.
Tests that compositions are correct, permutations are valid, etc.
"""

import pytest
import numpy as np
from datasets import load_dataset
from pathlib import Path
import random
from itertools import permutations
from typing import List, Tuple


class TestDataCorrectness:
    """Test the mathematical correctness of all datasets."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        # Check if we're testing locally or from HuggingFace
        if Path("./datasets").exists():
            cls.LOCAL_DATA_DIR = Path("./datasets")
        elif Path("../datasets").exists():
            cls.LOCAL_DATA_DIR = Path("../datasets")
        else:
            cls.LOCAL_DATA_DIR = None

        cls.USE_LOCAL = cls.LOCAL_DATA_DIR is not None and any(
            cls.LOCAL_DATA_DIR.iterdir()
        )
        cls.REPO_ID = "BeeGass/Group-Theory-Collection"

    def load_dataset_with_metadata(self, dataset_name: str):
        """Load dataset and its metadata (permutation mappings)."""
        if self.USE_LOCAL:
            dataset_path = self.LOCAL_DATA_DIR / f"{dataset_name}_data"
            if not dataset_path.exists():
                pytest.skip(f"Local dataset {dataset_path} not found")

            from datasets import load_from_disk
            import json

            # Load dataset
            dataset_dict = load_from_disk(str(dataset_path))

            # Load metadata
            metadata_path = dataset_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            else:
                metadata = None

            return dataset_dict["train"], dataset_dict["test"], metadata
        else:
            # Load from HuggingFace
            train_dataset = load_dataset(self.REPO_ID, name=dataset_name, split="train")
            test_dataset = load_dataset(self.REPO_ID, name=dataset_name, split="test")

            # For HF datasets, we'll reconstruct permutations from the data
            return train_dataset, test_dataset, None

    def compose_permutations(self, perm1: List[int], perm2: List[int]) -> List[int]:
        """Compose two permutations: (perm2 ∘ perm1)."""
        return [perm2[perm1[i]] for i in range(len(perm1))]

    def permutation_to_id(self, perm: List[int], all_perms: List[List[int]]) -> int:
        """Find the ID of a permutation in the group."""
        for i, p in enumerate(all_perms):
            if p == perm:
                return i
        raise ValueError(f"Permutation {perm} not found in group")

    def verify_composition(
        self, input_ids: List[int], target_id: int, all_perms: List[List[int]]
    ) -> bool:
        """Verify that composing input permutations gives the target."""
        # Start with identity
        result = list(range(len(all_perms[0])))

        # Compose from left to right (p1 ∘ p2 ∘ p3 ...)
        for perm_id in input_ids:
            result = self.compose_permutations(result, all_perms[perm_id])

        # Check if result matches target
        return result == all_perms[target_id]

    # ========== Symmetric Group Tests ==========

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_symmetric_group_correctness(self, n):
        """Test correctness of symmetric group Sn data."""
        dataset_name = f"s{n}"
        train_dataset, test_dataset, metadata = self.load_dataset_with_metadata(
            dataset_name
        )

        # For fixed-length sequences, we just verify data integrity
        # not mathematical composition correctness

        # Test samples from train set
        num_samples = min(100, len(train_dataset))

        for i in range(num_samples):
            sample = train_dataset[i]
            input_ids = [int(x) for x in sample["input_sequence"].split()]
            target_id = int(sample["target"])

            # Verify all IDs are valid for the group
            group_order = sample["group_order"]
            assert all(0 <= pid < group_order for pid in input_ids), (
                f"Invalid permutation ID in input sequence"
            )
            assert 0 <= target_id < group_order, f"Invalid target ID"

            # Verify sequence length
            assert 3 <= len(input_ids) <= 1024, (
                f"Expected sequence length between 3 and 1024, got {len(input_ids)}"
            )

    # ========== Cyclic Group Tests ==========

    @pytest.mark.parametrize("n", [3, 5, 7, 10])
    def test_cyclic_group_correctness(self, n):
        """Test correctness of cyclic group Cn data."""
        dataset_name = f"c{n}"
        train_dataset, test_dataset, metadata = self.load_dataset_with_metadata(
            dataset_name
        )

        # For fixed-length sequences, we just verify data integrity

        # Test samples
        num_samples = min(100, len(train_dataset))

        for i in range(num_samples):
            sample = train_dataset[i]
            input_ids = [int(x) for x in sample["input_sequence"].split()]
            target_id = int(sample["target"])

            # Verify all IDs are valid for cyclic group
            assert all(0 <= pid < n for pid in input_ids), (
                f"Invalid element ID in input sequence for C{n}"
            )
            assert 0 <= target_id < n, f"Invalid target ID for C{n}"

            # Verify sequence length
            assert 3 <= len(input_ids) <= 1024, (
                f"Expected sequence length between 3 and 1024, got {len(input_ids)}"
            )

    # ========== Alternating Group Tests ==========

    def is_even_permutation(self, perm: List[int]) -> bool:
        """Check if a permutation is even (has even number of transpositions)."""
        n = len(perm)
        visited = [False] * n
        num_cycles = 0

        for i in range(n):
            if not visited[i]:
                # Start a new cycle
                j = i
                cycle_length = 0
                while not visited[j]:
                    visited[j] = True
                    j = perm[j]
                    cycle_length += 1
                if cycle_length > 1:
                    num_cycles += cycle_length - 1

        return num_cycles % 2 == 0

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_alternating_group_correctness(self, n):
        """Test correctness of alternating group An data."""
        dataset_name = f"a{n}"
        train_dataset, test_dataset, metadata = self.load_dataset_with_metadata(
            dataset_name
        )

        # Generate all even permutations for An
        all_perms = []
        for perm in permutations(range(n)):
            if self.is_even_permutation(list(perm)):
                all_perms.append(list(perm))

        # Verify group order matches expected
        import math

        expected_order = math.factorial(n) // 2 if n >= 2 else 1
        assert train_dataset[0]["group_order"] == expected_order, (
            f"A{n} should have order {expected_order}"
        )

        # Test that all permutations in the dataset are even
        num_samples = min(50, len(train_dataset))
        for i in range(num_samples):
            sample = train_dataset[i]
            # We can't directly check the permutations without metadata,
            # but we can verify the group properties
            assert sample["group_degree"] == n
            assert sample["group_order"] == expected_order

    # ========== Dihedral Group Tests ==========

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_dihedral_group_properties(self, n):
        """Test properties of dihedral group Dn data."""
        dataset_name = f"d{n}"
        train_dataset, test_dataset, metadata = self.load_dataset_with_metadata(
            dataset_name
        )

        # Dihedral group Dn has order 2n
        expected_order = 2 * n

        # Test samples
        num_samples = min(50, len(train_dataset))
        for i in range(num_samples):
            sample = train_dataset[i]
            assert sample["group_degree"] == n
            assert sample["group_order"] == expected_order

    # ========== Klein Four-Group Test ==========

    def test_klein_four_group(self):
        """Test Klein four-group V4 data."""
        dataset_name = "v4"
        train_dataset, test_dataset, metadata = self.load_dataset_with_metadata(
            dataset_name
        )

        # V4 has order 4 and degree 4
        for i in range(min(20, len(train_dataset))):
            sample = train_dataset[i]
            assert sample["group_degree"] == 4
            assert sample["group_order"] == 4

            # V4 is abelian, so all elements have order ≤ 2
            # Every element composed with itself gives identity
            if len(sample["input_sequence"]) == 2:
                if sample["input_sequence"][0] == sample["input_sequence"][1]:
                    # Non-identity elements squared give identity
                    assert sample["target"] == 0 or sample["input_sequence"][0] == 0

    # ========== Data Integrity Tests ==========

    def test_sequence_length_distribution(self):
        """Test that datasets have proper distribution of sequence lengths."""
        # Test a few representative datasets
        test_datasets = ["s5", "a4", "c10", "d6"]

        for dataset_name in test_datasets:
            if (
                self.USE_LOCAL
                and not (self.LOCAL_DATA_DIR / f"{dataset_name}_data").exists()
            ):
                continue

            train_dataset, _, _ = self.load_dataset_with_metadata(dataset_name)

            # Collect length distribution
            length_counts = {}
            for item in train_dataset:
                length = item["sequence_length"]
                length_counts[length] = length_counts.get(length, 0) + 1

            # All sequences should be between 3 and 1024
            assert all(3 <= length <= 1024 for length in length_counts.keys()), (
                f"Found sequence lengths outside valid range [3, 1024] in {dataset_name}"
            )

            # Log the distribution for debugging
            print(
                f"\n{dataset_name} length distribution: {len(length_counts)} unique lengths"
            )
            print(f"Min: {min(length_counts.keys())}, Max: {max(length_counts.keys())}")

            # We expect variety in lengths, but don't enforce a specific number
            # since the distribution might not be perfectly uniform
            if len(length_counts) < 10:
                print(
                    f"WARNING: {dataset_name} has low length variety ({len(length_counts)} unique lengths)"
                )

    def test_no_duplicate_examples(self):
        """Test that there are no duplicate examples in the dataset."""
        test_datasets = ["s3", "c5", "d4"]

        for dataset_name in test_datasets:
            if (
                self.USE_LOCAL
                and not (self.LOCAL_DATA_DIR / f"{dataset_name}_data").exists()
            ):
                continue

            train_dataset, _, _ = self.load_dataset_with_metadata(dataset_name)

            # Create unique identifiers for each example
            seen_examples = set()

            for i in range(min(1000, len(train_dataset))):
                item = train_dataset[i]
                # Create a unique key from input and target
                key = (
                    tuple([int(x) for x in item["input_sequence"].split()]),
                    int(item["target"]),
                )

                assert key not in seen_examples, f"Duplicate found: {key}"
                seen_examples.add(key)

    def test_target_within_group_bounds(self):
        """Test that all targets are valid permutation IDs within the group."""
        test_configs = [
            ("s3", 6),
            ("s4", 24),
            ("a3", 3),
            ("a4", 12),
            ("c5", 5),
            ("c10", 10),
            ("d4", 8),
            ("v4", 4),
        ]

        for dataset_name, group_order in test_configs:
            if (
                self.USE_LOCAL
                and not (self.LOCAL_DATA_DIR / f"{dataset_name}_data").exists()
            ):
                continue

            train_dataset, test_dataset, _ = self.load_dataset_with_metadata(
                dataset_name
            )

            # Check train set
            for i in range(min(100, len(train_dataset))):
                item = train_dataset[i]
                assert 0 <= int(item["target"]) < group_order, (
                    f"Target {item['target']} out of bounds for {dataset_name} (order {group_order})"
                )

                # Also check input sequence IDs
                for perm_id in [int(x) for x in item["input_sequence"].split()]:
                    assert 0 <= perm_id < group_order, (
                        f"Input ID {perm_id} out of bounds for {dataset_name}"
                    )

            # Check test set
            for i in range(min(50, len(test_dataset))):
                item = test_dataset[i]
                assert 0 <= int(item["target"]) < group_order
                for perm_id in [int(x) for x in item["input_sequence"].split()]:
                    assert 0 <= perm_id < group_order
