#!/usr/bin/env python3
"""
Base test class for testing individual permutation group datasets.
Provides comprehensive testing framework for all group types.
"""

import pytest
from datasets import load_dataset
from pathlib import Path
import random
import json
from typing import List, Dict, Any


class BaseIndividualGroupTest:
    """Base class for exhaustive testing of individual permutation group datasets."""

    # Override these in subclasses
    GROUP_TYPE = None  # e.g., "symmetric", "alternating"
    GROUP_CONFIG = None  # Configuration with specific groups to test

    # Test parameters
    TEST_LENGTHS = [3, 4, 5, 8, 10, 16, 32, 50, 64, 100, 128, 256, 512, 1000, 1024]
    SAMPLE_SIZES = {
        "small": 100,  # For detailed checks
        "medium": 500,  # For statistical checks
        "large": 1000,  # For distribution checks
    }

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

    def get_dataset_name(self, degree):
        """Get the dataset name for a specific degree."""
        prefix = self.GROUP_CONFIG.get("prefix", "")
        if self.GROUP_TYPE == "elementary_abelian":
            # Special handling for elementary abelian groups
            if degree in [2, 4, 8, 16, 32]:
                p, k = 2, {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[degree]
                return f"z{p}_{k}"
            elif degree in [3, 9, 27]:
                p, k = 3, {3: 1, 9: 2, 27: 3}[degree]
                return f"z{p}_{k}"
            elif degree in [5, 25]:
                p, k = 5, {5: 1, 25: 2}[degree]
                return f"z{p}_{k}"
        elif self.GROUP_TYPE == "psl":
            # Special handling for PSL groups
            if degree == 6:
                return "psl2_5"
            elif degree == 8:
                return "psl2_7"
        elif self.GROUP_TYPE == "frobenius":
            # Special handling for Frobenius groups
            # Map from degree to order for dataset names
            if degree == 5:
                return "f20"  # F20 acts on 5 points
            elif degree == 7:
                return "f21"  # F21 acts on 7 points
        else:
            # Standard naming convention
            return f"{prefix.lower()}{degree}"

    def load_individual_dataset(self, degree, split="train"):
        """Load a specific individual dataset."""
        dataset_name = self.get_dataset_name(degree)

        if self.USE_LOCAL:
            # Load from local datasets directory
            dataset_path = self.LOCAL_DATA_DIR / f"{dataset_name}_data"
            if not dataset_path.exists():
                pytest.skip(f"Local dataset {dataset_path} not found")

            from datasets import load_from_disk

            dataset_dict = load_from_disk(str(dataset_path))
            return dataset_dict[split]
        else:
            # Load from HuggingFace
            return load_dataset(
                self.REPO_ID, data_dir=f"data/{dataset_name}", split=split
            )

    def load_metadata(self, degree) -> Dict[str, Any]:
        """Load metadata for a specific dataset if available."""
        if self.USE_LOCAL:
            dataset_name = self.get_dataset_name(degree)
            metadata_path = (
                self.LOCAL_DATA_DIR / f"{dataset_name}_data" / "metadata.json"
            )
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    return json.load(f)
        return None

    def filter_by_length(self, dataset, min_len=None, max_len=None):
        """Filter dataset by sequence length."""
        if min_len is not None and max_len is not None:
            return dataset.filter(lambda x: min_len <= x["sequence_length"] <= max_len)
        elif min_len is not None:
            return dataset.filter(lambda x: x["sequence_length"] >= min_len)
        elif max_len is not None:
            return dataset.filter(lambda x: x["sequence_length"] <= max_len)
        return dataset

    def verify_composition(
        self,
        input_ids: List[int],
        target_id: int,
        permutation_map: Dict[int, List[int]],
    ) -> bool:
        """Verify that composing input permutations gives the target."""
        if not permutation_map:
            return True  # Skip if no metadata available

        # Start with identity
        n = len(permutation_map[0])
        result = list(range(n))

        # Compose from left to right
        for perm_id in input_ids:
            perm = permutation_map[perm_id]
            result = [result[perm[i]] for i in range(n)]

        # Check if result matches target
        return result == permutation_map[target_id]

    # ========== Existence and Loading Tests ==========

    def test_all_datasets_exist(self):
        """Test that all configured datasets exist and can be loaded."""
        degrees = self.GROUP_CONFIG["degrees"]
        missing = []

        for degree in degrees:
            try:
                dataset = self.load_individual_dataset(degree, split="train")
                assert len(dataset) > 0
            except Exception as e:
                missing.append((degree, str(e)))

        assert len(missing) == 0, f"Missing datasets: {missing}"

    def test_train_test_splits_exist(self):
        """Test that both train and test splits exist for all datasets."""
        degrees = self.GROUP_CONFIG["degrees"]

        for degree in degrees:
            train_dataset = self.load_individual_dataset(degree, split="train")
            test_dataset = self.load_individual_dataset(degree, split="test")

            assert len(train_dataset) > 0, f"Train split empty for degree {degree}"
            assert len(test_dataset) > 0, f"Test split empty for degree {degree}"

    # ========== Data Structure Tests ==========

    def test_data_structure_consistency(self):
        """Test that all datasets have consistent structure."""
        degrees = self.GROUP_CONFIG["degrees"]

        # Check first dataset to establish expected structure
        if degrees:
            first_dataset = self.load_individual_dataset(degrees[0])
            expected_fields = set(first_dataset[0].keys())

            # Required fields
            required_fields = {
                "input_sequence",
                "target",
                "sequence_length",
                "group_degree",
                "group_order",
                "group_type",
            }
            assert required_fields.issubset(expected_fields), (
                f"Missing required fields: {required_fields - expected_fields}"
            )

            # Check all datasets have same structure
            for degree in degrees:
                dataset = self.load_individual_dataset(degree)
                assert set(dataset[0].keys()) == expected_fields, (
                    f"Inconsistent fields for degree {degree}"
                )

    def test_data_types(self):
        """Test that all fields have correct data types."""
        degrees = self.GROUP_CONFIG["degrees"]

        for degree in degrees[:3]:  # Test first 3 degrees
            dataset = self.load_individual_dataset(degree)

            # Sample multiple examples
            for i in range(min(50, len(dataset))):
                sample = dataset[i]

                # Check types
                assert isinstance(sample["input_sequence"], str), (
                    f"input_sequence not a string"
                )
                assert isinstance(sample["target"], str), "target not a string"
                assert isinstance(sample["sequence_length"], int), (
                    "sequence_length not an integer"
                )
                assert isinstance(sample["group_degree"], int), (
                    "group_degree not an integer"
                )
                assert isinstance(sample["group_order"], int), (
                    "group_order not an integer"
                )
                assert isinstance(sample["group_type"], str), "group_type not a string"

                # Parse string fields
                input_ids = [int(x) for x in sample["input_sequence"].split()]
                target_id = int(sample["target"])

                # Check value ranges
                assert target_id >= 0, "negative target"
                assert sample["sequence_length"] >= 3, "sequence too short"
                assert sample["sequence_length"] <= 1024, "sequence too long"
                assert len(input_ids) == sample["sequence_length"], (
                    "sequence_length mismatch"
                )
                assert all(x >= 0 for x in input_ids), (
                    "negative values in input_sequence"
                )

    # ========== Group Property Tests ==========

    @pytest.mark.parametrize("degree", [])  # Will be overridden in subclasses
    def test_group_properties(self, degree):
        """Test that group properties are correct for each degree."""
        dataset = self.load_individual_dataset(degree)
        expected_order = self.GROUP_CONFIG["orders"][degree]

        # Check multiple samples
        errors = []
        for i in range(min(100, len(dataset))):
            sample = dataset[i]

            if sample["group_degree"] != degree:
                errors.append(
                    f"Sample {i}: degree {sample['group_degree']} != {degree}"
                )
            if sample["group_order"] != expected_order:
                errors.append(
                    f"Sample {i}: order {sample['group_order']} != {expected_order}"
                )

        assert len(errors) == 0, f"Property errors:\n" + "\n".join(errors[:10])

    def test_permutation_id_bounds(self):
        """Test that all permutation IDs are within valid bounds."""
        degrees = self.GROUP_CONFIG["degrees"]

        for degree in degrees[:5]:  # Test first 5 degrees
            dataset = self.load_individual_dataset(degree)
            group_order = self.GROUP_CONFIG["orders"][degree]

            errors = []
            for i in range(min(200, len(dataset))):
                sample = dataset[i]

                # Parse string fields
                input_ids = [int(x) for x in sample["input_sequence"].split()]
                target_id = int(sample["target"])

                # Check target
                if not (0 <= target_id < group_order):
                    errors.append(f"Sample {i}: target {target_id} out of bounds")

                # Check input sequence
                for j, perm_id in enumerate(input_ids):
                    if not (0 <= perm_id < group_order):
                        errors.append(
                            f"Sample {i}, position {j}: ID {perm_id} out of bounds"
                        )

            assert len(errors) == 0, (
                f"ID bound errors for degree {degree}:\n" + "\n".join(errors[:10])
            )

    # ========== Length Distribution Tests ==========

    def test_length_consistency(self):
        """Test that the sequence_length field matches actual sequence length."""
        degrees = self.GROUP_CONFIG["degrees"]

        for degree in degrees[:3]:
            dataset = self.load_individual_dataset(degree)

            errors = []
            indices = random.sample(range(len(dataset)), min(500, len(dataset)))

            for idx in indices:
                sample = dataset[idx]
                input_ids = sample["input_sequence"].split()
                actual_length = len(input_ids)
                if sample["sequence_length"] != actual_length:
                    errors.append(
                        f"Index {idx}: sequence_length field {sample['sequence_length']} != actual {actual_length}"
                    )

            assert len(errors) == 0, (
                f"Length inconsistencies for degree {degree}:\n" + "\n".join(errors[:5])
            )

    def test_length_distribution(self):
        """Test that datasets have proper distribution of sequence lengths."""
        degrees = self.GROUP_CONFIG["degrees"]

        for degree in degrees[:2]:  # Test first 2 degrees
            dataset = self.load_individual_dataset(degree)

            # Collect length distribution
            length_counts = {}
            for i in range(min(2000, len(dataset))):
                length = dataset[i]["sequence_length"]
                length_counts[length] = length_counts.get(length, 0) + 1

            # Check coverage
            unique_lengths = sorted(length_counts.keys())
            assert min(unique_lengths) >= 3, f"Minimum length {min(unique_lengths)} < 3"
            assert max(unique_lengths) <= 1024, (
                f"Maximum length {max(unique_lengths)} > 1024"
            )

            # Should have good coverage of different lengths or fixed length
            if len(unique_lengths) == 1 and unique_lengths[0] == 1024:
                # Fixed length dataset - all sequences are 1024
                pass  # This is expected
            else:
                # Variable length dataset
                assert len(unique_lengths) >= 50, (
                    f"Only {len(unique_lengths)} unique lengths for degree {degree}"
                )

                # Check for major gaps
                for i in range(1, len(unique_lengths)):
                    gap = unique_lengths[i] - unique_lengths[i - 1]
                    assert gap <= 50, (
                        f"Large gap {gap} between lengths {unique_lengths[i - 1]} and {unique_lengths[i]}"
                    )

    @pytest.mark.parametrize("max_len", [8, 16, 32, 64, 128, 256, 512, 1024])
    def test_length_filtering(self, max_len):
        """Test filtering by different maximum lengths."""
        if self.GROUP_CONFIG["degrees"]:
            degree = self.GROUP_CONFIG["degrees"][0]
            dataset = self.load_individual_dataset(degree)

            # Filter dataset
            filtered = self.filter_by_length(dataset, max_len=max_len)

            # Verify all sequences respect the limit
            errors = []
            for i in range(min(100, len(filtered))):
                if filtered[i]["sequence_length"] > max_len:
                    errors.append(
                        f"Sample {i} has length {filtered[i]['sequence_length']} > {max_len}"
                    )

            assert len(errors) == 0, f"Length filtering failed:\n" + "\n".join(errors)

    # ========== Data Integrity Tests ==========

    def test_no_duplicates_in_dataset(self):
        """Test that there are no duplicate examples within a dataset."""
        degrees = self.GROUP_CONFIG["degrees"]

        for degree in degrees[:2]:  # Test first 2 degrees
            dataset = self.load_individual_dataset(degree)

            # Create signatures for examples
            seen = set()
            duplicates = []

            for i in range(min(1000, len(dataset))):
                sample = dataset[i]
                signature = (tuple(sample["input_sequence"]), sample["target"])

                if signature in seen:
                    duplicates.append(f"Duplicate at index {i}: {signature}")
                seen.add(signature)

            assert len(duplicates) == 0, (
                f"Duplicates found for degree {degree}:\n" + "\n".join(duplicates[:5])
            )

    def test_train_test_split_ratio(self):
        """Test that train/test split is approximately 80/20."""
        degrees = self.GROUP_CONFIG["degrees"]

        for degree in degrees[:3]:
            train_dataset = self.load_individual_dataset(degree, split="train")
            test_dataset = self.load_individual_dataset(degree, split="test")

            total = len(train_dataset) + len(test_dataset)
            train_ratio = len(train_dataset) / total

            assert 0.75 <= train_ratio <= 0.85, (
                f"Degree {degree}: train ratio {train_ratio:.2f} outside expected range [0.75, 0.85]"
            )

    def test_no_train_test_leakage(self):
        """Test that there's no data leakage between train and test sets."""
        degrees = self.GROUP_CONFIG["degrees"]

        for degree in degrees[:2]:  # Test first 2 degrees
            train_dataset = self.load_individual_dataset(degree, split="train")
            test_dataset = self.load_individual_dataset(degree, split="test")

            # Sample train signatures
            train_signatures = set()
            for i in range(min(1000, len(train_dataset))):
                sample = train_dataset[i]
                signature = (tuple(sample["input_sequence"]), sample["target"])
                train_signatures.add(signature)

            # Check test set
            leaks = []
            for i in range(min(500, len(test_dataset))):
                sample = test_dataset[i]
                signature = (tuple(sample["input_sequence"]), sample["target"])
                if signature in train_signatures:
                    leaks.append(f"Test sample {i} found in train: {signature}")

            assert len(leaks) == 0, f"Data leakage for degree {degree}:\n" + "\n".join(
                leaks[:5]
            )

    # ========== Composition Correctness Tests ==========

    def test_composition_correctness(self):
        """Test that permutation compositions are mathematically correct."""
        degrees = self.GROUP_CONFIG["degrees"]

        for degree in degrees[:2]:  # Test first 2 degrees
            metadata = self.load_metadata(degree)
            if not metadata:
                continue  # Skip if no metadata available

            dataset = self.load_individual_dataset(degree)
            permutation_map = {
                int(k): v for k, v in metadata.get("permutation_map", {}).items()
            }

            errors = []
            indices = random.sample(range(len(dataset)), min(100, len(dataset)))

            for idx in indices:
                sample = dataset[idx]
                input_ids = [int(x) for x in sample["input_sequence"].split()]
                target_id = int(sample["target"])

                if not self.verify_composition(input_ids, target_id, permutation_map):
                    errors.append(f"Sample {idx}: {input_ids} -> {target_id}")

            assert len(errors) == 0, (
                f"Composition errors for degree {degree}:\n" + "\n".join(errors[:10])
            )

    # ========== Statistical Tests ==========

    def test_target_distribution(self):
        """Test that targets are well-distributed across the group."""
        degrees = self.GROUP_CONFIG["degrees"]

        for degree in degrees[:2]:
            dataset = self.load_individual_dataset(degree)
            group_order = self.GROUP_CONFIG["orders"][degree]

            # Sample targets
            target_counts = {}
            for i in range(min(1000, len(dataset))):
                target = int(dataset[i]["target"])
                target_counts[target] = target_counts.get(target, 0) + 1

            # Check coverage (should see most group elements as targets)
            coverage = len(target_counts) / group_order
            
            # For large groups, we can't expect to see 50% of all elements in just 1000 samples
            # Adjust expectation based on group size
            if group_order > 1000:
                # For large groups, expect to see at least 10% or 100 unique targets, whichever is smaller
                expected_coverage = min(0.1, 100.0 / group_order)
            else:
                # For small groups, maintain the 50% expectation
                expected_coverage = 0.5
                
            assert coverage >= expected_coverage, (
                f"Poor target coverage {coverage:.2f} for degree {degree}, expected at least {expected_coverage:.2f}"
            )

            # Check for extreme imbalance
            counts = list(target_counts.values())
            max_count = max(counts)
            avg_count = sum(counts) / len(counts)

            assert max_count <= 10 * avg_count, (
                f"Extreme imbalance in target distribution for degree {degree}"
            )

    def test_input_id_distribution(self):
        """Test that input permutation IDs are well-distributed."""
        degrees = self.GROUP_CONFIG["degrees"]

        for degree in degrees[:1]:  # Test first degree
            dataset = self.load_individual_dataset(degree)
            group_order = self.GROUP_CONFIG["orders"][degree]

            # Collect all input IDs
            id_counts = {}
            for i in range(min(500, len(dataset))):
                input_ids = [int(x) for x in dataset[i]["input_sequence"].split()]
                for perm_id in input_ids:
                    id_counts[perm_id] = id_counts.get(perm_id, 0) + 1

            # Check coverage
            coverage = len(id_counts) / group_order
            assert coverage >= 0.3, (
                f"Poor input ID coverage {coverage:.2f} for degree {degree}"
            )
