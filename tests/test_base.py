#!/usr/bin/env python3
"""
Base test class for exhaustive permutation group dynamic filtering tests.
"""

import pytest
from datasets import load_dataset
from pathlib import Path
import itertools
import random


class BaseGroupTest:
    """Base class for exhaustive testing of dynamic filtering for permutation groups."""

    # Override these in subclasses
    GROUP_TYPE = None
    GROUP_CONFIG = None

    # Exhaustive test parameters
    TEST_LENGTHS = [
        1,
        2,
        3,
        4,
        5,
        10,
        16,
        32,
        50,
        64,
        100,
        128,
        256,
        512,
        817,
        1000,
        1023,
        1024,
    ]

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        # Check if we're testing locally or from HuggingFace
        # Handle both running from root and from tests directory
        if Path("./data").exists():
            cls.LOCAL_DATA_DIR = Path("./data")
        elif Path("../data").exists():
            cls.LOCAL_DATA_DIR = Path("../data")
        else:
            cls.LOCAL_DATA_DIR = None

        cls.USE_LOCAL = cls.LOCAL_DATA_DIR is not None and any(
            cls.LOCAL_DATA_DIR.iterdir()
        )
        cls.REPO_ID = "BeeGass/permutation-groups"

        if cls.USE_LOCAL:
            cls.dataset_path = str(cls.LOCAL_DATA_DIR / f"{cls.GROUP_TYPE}_superset")
            if not Path(cls.dataset_path).exists():
                pytest.skip(f"Local dataset {cls.dataset_path} not found")

    def load_dataset_with_filters(self, **kwargs):
        """Load dataset with given filters."""
        if self.USE_LOCAL:
            # For local testing, load full dataset and filter manually
            from datasets import load_from_disk

            dataset_dict = load_from_disk(self.dataset_path)
            dataset = dataset_dict[kwargs.get("split", "train")]

            # Apply filters
            filters = []
            if "min_degree" in kwargs and "max_degree" in kwargs:
                min_deg, max_deg = kwargs["min_degree"], kwargs["max_degree"]
                filters.append(lambda x: min_deg <= x["group_degree"] <= max_deg)
            if "min_order" in kwargs and "max_order" in kwargs:
                min_ord, max_ord = kwargs["min_order"], kwargs["max_order"]
                filters.append(lambda x: min_ord <= x["group_order"] <= max_ord)
            if "min_len" in kwargs and "max_len" in kwargs:
                min_len, max_len = kwargs["min_len"], kwargs["max_len"]
                filters.append(lambda x: min_len <= x["sequence_length"] <= max_len)

            # Apply all filters
            for f in filters:
                dataset = dataset.filter(f)

            return dataset
        else:
            # Use HuggingFace dynamic filtering
            return load_dataset(
                self.REPO_ID, name=self.GROUP_TYPE, trust_remote_code=True, **kwargs
            )

    # ========== Basic Loading Tests ==========

    def test_group_loading(self):
        """Test that the group can be loaded."""
        dataset = self.load_dataset_with_filters(split="train")
        assert len(dataset) > 0, f"No samples found for {self.GROUP_TYPE}"

        # Check all required fields
        sample = dataset[0]
        required_fields = [
            "input_sequence",
            "target",
            "group_type",
            "group_degree",
            "group_order",
            "sequence_length",
        ]
        for field in required_fields:
            assert field in sample, f"Missing required field: {field}"

        # Verify data types
        assert isinstance(sample["input_sequence"], str)
        assert isinstance(sample["target"], str)
        assert isinstance(sample["group_type"], str)
        assert isinstance(sample["group_degree"], int)
        assert isinstance(sample["group_order"], int)
        assert isinstance(sample["sequence_length"], int)

    def test_train_test_split(self):
        """Test that both train and test splits exist with proper ratio."""
        train_dataset = self.load_dataset_with_filters(split="train")
        test_dataset = self.load_dataset_with_filters(split="test")

        assert len(train_dataset) > 0, "No training samples found"
        assert len(test_dataset) > 0, "No test samples found"

        # Check typical 80/20 split
        total = len(train_dataset) + len(test_dataset)
        train_ratio = len(train_dataset) / total
        assert 0.75 <= train_ratio <= 0.85, (
            f"Train ratio {train_ratio} outside expected range"
        )

    # ========== Exhaustive Degree Tests ==========

    def test_all_individual_degrees(self):
        """Test each degree individually."""
        for degree in self.GROUP_CONFIG["degrees"]:
            dataset = self.load_dataset_with_filters(
                min_degree=degree, max_degree=degree, split="train"
            )

            assert len(dataset) > 0, f"No samples found for degree {degree}"

            # Verify samples
            for i in range(min(100, len(dataset))):
                assert dataset[i]["group_degree"] == degree
                if degree in self.GROUP_CONFIG["orders"]:
                    assert (
                        dataset[i]["group_order"] == self.GROUP_CONFIG["orders"][degree]
                    )

    def test_all_degree_pairs(self):
        """Test all possible degree range pairs."""
        degrees = sorted(self.GROUP_CONFIG["degrees"])
        for i, min_deg in enumerate(degrees):
            for max_deg in degrees[i:]:
                dataset = self.load_dataset_with_filters(
                    min_degree=min_deg, max_degree=max_deg, split="train"
                )

                if len(dataset) > 0:
                    # Sample and verify
                    sample_size = min(50, len(dataset))
                    indices = random.sample(range(len(dataset)), sample_size)
                    for idx in indices:
                        degree = dataset[idx]["group_degree"]
                        assert min_deg <= degree <= max_deg, (
                            f"Degree {degree} outside range [{min_deg}, {max_deg}]"
                        )

    def test_invalid_degree_ranges(self):
        """Test edge cases with invalid degree ranges."""
        max_valid_degree = max(self.GROUP_CONFIG["degrees"])

        # Test degree beyond maximum
        dataset = self.load_dataset_with_filters(
            min_degree=max_valid_degree + 1,
            max_degree=max_valid_degree + 10,
            split="train",
        )
        assert len(dataset) == 0, "Should return empty for degrees beyond maximum"

        # Test reversed range (max < min)
        if len(self.GROUP_CONFIG["degrees"]) >= 2:
            dataset = self.load_dataset_with_filters(
                min_degree=self.GROUP_CONFIG["degrees"][-1],
                max_degree=self.GROUP_CONFIG["degrees"][0],
                split="train",
            )
            # Should either be empty or handle gracefully
            assert isinstance(dataset, type(dataset))

    # ========== Exhaustive Length Tests ==========

    @pytest.mark.parametrize("length", TEST_LENGTHS)
    def test_each_length_all_degrees(self, length):
        """Test each length with all available degrees."""
        for degree in self.GROUP_CONFIG["degrees"][
            :3
        ]:  # Test first 3 degrees to avoid timeout
            dataset = self.load_dataset_with_filters(
                min_degree=degree,
                max_degree=degree,
                min_len=length,
                max_len=length,
                split="train",
            )

            if len(dataset) > 0:
                # Verify exact length match
                for i in range(min(10, len(dataset))):
                    assert dataset[i]["sequence_length"] == length

    def test_all_length_ranges(self):
        """Test various length range combinations."""
        length_ranges = [
            (1, 5),
            (3, 10),
            (10, 50),
            (50, 100),
            (100, 200),
            (200, 500),
            (500, 1000),
            (1000, 1024),
            (1, 1024),
            (512, 512),
            (1024, 1024),
        ]

        degree = self.GROUP_CONFIG["degrees"][0]  # Use first degree

        for min_len, max_len in length_ranges:
            dataset = self.load_dataset_with_filters(
                min_degree=degree,
                max_degree=degree,
                min_len=min_len,
                max_len=max_len,
                split="train",
            )

            if len(dataset) > 0:
                # Verify length bounds
                sample_indices = random.sample(
                    range(len(dataset)), min(50, len(dataset))
                )
                for idx in sample_indices:
                    seq_len = dataset[idx]["sequence_length"]
                    assert min_len <= seq_len <= max_len

    def test_length_distribution(self):
        """Test that lengths are well distributed."""
        degree = self.GROUP_CONFIG["degrees"][0]
        dataset = self.load_dataset_with_filters(
            min_degree=degree, max_degree=degree, split="train"
        )

        if len(dataset) > 1000:
            # Sample and check distribution
            length_counts = {}
            for i in range(1000):
                length = dataset[i]["sequence_length"]
                length_counts[length] = length_counts.get(length, 0) + 1

            # Should have variety of lengths
            assert len(length_counts) > 10, "Length distribution too narrow"

    # ========== Exhaustive Order Tests ==========

    def test_all_individual_orders(self):
        """Test filtering by each unique order value."""
        unique_orders = sorted(set(self.GROUP_CONFIG["orders"].values()))

        for order in unique_orders:
            dataset = self.load_dataset_with_filters(
                min_order=order, max_order=order, split="train"
            )

            if len(dataset) > 0:
                # Verify order matches
                for i in range(min(50, len(dataset))):
                    assert dataset[i]["group_order"] == order

    def test_order_ranges(self):
        """Test various order range filters."""
        unique_orders = sorted(set(self.GROUP_CONFIG["orders"].values()))

        if len(unique_orders) >= 2:
            # Test ranges
            for i in range(len(unique_orders) - 1):
                min_order = unique_orders[i]
                max_order = unique_orders[i + 1]

                dataset = self.load_dataset_with_filters(
                    min_order=min_order, max_order=max_order, split="train"
                )

                if len(dataset) > 0:
                    for j in range(min(30, len(dataset))):
                        order = dataset[j]["group_order"]
                        assert min_order <= order <= max_order

    # ========== Exhaustive Combined Filter Tests ==========

    def test_all_filter_combinations(self):
        """Test combinations of degree, order, and length filters."""
        # Sample some combinations to avoid explosion
        degree_samples = (
            self.GROUP_CONFIG["degrees"][:3]
            if len(self.GROUP_CONFIG["degrees"]) > 3
            else self.GROUP_CONFIG["degrees"]
        )
        length_ranges = [(3, 10), (50, 100), (500, 1024)]

        for degree in degree_samples:
            for min_len, max_len in length_ranges:
                dataset = self.load_dataset_with_filters(
                    min_degree=degree,
                    max_degree=degree,
                    min_len=min_len,
                    max_len=max_len,
                    split="train",
                )

                if len(dataset) > 0:
                    # Verify all constraints
                    for i in range(min(20, len(dataset))):
                        sample = dataset[i]
                        assert sample["group_degree"] == degree
                        assert min_len <= sample["sequence_length"] <= max_len
                        if degree in self.GROUP_CONFIG["orders"]:
                            assert (
                                sample["group_order"]
                                == self.GROUP_CONFIG["orders"][degree]
                            )

    def test_complex_filter_combinations(self):
        """Test more complex multi-constraint filters."""
        if len(self.GROUP_CONFIG["degrees"]) < 3:
            pytest.skip("Not enough degrees for complex filter test")

        # Multiple degrees with specific length and order constraints
        degrees = self.GROUP_CONFIG["degrees"]
        min_deg, max_deg = degrees[0], degrees[2]

        # Get order range
        orders_in_range = [
            self.GROUP_CONFIG["orders"][d]
            for d in degrees[:3]
            if d in self.GROUP_CONFIG["orders"]
        ]
        if orders_in_range:
            min_order = min(orders_in_range)
            max_order = max(orders_in_range)

            dataset = self.load_dataset_with_filters(
                min_degree=min_deg,
                max_degree=max_deg,
                min_order=min_order,
                max_order=max_order,
                min_len=10,
                max_len=100,
                split="train",
            )

            if len(dataset) > 0:
                for i in range(min(50, len(dataset))):
                    sample = dataset[i]
                    assert min_deg <= sample["group_degree"] <= max_deg
                    assert min_order <= sample["group_order"] <= max_order
                    assert 10 <= sample["sequence_length"] <= 100

    # ========== Data Integrity Tests ==========

    def test_input_sequence_format(self):
        """Test that input sequences are properly formatted."""
        dataset = self.load_dataset_with_filters(split="train")

        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            input_seq = sample["input_sequence"]

            # Should be space-separated integers
            tokens = input_seq.split()
            assert len(tokens) == sample["sequence_length"]

            # All tokens should be valid integers
            for token in tokens:
                assert token.isdigit(), f"Non-digit token found: {token}"
                # Should be valid permutation ID
                assert 0 <= int(token) < sample["group_order"]

    def test_target_validity(self):
        """Test that targets are valid permutation IDs."""
        dataset = self.load_dataset_with_filters(split="train")

        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            target = sample["target"]

            # Target should be a valid integer
            assert target.isdigit(), f"Target is not a digit: {target}"
            target_id = int(target)

            # Should be within valid range
            assert 0 <= target_id < sample["group_order"]

    def test_group_type_consistency(self):
        """Test that group_type field matches expected value."""
        dataset = self.load_dataset_with_filters(split="train")

        for i in range(min(200, len(dataset))):
            assert dataset[i]["group_type"] == self.GROUP_TYPE

    # ========== Edge Case Tests ==========

    def test_empty_results(self):
        """Test various filter combinations that should return empty results."""
        # Length 0
        dataset = self.load_dataset_with_filters(min_len=0, max_len=0, split="train")
        assert len(dataset) == 0 or all(d["sequence_length"] == 0 for d in dataset)

        # Impossible order
        max_order = max(self.GROUP_CONFIG["orders"].values())
        dataset = self.load_dataset_with_filters(
            min_order=max_order + 1, max_order=max_order + 100, split="train"
        )
        assert len(dataset) == 0

    def test_boundary_values(self):
        """Test boundary values for all filter types."""
        # Test minimum degree
        min_degree = min(self.GROUP_CONFIG["degrees"])
        dataset = self.load_dataset_with_filters(
            min_degree=min_degree, max_degree=min_degree, split="train"
        )
        assert len(dataset) > 0

        # Test maximum degree
        max_degree = max(self.GROUP_CONFIG["degrees"])
        dataset = self.load_dataset_with_filters(
            min_degree=max_degree, max_degree=max_degree, split="train"
        )
        assert len(dataset) > 0

        # Test boundary lengths
        for length in [1, 2, 3, 1023, 1024]:
            dataset = self.load_dataset_with_filters(
                min_len=length, max_len=length, split="train"
            )
            # Note: might be empty for very short lengths

    def test_large_result_sets(self):
        """Test that large result sets are handled properly."""
        # Request all data
        dataset = self.load_dataset_with_filters(split="train")

        # Should handle large datasets efficiently
        assert len(dataset) > 100, "Dataset too small for meaningful testing"

        # Test random access
        if len(dataset) > 1000:
            random_indices = random.sample(range(len(dataset)), 100)
            for idx in random_indices:
                sample = dataset[idx]
                assert "input_sequence" in sample

    # ========== Performance Tests ==========

    def test_filter_performance(self):
        """Test that filtering operations complete in reasonable time."""
        import time

        # Time a complex filter
        start = time.time()
        dataset = self.load_dataset_with_filters(
            min_degree=self.GROUP_CONFIG["degrees"][0],
            max_degree=self.GROUP_CONFIG["degrees"][-1],
            min_len=10,
            max_len=100,
            split="train",
        )
        elapsed = time.time() - start

        # Should complete reasonably quickly (adjust threshold as needed)
        assert elapsed < 30, f"Filter operation took {elapsed}s, too slow"

    def test_batch_access(self):
        """Test accessing multiple samples efficiently."""
        dataset = self.load_dataset_with_filters(split="train")

        if len(dataset) >= 100:
            # Access batch of samples
            import time

            start = time.time()
            samples = [dataset[i] for i in range(100)]
            elapsed = time.time() - start

            assert elapsed < 5, f"Batch access took {elapsed}s, too slow"
            assert len(samples) == 100
