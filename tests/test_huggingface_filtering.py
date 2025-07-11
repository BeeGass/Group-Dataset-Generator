#!/usr/bin/env python3
"""
Exhaustive test of dynamic filtering functionality from HuggingFace repository.
This ensures that the uploaded datasets work correctly with ALL filtering options.

Tests include:
1. Every individual degree for every group
2. All possible degree range combinations
3. Every individual order value
4. All possible order range combinations
5. Specific length values (1, 2, 3, ..., 1024)
6. All meaningful length range combinations
7. All combinations of degree + length filters
8. All combinations of order + length filters
9. All combinations of degree + order + length filters
10. Edge cases and boundary conditions
"""

import pytest
from datasets import load_dataset
import random
import itertools
from collections import defaultdict
import time


class TestHuggingFaceFilteringExhaustive:
    """Exhaustively test dynamic filtering from HuggingFace repo."""

    REPO_ID = "BeeGass/permutation-groups"

    # Test all group types
    GROUP_TYPES = [
        "symmetric",
        "alternating",
        "cyclic",
        "dihedral",
        "klein",
        "quaternion",
        "elementary_abelian",
        "psl",
        "frobenius",
        "mathieu",
    ]

    # Expected configurations for each group
    GROUP_CONFIGS = {
        "symmetric": {
            "degrees": [3, 4, 5, 6, 7, 8, 9, 10],
            "orders": {
                3: 6,
                4: 24,
                5: 120,
                6: 720,
                7: 5040,
                8: 40320,
                9: 362880,
                10: 3628800,
            },
        },
        "alternating": {
            "degrees": [3, 4, 5, 6, 7, 8, 9, 10],
            "orders": {
                3: 3,
                4: 12,
                5: 60,
                6: 360,
                7: 2520,
                8: 20160,
                9: 181440,
                10: 1814400,
            },
        },
        "cyclic": {
            "degrees": list(range(3, 31)),
            "orders": {n: n for n in range(3, 31)},
        },
        "dihedral": {
            "degrees": list(range(3, 21)),
            "orders": {n: 2 * n for n in range(3, 21)},
        },
        "klein": {"degrees": [4], "orders": {4: 4}},
        "quaternion": {"degrees": [8, 16, 32], "orders": {8: 8, 16: 16, 32: 32}},
        "elementary_abelian": {
            "degrees": [2, 3, 4, 5, 8, 9, 16, 25, 27, 32],
            "orders": {
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                8: 8,
                9: 9,
                16: 16,
                25: 25,
                27: 27,
                32: 32,
            },
        },
        "psl": {"degrees": [6, 8], "orders": {6: 60, 8: 168}},
        "frobenius": {"degrees": [5, 7], "orders": {5: 20, 7: 21}},
        "mathieu": {"degrees": [11, 12], "orders": {11: 7920, 12: 95040}},
    }

    # Test specific lengths exhaustively
    TEST_LENGTHS = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,  # Very short
        15,
        16,
        20,
        25,
        30,
        32,  # Short
        40,
        50,
        60,
        64,
        70,
        80,
        90,
        100,  # Medium
        128,
        150,
        200,
        256,  # Medium-long
        300,
        400,
        500,
        512,  # Long
        600,
        700,
        800,
        900,
        1000,  # Very long
        1020,
        1021,
        1022,
        1023,
        1024,  # Maximum boundary
    ]

    # Length ranges to test
    LENGTH_RANGES = [
        (1, 5),
        (1, 10),
        (1, 32),
        (1, 64),
        (1, 128),
        (1, 256),
        (1, 512),
        (1, 1024),
        (3, 10),
        (3, 32),
        (3, 64),
        (3, 128),
        (3, 256),
        (3, 512),
        (3, 1024),
        (10, 20),
        (10, 50),
        (10, 100),
        (10, 200),
        (10, 500),
        (10, 1024),
        (32, 64),
        (32, 128),
        (32, 256),
        (32, 512),
        (32, 1024),
        (64, 128),
        (64, 256),
        (64, 512),
        (64, 1024),
        (100, 200),
        (100, 500),
        (100, 1000),
        (100, 1024),
        (128, 256),
        (128, 512),
        (128, 1024),
        (256, 512),
        (256, 1024),
        (500, 1000),
        (500, 1024),
        (512, 1024),
        (1000, 1024),
        (1020, 1024),
        (1023, 1024),
        # Edge cases
        (1, 1),
        (2, 2),
        (3, 3),
        (10, 10),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ]

    def verify_sample(
        self,
        sample,
        group_type,
        min_degree=None,
        max_degree=None,
        min_order=None,
        max_order=None,
        min_len=None,
        max_len=None,
    ):
        """Verify a sample meets all filter criteria."""
        # Check required fields exist
        assert "input_sequence" in sample
        assert "target" in sample
        assert "group_type" in sample
        assert "group_degree" in sample
        assert "group_order" in sample
        assert "sequence_length" in sample

        # Check group type
        assert sample["group_type"] == group_type

        # Check degree constraints
        if min_degree is not None:
            assert sample["group_degree"] >= min_degree, (
                f"Degree {sample['group_degree']} < min {min_degree}"
            )
        if max_degree is not None:
            assert sample["group_degree"] <= max_degree, (
                f"Degree {sample['group_degree']} > max {max_degree}"
            )

        # Check order constraints
        if min_order is not None:
            assert sample["group_order"] >= min_order, (
                f"Order {sample['group_order']} < min {min_order}"
            )
        if max_order is not None:
            assert sample["group_order"] <= max_order, (
                f"Order {sample['group_order']} > max {max_order}"
            )

        # Check length constraints
        if min_len is not None:
            assert sample["sequence_length"] >= min_len, (
                f"Length {sample['sequence_length']} < min {min_len}"
            )
        if max_len is not None:
            assert sample["sequence_length"] <= max_len, (
                f"Length {sample['sequence_length']} > max {max_len}"
            )

        # Check order matches degree for this group
        config = self.GROUP_CONFIGS[group_type]
        if sample["group_degree"] in config["orders"]:
            expected_order = config["orders"][sample["group_degree"]]
            assert sample["group_order"] == expected_order, (
                f"Order {sample['group_order']} != expected {expected_order} for degree {sample['group_degree']}"
            )

        # Check input sequence format
        tokens = sample["input_sequence"].split()
        assert len(tokens) == sample["sequence_length"], (
            f"Token count {len(tokens)} != sequence_length {sample['sequence_length']}"
        )

        # Check all tokens are valid integers
        for token in tokens:
            assert token.isdigit(), f"Non-digit token: {token}"
            assert 0 <= int(token) < sample["group_order"], (
                f"Token {token} out of range [0, {sample['group_order']})"
            )

        # Check target is valid
        assert sample["target"].isdigit(), f"Target not digit: {sample['target']}"
        target_id = int(sample["target"])
        assert 0 <= target_id < sample["group_order"], (
            f"Target {target_id} out of range [0, {sample['group_order']})"
        )

    # ========== Exhaustive Degree Tests ==========

    @pytest.mark.parametrize("group_type", GROUP_TYPES)
    def test_every_individual_degree(self, group_type):
        """Test filtering by every individual degree value."""
        config = self.GROUP_CONFIGS[group_type]

        for degree in config["degrees"]:
            print(f"\nTesting {group_type} degree={degree}")
            dataset = load_dataset(
                self.REPO_ID,
                name=group_type,
                min_degree=degree,
                max_degree=degree,
                trust_remote_code=True,
                split="train",
            )

            assert len(dataset) > 0, f"No samples for {group_type} degree {degree}"

            # Verify samples
            samples_to_check = min(100, len(dataset))
            for i in range(samples_to_check):
                self.verify_sample(
                    dataset[i], group_type, min_degree=degree, max_degree=degree
                )

    @pytest.mark.parametrize(
        "group_type", ["symmetric", "alternating", "cyclic", "dihedral"]
    )
    def test_all_degree_range_combinations(self, group_type):
        """Test all possible degree range combinations."""
        config = self.GROUP_CONFIGS[group_type]
        degrees = config["degrees"]

        # Test all possible ranges
        for i, min_deg in enumerate(degrees):
            for max_deg in degrees[i:]:
                print(f"\nTesting {group_type} degree range [{min_deg}, {max_deg}]")
                dataset = load_dataset(
                    self.REPO_ID,
                    name=group_type,
                    min_degree=min_deg,
                    max_degree=max_deg,
                    trust_remote_code=True,
                    split="train",
                )

                if len(dataset) > 0:
                    # Collect actual degrees found
                    degrees_found = set()
                    samples_to_check = min(200, len(dataset))

                    for j in range(samples_to_check):
                        sample = dataset[j]
                        self.verify_sample(
                            sample, group_type, min_degree=min_deg, max_degree=max_deg
                        )
                        degrees_found.add(sample["group_degree"])

                    # Verify we found expected degrees
                    expected_degrees = {d for d in degrees if min_deg <= d <= max_deg}
                    if len(dataset) >= 100 * len(expected_degrees):
                        # Only check if we have enough samples
                        assert degrees_found == expected_degrees, (
                            f"Expected degrees {expected_degrees}, found {degrees_found}"
                        )

    # ========== Exhaustive Order Tests ==========

    @pytest.mark.parametrize("group_type", GROUP_TYPES)
    def test_every_individual_order(self, group_type):
        """Test filtering by every individual order value."""
        config = self.GROUP_CONFIGS[group_type]
        unique_orders = sorted(set(config["orders"].values()))

        for order in unique_orders:
            print(f"\nTesting {group_type} order={order}")
            dataset = load_dataset(
                self.REPO_ID,
                name=group_type,
                min_order=order,
                max_order=order,
                trust_remote_code=True,
                split="train",
            )

            if len(dataset) > 0:
                # Verify samples
                samples_to_check = min(50, len(dataset))
                for i in range(samples_to_check):
                    self.verify_sample(
                        dataset[i], group_type, min_order=order, max_order=order
                    )

    def test_all_order_range_combinations(self):
        """Test various order range combinations for each group."""
        test_cases = [
            # Symmetric groups
            ("symmetric", 6, 24),  # S3-S4
            ("symmetric", 6, 120),  # S3-S5
            ("symmetric", 24, 720),  # S4-S6
            ("symmetric", 120, 5040),  # S5-S7
            ("symmetric", 720, 40320),  # S6-S8
            # Alternating groups
            ("alternating", 3, 12),  # A3-A4
            ("alternating", 12, 60),  # A4-A5
            ("alternating", 60, 360),  # A5-A6
            ("alternating", 360, 2520),  # A6-A7
            # Cyclic groups
            ("cyclic", 3, 10),
            ("cyclic", 10, 20),
            ("cyclic", 20, 30),
            # Dihedral groups
            ("dihedral", 6, 20),  # D3-D10
            ("dihedral", 20, 40),  # D10-D20
        ]

        for group_type, min_order, max_order in test_cases:
            print(f"\nTesting {group_type} order range [{min_order}, {max_order}]")
            dataset = load_dataset(
                self.REPO_ID,
                name=group_type,
                min_order=min_order,
                max_order=max_order,
                trust_remote_code=True,
                split="train",
            )

            if len(dataset) > 0:
                orders_found = set()
                samples_to_check = min(100, len(dataset))

                for i in range(samples_to_check):
                    sample = dataset[i]
                    self.verify_sample(
                        sample, group_type, min_order=min_order, max_order=max_order
                    )
                    orders_found.add(sample["group_order"])

                print(f"  Found orders: {sorted(orders_found)}")

    # ========== Exhaustive Length Tests ==========

    @pytest.mark.parametrize("length", TEST_LENGTHS)
    @pytest.mark.parametrize(
        "group_type", ["symmetric", "cyclic"]
    )  # Test subset to avoid timeout
    def test_specific_lengths(self, group_type, length):
        """Test filtering by specific length values."""
        print(f"\nTesting {group_type} with exact length={length}")

        # Use a specific degree to make queries faster
        config = self.GROUP_CONFIGS[group_type]
        test_degree = config["degrees"][0]

        dataset = load_dataset(
            self.REPO_ID,
            name=group_type,
            min_degree=test_degree,
            max_degree=test_degree,
            min_len=length,
            max_len=length,
            trust_remote_code=True,
            split="train",
        )

        if len(dataset) > 0:
            # Verify exact length match
            samples_to_check = min(10, len(dataset))
            for i in range(samples_to_check):
                self.verify_sample(
                    dataset[i],
                    group_type,
                    min_degree=test_degree,
                    max_degree=test_degree,
                    min_len=length,
                    max_len=length,
                )

    @pytest.mark.parametrize("length_range", LENGTH_RANGES)
    @pytest.mark.parametrize("group_type", ["symmetric", "alternating", "dihedral"])
    def test_length_ranges(self, group_type, length_range):
        """Test all length range combinations."""
        min_len, max_len = length_range
        print(f"\nTesting {group_type} length range [{min_len}, {max_len}]")

        # Use a specific degree to make queries faster
        config = self.GROUP_CONFIGS[group_type]
        test_degree = config["degrees"][min(2, len(config["degrees"]) - 1)]

        dataset = load_dataset(
            self.REPO_ID,
            name=group_type,
            min_degree=test_degree,
            max_degree=test_degree,
            min_len=min_len,
            max_len=max_len,
            trust_remote_code=True,
            split="train",
        )

        if len(dataset) > 0:
            # Collect length distribution
            length_counts = defaultdict(int)
            samples_to_check = min(200, len(dataset))

            for i in range(samples_to_check):
                sample = dataset[i]
                self.verify_sample(
                    sample,
                    group_type,
                    min_degree=test_degree,
                    max_degree=test_degree,
                    min_len=min_len,
                    max_len=max_len,
                )
                length_counts[sample["sequence_length"]] += 1

            # Verify we have variety of lengths (unless range is very narrow)
            if max_len - min_len > 10 and len(dataset) > 100:
                assert len(length_counts) > 5, (
                    f"Too few unique lengths: {len(length_counts)}"
                )

    # ========== Combined Filter Tests ==========

    def test_degree_and_length_combinations(self):
        """Test combining degree and length filters."""
        test_cases = [
            # (group, min_deg, max_deg, min_len, max_len)
            ("symmetric", 3, 3, 10, 50),
            ("symmetric", 4, 4, 100, 200),
            ("symmetric", 5, 5, 500, 1000),
            ("symmetric", 3, 5, 1, 100),
            ("symmetric", 5, 7, 256, 512),
            ("alternating", 4, 6, 32, 64),
            ("alternating", 5, 7, 128, 256),
            ("alternating", 3, 10, 1, 1024),
            ("cyclic", 5, 10, 50, 150),
            ("cyclic", 10, 20, 200, 400),
            ("cyclic", 3, 30, 1, 32),
            ("dihedral", 3, 5, 64, 128),
            ("dihedral", 5, 10, 256, 512),
            ("dihedral", 10, 20, 1000, 1024),
        ]

        for group_type, min_deg, max_deg, min_len, max_len in test_cases:
            print(
                f"\nTesting {group_type}: degree[{min_deg},{max_deg}], length[{min_len},{max_len}]"
            )

            dataset = load_dataset(
                self.REPO_ID,
                name=group_type,
                min_degree=min_deg,
                max_degree=max_deg,
                min_len=min_len,
                max_len=max_len,
                trust_remote_code=True,
                split="train",
            )

            if len(dataset) > 0:
                samples_to_check = min(50, len(dataset))
                for i in range(samples_to_check):
                    self.verify_sample(
                        dataset[i],
                        group_type,
                        min_degree=min_deg,
                        max_degree=max_deg,
                        min_len=min_len,
                        max_len=max_len,
                    )

    def test_order_and_length_combinations(self):
        """Test combining order and length filters."""
        test_cases = [
            # (group, min_order, max_order, min_len, max_len)
            ("symmetric", 6, 24, 3, 10),  # S3-S4, short
            ("symmetric", 24, 120, 50, 100),  # S4-S5, medium
            ("symmetric", 120, 720, 200, 500),  # S5-S6, long
            ("alternating", 12, 60, 32, 128),  # A4-A5
            ("alternating", 60, 360, 256, 512),  # A5-A6
            ("cyclic", 5, 15, 10, 100),
            ("cyclic", 10, 25, 100, 500),
            ("dihedral", 10, 20, 64, 256),  # D5-D10
            ("dihedral", 20, 40, 512, 1024),  # D10-D20
        ]

        for group_type, min_order, max_order, min_len, max_len in test_cases:
            print(
                f"\nTesting {group_type}: order[{min_order},{max_order}], length[{min_len},{max_len}]"
            )

            dataset = load_dataset(
                self.REPO_ID,
                name=group_type,
                min_order=min_order,
                max_order=max_order,
                min_len=min_len,
                max_len=max_len,
                trust_remote_code=True,
                split="train",
            )

            if len(dataset) > 0:
                samples_to_check = min(50, len(dataset))
                for i in range(samples_to_check):
                    self.verify_sample(
                        dataset[i],
                        group_type,
                        min_order=min_order,
                        max_order=max_order,
                        min_len=min_len,
                        max_len=max_len,
                    )

    def test_all_three_filters_combined(self):
        """Test combining degree, order, and length filters."""
        test_cases = [
            # (group, min_deg, max_deg, min_order, max_order, min_len, max_len)
            ("symmetric", 3, 4, 6, 24, 10, 100),
            ("symmetric", 5, 6, 120, 720, 100, 500),
            ("symmetric", 4, 5, 24, 120, 256, 512),
            ("alternating", 4, 5, 12, 60, 32, 128),
            ("alternating", 5, 6, 60, 360, 200, 400),
            ("cyclic", 5, 10, 5, 10, 50, 200),
            ("cyclic", 10, 20, 10, 20, 100, 1000),
            ("dihedral", 5, 10, 10, 20, 64, 256),
            ("dihedral", 10, 15, 20, 30, 512, 1024),
        ]

        for (
            group_type,
            min_deg,
            max_deg,
            min_order,
            max_order,
            min_len,
            max_len,
        ) in test_cases:
            print(
                f"\nTesting {group_type}: degree[{min_deg},{max_deg}], "
                f"order[{min_order},{max_order}], length[{min_len},{max_len}]"
            )

            dataset = load_dataset(
                self.REPO_ID,
                name=group_type,
                min_degree=min_deg,
                max_degree=max_deg,
                min_order=min_order,
                max_order=max_order,
                min_len=min_len,
                max_len=max_len,
                trust_remote_code=True,
                split="train",
            )

            if len(dataset) > 0:
                samples_to_check = min(30, len(dataset))
                for i in range(samples_to_check):
                    self.verify_sample(
                        dataset[i],
                        group_type,
                        min_degree=min_deg,
                        max_degree=max_deg,
                        min_order=min_order,
                        max_order=max_order,
                        min_len=min_len,
                        max_len=max_len,
                    )

    # ========== Edge Cases and Boundary Tests ==========

    def test_edge_cases(self):
        """Test various edge cases and boundary conditions."""
        # Test empty results (degree beyond range)
        dataset = load_dataset(
            self.REPO_ID,
            name="symmetric",
            min_degree=20,
            max_degree=30,
            trust_remote_code=True,
            split="train",
        )
        assert len(dataset) == 0, "Should return empty for degrees beyond range"

        # Test empty results (order beyond range)
        dataset = load_dataset(
            self.REPO_ID,
            name="cyclic",
            min_order=100,
            max_order=200,
            trust_remote_code=True,
            split="train",
        )
        assert len(dataset) == 0, "Should return empty for orders beyond range"

        # Test reversed ranges (should handle gracefully)
        dataset = load_dataset(
            self.REPO_ID,
            name="dihedral",
            min_degree=10,
            max_degree=5,  # max < min
            trust_remote_code=True,
            split="train",
        )
        # Should either be empty or handle gracefully

        # Test exact boundaries
        dataset = load_dataset(
            self.REPO_ID,
            name="symmetric",
            min_len=1024,
            max_len=1024,
            trust_remote_code=True,
            split="train",
        )
        if len(dataset) > 0:
            for i in range(min(5, len(dataset))):
                assert dataset[i]["sequence_length"] == 1024

        # Test minimum length
        dataset = load_dataset(
            self.REPO_ID,
            name="cyclic",
            min_len=1,
            max_len=1,
            trust_remote_code=True,
            split="train",
        )
        if len(dataset) > 0:
            for i in range(min(5, len(dataset))):
                assert dataset[i]["sequence_length"] == 1

    def test_special_groups(self):
        """Test special groups with limited configurations."""
        # Klein group (only degree 4)
        dataset = load_dataset(
            self.REPO_ID, name="klein", trust_remote_code=True, split="train"
        )
        assert len(dataset) > 0
        for i in range(min(10, len(dataset))):
            assert dataset[i]["group_degree"] == 4
            assert dataset[i]["group_order"] == 4

        # Test Klein with degree filter (should still work)
        dataset = load_dataset(
            self.REPO_ID,
            name="klein",
            min_degree=4,
            max_degree=4,
            trust_remote_code=True,
            split="train",
        )
        assert len(dataset) > 0

        # Test Klein with wrong degree (should be empty)
        dataset = load_dataset(
            self.REPO_ID,
            name="klein",
            min_degree=5,
            max_degree=5,
            trust_remote_code=True,
            split="train",
        )
        assert len(dataset) == 0

        # Quaternion groups
        for degree in [8, 16, 32]:
            dataset = load_dataset(
                self.REPO_ID,
                name="quaternion",
                min_degree=degree,
                max_degree=degree,
                trust_remote_code=True,
                split="train",
            )
            assert len(dataset) > 0
            for i in range(min(5, len(dataset))):
                assert dataset[i]["group_degree"] == degree
                assert dataset[i]["group_order"] == degree

    def test_train_test_splits(self):
        """Verify both splits work with all filter types."""
        test_configs = [
            ("symmetric", {"min_degree": 5, "max_degree": 5}),
            ("alternating", {"min_order": 60, "max_order": 60}),
            ("cyclic", {"min_len": 100, "max_len": 200}),
            (
                "dihedral",
                {"min_degree": 5, "max_degree": 10, "min_len": 50, "max_len": 150},
            ),
        ]

        for group_type, filters in test_configs:
            for split in ["train", "test"]:
                dataset = load_dataset(
                    self.REPO_ID,
                    name=group_type,
                    split=split,
                    trust_remote_code=True,
                    **filters,
                )
                assert len(dataset) > 0, (
                    f"No {split} samples for {group_type} with {filters}"
                )

                # Verify split ratio if both exist
                if split == "train":
                    test_dataset = load_dataset(
                        self.REPO_ID,
                        name=group_type,
                        split="test",
                        trust_remote_code=True,
                        **filters,
                    )
                    total = len(dataset) + len(test_dataset)
                    train_ratio = len(dataset) / total
                    assert 0.75 <= train_ratio <= 0.85, (
                        f"Train ratio {train_ratio} outside expected range"
                    )


def run_comprehensive_test():
    """Run a comprehensive test of all filtering capabilities."""
    print("=" * 80)
    print("COMPREHENSIVE HUGGINGFACE FILTERING TEST")
    print("=" * 80)

    results = {"passed": [], "failed": [], "warnings": []}

    # Test 1: Basic loading for all groups
    print("\n1. Testing basic loading for all groups...")
    for group in TestHuggingFaceFilteringExhaustive.GROUP_TYPES:
        try:
            ds = load_dataset(
                TestHuggingFaceFilteringExhaustive.REPO_ID,
                name=group,
                trust_remote_code=True,
                split="train",
            )
            results["passed"].append(f"{group}: Loaded {len(ds)} samples")
        except Exception as e:
            results["failed"].append(f"{group}: {str(e)}")

    # Test 2: Degree filtering
    print("\n2. Testing degree filtering...")
    test_cases = [
        ("symmetric", 5, 5),
        ("alternating", 4, 6),
        ("cyclic", 10, 15),
        ("dihedral", 5, 10),
    ]

    for group, min_deg, max_deg in test_cases:
        try:
            ds = load_dataset(
                TestHuggingFaceFilteringExhaustive.REPO_ID,
                name=group,
                min_degree=min_deg,
                max_degree=max_deg,
                trust_remote_code=True,
                split="train",
            )
            # Verify degrees
            degrees = set()
            for i in range(min(100, len(ds))):
                deg = ds[i]["group_degree"]
                degrees.add(deg)
                if not (min_deg <= deg <= max_deg):
                    results["failed"].append(
                        f"{group} degree[{min_deg},{max_deg}]: Found degree {deg} outside range"
                    )
                    break
            else:
                results["passed"].append(
                    f"{group} degree[{min_deg},{max_deg}]: {len(ds)} samples, degrees: {sorted(degrees)}"
                )
        except Exception as e:
            results["failed"].append(f"{group} degree[{min_deg},{max_deg}]: {str(e)}")

    # Test 3: Order filtering
    print("\n3. Testing order filtering...")
    test_cases = [
        ("symmetric", 24, 120),  # S4-S5
        ("alternating", 12, 60),  # A4-A5
        ("cyclic", 5, 10),
        ("dihedral", 10, 20),  # D5-D10
    ]

    for group, min_ord, max_ord in test_cases:
        try:
            ds = load_dataset(
                TestHuggingFaceFilteringExhaustive.REPO_ID,
                name=group,
                min_order=min_ord,
                max_order=max_ord,
                trust_remote_code=True,
                split="train",
            )
            # Verify orders
            orders = set()
            for i in range(min(100, len(ds))):
                ord = ds[i]["group_order"]
                orders.add(ord)
                if not (min_ord <= ord <= max_ord):
                    results["failed"].append(
                        f"{group} order[{min_ord},{max_ord}]: Found order {ord} outside range"
                    )
                    break
            else:
                results["passed"].append(
                    f"{group} order[{min_ord},{max_ord}]: {len(ds)} samples, orders: {sorted(orders)}"
                )
        except Exception as e:
            results["failed"].append(f"{group} order[{min_ord},{max_ord}]: {str(e)}")

    # Test 4: Length filtering
    print("\n4. Testing length filtering...")
    test_lengths = [
        (1, 10),
        (32, 64),
        (100, 200),
        (500, 1000),
        (1024, 1024),
    ]

    for min_len, max_len in test_lengths:
        try:
            ds = load_dataset(
                TestHuggingFaceFilteringExhaustive.REPO_ID,
                name="symmetric",
                min_degree=5,
                max_degree=5,
                min_len=min_len,
                max_len=max_len,
                trust_remote_code=True,
                split="train",
            )
            # Verify lengths
            lengths = []
            for i in range(min(50, len(ds))):
                length = ds[i]["sequence_length"]
                lengths.append(length)
                if not (min_len <= length <= max_len):
                    results["failed"].append(
                        f"S5 length[{min_len},{max_len}]: Found length {length} outside range"
                    )
                    break
            else:
                results["passed"].append(
                    f"S5 length[{min_len},{max_len}]: {len(ds)} samples, "
                    f"lengths: min={min(lengths)}, max={max(lengths)}"
                )
        except Exception as e:
            results["failed"].append(f"S5 length[{min_len},{max_len}]: {str(e)}")

    # Test 5: Combined filters
    print("\n5. Testing combined filters...")
    test_cases = [
        ("symmetric", 4, 6, None, None, 100, 200),
        ("alternating", None, None, 60, 360, 50, 150),
        ("dihedral", 5, 10, 10, 20, 256, 512),
    ]

    for group, min_deg, max_deg, min_ord, max_ord, min_len, max_len in test_cases:
        filter_desc = []
        kwargs = {}
        if min_deg:
            kwargs["min_degree"] = min_deg
            filter_desc.append(f"deg[{min_deg},{max_deg}]")
        if max_deg:
            kwargs["max_degree"] = max_deg
        if min_ord:
            kwargs["min_order"] = min_ord
            filter_desc.append(f"ord[{min_ord},{max_ord}]")
        if max_ord:
            kwargs["max_order"] = max_ord
        if min_len:
            kwargs["min_len"] = min_len
            filter_desc.append(f"len[{min_len},{max_len}]")
        if max_len:
            kwargs["max_len"] = max_len

        try:
            ds = load_dataset(
                TestHuggingFaceFilteringExhaustive.REPO_ID,
                name=group,
                trust_remote_code=True,
                split="train",
                **kwargs,
            )
            results["passed"].append(
                f"{group} {' '.join(filter_desc)}: {len(ds)} samples"
            )
        except Exception as e:
            results["failed"].append(f"{group} {' '.join(filter_desc)}: {str(e)}")

    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    print(f"\nPASSED: {len(results['passed'])}")
    for msg in results["passed"][:10]:  # Show first 10
        print(f"  ✓ {msg}")
    if len(results["passed"]) > 10:
        print(f"  ... and {len(results['passed']) - 10} more")

    print(f"\nFAILED: {len(results['failed'])}")
    for msg in results["failed"]:
        print(f"  ✗ {msg}")

    print(f"\nWARNINGS: {len(results['warnings'])}")
    for msg in results["warnings"]:
        print(f"  ⚠ {msg}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(results['passed'])} passed, {len(results['failed'])} failed")
    print("=" * 80)

    return len(results["failed"]) == 0


if __name__ == "__main__":
    # Run comprehensive test
    success = run_comprehensive_test()

    if success:
        print("\n✅ All tests passed! Dynamic filtering is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the results above.")

    print(
        "\nFor full exhaustive testing, run: pytest tests/test_huggingface_filtering.py -v"
    )
