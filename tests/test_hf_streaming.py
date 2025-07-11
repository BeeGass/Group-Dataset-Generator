#!/usr/bin/env python3
"""
Test loading individual datasets from HuggingFace in streaming mode.
"""

import pytest
from datasets import load_dataset
import time


class TestHuggingFaceStreaming:
    """Test streaming individual datasets from HuggingFace."""

    REPO_ID = "BeeGass/permutation-groups"

    # Sample of datasets to test (one from each complexity class)
    TC0_EXAMPLES = [
        ("s3", 6, 3),  # Symmetric solvable
        ("a4", 12, 4),  # Alternating solvable
        ("c10", 10, 10),  # Cyclic
        ("d6", 12, 6),  # Dihedral
        ("v4", 4, 4),  # Klein
        ("q8", 8, 8),  # Quaternion
        ("z2_3", 8, 8),  # Elementary abelian
        ("f20", 20, 5),  # Frobenius
    ]

    NC1_EXAMPLES = [
        ("s5", 120, 5),  # Symmetric non-solvable
        ("a5", 60, 5),  # Alternating non-solvable
        ("psl2_5", 60, 6),  # PSL group
        ("m11", 7920, 11),  # Mathieu group
    ]

    @pytest.mark.parametrize(
        "dataset_name,expected_order,expected_degree", TC0_EXAMPLES + NC1_EXAMPLES
    )
    def test_streaming_individual_dataset(
        self, dataset_name, expected_order, expected_degree
    ):
        """Test streaming a specific individual dataset."""
        # Load dataset in streaming mode
        dataset = load_dataset(
            self.REPO_ID, data_dir=f"data/{dataset_name}", split="train", streaming=True
        )

        # Take first few examples
        examples = list(dataset.take(5))
        assert len(examples) > 0, f"No examples found for {dataset_name}"

        # Verify structure
        for example in examples:
            assert "input_sequence" in example
            assert "target" in example
            assert "length" in example
            assert "group_degree" in example
            assert "group_order" in example

            # Verify expected values
            assert example["group_order"] == expected_order
            assert example["group_degree"] == expected_degree
            assert isinstance(example["input_sequence"], list)
            assert isinstance(example["target"], int)
            assert example["length"] == len(example["input_sequence"])

    def test_streaming_with_length_filter(self):
        """Test streaming with manual length filtering."""
        # Load S5 dataset in streaming mode
        dataset = load_dataset(
            self.REPO_ID, data_dir="data/s5", split="train", streaming=True
        )

        # Filter for short sequences
        short_sequences = dataset.filter(lambda x: x["length"] <= 32)

        # Take some examples
        examples = list(short_sequences.take(10))

        # Verify all are short
        for example in examples:
            assert example["length"] <= 32

    def test_streaming_from_complexity_dirs(self):
        """Test streaming from TC0 and NC1 directories."""
        # Test TC0 dataset
        tc0_dataset = load_dataset(
            self.REPO_ID, data_dir="TC0/c5", split="train", streaming=True
        )

        tc0_examples = list(tc0_dataset.take(3))
        assert len(tc0_examples) > 0
        assert all(ex["group_order"] == 5 for ex in tc0_examples)

        # Test NC1 dataset
        nc1_dataset = load_dataset(
            self.REPO_ID, data_dir="NC1/a5", split="train", streaming=True
        )

        nc1_examples = list(nc1_dataset.take(3))
        assert len(nc1_examples) > 0
        assert all(ex["group_order"] == 60 for ex in nc1_examples)

    def test_streaming_performance(self):
        """Test that streaming is efficient for large datasets."""
        start_time = time.time()

        # Load a large dataset in streaming mode
        dataset = load_dataset(
            self.REPO_ID,
            data_dir="data/s7",  # S7 has order 5040, should be large
            split="train",
            streaming=True,
        )

        # Take only first 100 examples
        examples = list(dataset.take(100))

        elapsed_time = time.time() - start_time

        # Should be fast (under 30 seconds even with network latency)
        assert elapsed_time < 30, f"Streaming took too long: {elapsed_time:.2f}s"
        assert len(examples) == 100

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_both_splits_available(self, split):
        """Test that both train and test splits are available."""
        dataset = load_dataset(
            self.REPO_ID, data_dir="data/s4", split=split, streaming=True
        )

        # Take one example to verify split exists
        examples = list(dataset.take(1))
        assert len(examples) == 1, f"{split} split not found"
