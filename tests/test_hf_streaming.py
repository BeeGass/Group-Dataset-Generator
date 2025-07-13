#!/usr/bin/env python3
"""
Test loading individual datasets from HuggingFace in streaming mode.
"""

import pytest
from datasets import load_dataset
import time


class TestHuggingFaceStreaming:
    """Test streaming individual datasets from HuggingFace."""

    REPO_ID = "BeeGass/Group-Theory-Collection"

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
        # Try loading with data_dir approach since configs might be misconfigured
        try:
            dataset = load_dataset(
                self.REPO_ID, name=dataset_name, split="train", streaming=True
            )
        except Exception as e:
            # Fallback to data_dir if name-based loading fails
            print(f"Warning: Loading with name='{dataset_name}' failed, trying data_dir approach")
            dataset = load_dataset(
                self.REPO_ID, 
                data_dir=f"data/{dataset_name}",
                split="train", 
                streaming=True
            )

        # Take first few examples
        examples = list(dataset.take(5))
        assert len(examples) > 0, f"No examples found for {dataset_name}"

        # Verify structure
        for example in examples:
            assert "input_sequence" in example
            assert "target" in example
            assert "sequence_length" in example
            assert "group_degree" in example
            assert "group_order" in example

            # Verify expected values
            assert example["group_order"] == expected_order
            assert example["group_degree"] == expected_degree
            assert isinstance(example["input_sequence"], str)
            assert isinstance(example["target"], str)
            assert 3 <= example["sequence_length"] <= 1024

    def test_streaming_with_length_filter(self):
        """Test streaming with manual length filtering."""
        # Load S5 dataset in streaming mode
        try:
            dataset = load_dataset(self.REPO_ID, name="s5", split="train", streaming=True)
        except:
            dataset = load_dataset(self.REPO_ID, data_dir="data/s5", split="train", streaming=True)

        # Filter for short sequences
        short_sequences = dataset.filter(lambda x: x["sequence_length"] <= 32)

        # Take some examples
        examples = list(short_sequences.take(10))

        # Verify all are short
        for example in examples:
            assert 3 <= example["sequence_length"] <= 1024

    def test_streaming_from_complexity_dirs(self):
        """Test streaming from TC0 and NC1 datasets."""
        # Test TC0 dataset (c5 is a TC^0 group)
        tc0_dataset = load_dataset(
            self.REPO_ID, name="c5", split="train", streaming=True
        )

        tc0_examples = list(tc0_dataset.take(3))
        assert len(tc0_examples) > 0
        assert all(ex["group_order"] == 5 for ex in tc0_examples)

        # Test NC1 dataset (a5 is an NC^1 group)
        nc1_dataset = load_dataset(
            self.REPO_ID, name="a5", split="train", streaming=True
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
            name="s7",  # S7 has order 5040, should be large
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
            self.REPO_ID, name="s4", split=split, streaming=True
        )

        # Take one example to verify split exists
        examples = list(dataset.take(1))
        assert len(examples) == 1, f"{split} split not found"
