#!/usr/bin/env python3
"""Test loading datasets directly without the loading script."""

from datasets import load_dataset
import itertools

print("Testing direct dataset loading without custom script...")

# Try loading symmetric dataset directly with streaming
try:
    # Method 1: Direct data files with streaming
    dataset = load_dataset(
        "BeeGass/permutation-groups",
        data_files={
            "train": "data/symmetric_superset/train/data-*.arrow",
            "test": "data/symmetric_superset/test/data-*.arrow",
        },
        streaming=True,
    )
    print("✓ Method 1 worked (streaming)")
    # Take first few samples
    train_samples = list(itertools.islice(dataset["train"], 5))
    print(f"  First sample: {train_samples[0]}")
    print(f"  Columns: {list(train_samples[0].keys())}")
except Exception as e:
    print(f"✗ Method 1 failed: {e}")

# Try method 2: data_dir with streaming
try:
    dataset = load_dataset(
        "BeeGass/permutation-groups", data_dir="data/symmetric_superset", streaming=True
    )
    print("✓ Method 2 worked (streaming)")
    train_sample = next(iter(dataset["train"]))
    print(f"  First sample group: {train_sample['group_type']}")
except Exception as e:
    print(f"✗ Method 2 failed: {e}")

# Try method 3: Using arrow format directly with streaming
try:
    dataset = load_dataset(
        "arrow",
        data_files={
            "train": "hf://datasets/BeeGass/permutation-groups/data/symmetric_superset/train/data-*.arrow",
            "test": "hf://datasets/BeeGass/permutation-groups/data/symmetric_superset/test/data-*.arrow",
        },
        streaming=True,
    )
    print("✓ Method 3 worked (streaming)")

    # Test filtering with streaming
    filtered = dataset["train"].filter(lambda x: x["group_degree"] == 5)
    filtered_samples = list(itertools.islice(filtered, 5))
    print(f"  Found {len(filtered_samples)} samples with degree 5 (showing first 5)")
    if filtered_samples:
        print(
            f"  First filtered sample: degree={filtered_samples[0]['group_degree']}, order={filtered_samples[0]['group_order']}"
        )
except Exception as e:
    print(f"✗ Method 3 failed: {e}")

# Try method 4: Loading without script detection
try:
    import os

    # Temporarily set environment variable to bypass script detection
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "0"

    dataset = load_dataset(
        "BeeGass/permutation-groups",
        revision="main",
        data_files={
            "train": "data/symmetric_superset/train/data-*.arrow",
            "test": "data/symmetric_superset/test/data-*.arrow",
        },
        streaming=True,
    )
    print("✓ Method 4 worked (with env var)")
except Exception as e:
    print(f"✗ Method 4 failed: {e}")
