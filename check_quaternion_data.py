#!/usr/bin/env python3
from datasets import load_from_disk

# Load Q16 dataset
dataset = load_from_disk("individual_datasets/q16_data")
train_data = dataset["train"]

# Check first few samples
for i in range(5):
    sample = train_data[i]
    input_ids = [int(x) for x in sample["input_sequence"].split()]
    target_id = int(sample["target"])

    print(f"Sample {i}:")
    print(f"  Group order: {sample['group_order']}")
    print(f"  Input IDs: {input_ids[:10]}... (showing first 10)")
    print(f"  Max input ID: {max(input_ids)}")
    print(f"  Target ID: {target_id}")
    print()
