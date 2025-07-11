#!/usr/bin/env python3
"""
Minimum working examples for loading permutation groups from HuggingFace in streaming mode.
This demonstrates efficient loading without downloading the entire dataset.
"""

from datasets import load_dataset

def example_single_group_streaming():
    """Load a single group dataset in streaming mode."""
    print("=== Example 1: Single Group Streaming ===")
    
    # Load S5 (symmetric group) in streaming mode
    dataset = load_dataset(
        "BeeGass/Group-Theory-Collection", 
        data_dir="data/s5",
        streaming=True
    )
    
    # Access the train split
    train_stream = dataset["train"]
    
    # Process first 5 examples
    print("\nFirst 5 examples from S5:")
    for i, example in enumerate(train_stream):
        if i >= 5:
            break
        print(f"\nExample {i+1}:")
        print(f"  Input: {example['input_sequence']}")
        print(f"  Target: {example['target']}")
        print(f"  Length: {example['sequence_length']}")
        print(f"  Group: {example['group_type']} (order {example['group_order']})")


def example_filter_by_length_streaming():
    """Filter examples by sequence length in streaming mode."""
    print("\n=== Example 2: Filter by Length in Streaming ===")
    
    # Load C10 (cyclic group) in streaming mode
    dataset = load_dataset(
        "BeeGass/Group-Theory-Collection", 
        data_dir="data/c10",
        streaming=True
    )
    
    # Filter for sequences of length 16
    filtered_stream = dataset["train"].filter(
        lambda x: x["sequence_length"] == 16
    )
    
    print("\nExamples with length 16 from C10:")
    for i, example in enumerate(filtered_stream):
        if i >= 3:
            break
        print(f"\nExample {i+1}:")
        print(f"  Input: {example['input_sequence']}")
        print(f"  Target: {example['target']}")


def example_multiple_groups_streaming():
    """Load multiple groups and interleave them."""
    print("\n=== Example 3: Multiple Groups Streaming ===")
    
    groups = ["s3", "a4", "d5"]
    datasets = []
    
    # Load each group
    for group in groups:
        ds = load_dataset(
            "BeeGass/Group-Theory-Collection",
            data_dir=f"data/{group}",
            streaming=True
        )
        datasets.append(ds["train"])
    
    # Process one example from each group
    print("\nOne example from each group:")
    for group, stream in zip(groups, datasets):
        example = next(iter(stream))
        print(f"\n{group.upper()}:")
        print(f"  Input: {example['input_sequence']}")
        print(f"  Target: {example['target']}")
        print(f"  Order: {example['group_order']}")


def example_complexity_class_streaming():
    """Load groups by complexity class."""
    print("\n=== Example 4: Load by Complexity Class ===")
    
    # TC^0 (solvable) example
    print("\nTC^0 Group - F20 (Frobenius):")
    tc0_dataset = load_dataset(
        "BeeGass/Group-Theory-Collection",
        data_dir="data/f20",
        streaming=True
    )
    tc0_example = next(iter(tc0_dataset["train"]))
    print(f"  Order: {tc0_example['group_order']}")
    print(f"  Degree: {tc0_example['group_degree']}")
    print(f"  Type: {tc0_example['group_type']}")
    
    # NC^1 (non-solvable) example
    print("\nNC^1 Group - M11 (Mathieu):")
    nc1_dataset = load_dataset(
        "BeeGass/Group-Theory-Collection",
        data_dir="data/m11",
        streaming=True
    )
    nc1_example = next(iter(nc1_dataset["train"]))
    print(f"  Order: {nc1_example['group_order']}")
    print(f"  Degree: {nc1_example['group_degree']}")
    print(f"  Type: {nc1_example['group_type']}")


def example_batch_processing_streaming():
    """Process data in batches from streaming dataset."""
    print("\n=== Example 5: Batch Processing in Streaming ===")
    
    # Load A5 (alternating group) in streaming mode
    dataset = load_dataset(
        "BeeGass/Group-Theory-Collection",
        data_dir="data/a5",
        streaming=True
    )
    
    # Process in batches of 10
    batch_size = 10
    batch = []
    
    print(f"\nProcessing A5 in batches of {batch_size}:")
    for i, example in enumerate(dataset["train"]):
        batch.append(example)
        
        if len(batch) == batch_size:
            # Process batch
            avg_length = sum(ex["sequence_length"] for ex in batch) / len(batch)
            print(f"  Batch {i//batch_size + 1}: avg length = {avg_length:.1f}")
            batch = []
        
        if i >= 29:  # Process first 3 batches
            break


def example_large_group_streaming():
    """Efficiently handle large groups in streaming mode."""
    print("\n=== Example 6: Large Group Streaming ===")
    
    # Load S7 (order 5040) in streaming mode
    dataset = load_dataset(
        "BeeGass/Group-Theory-Collection",
        data_dir="data/s7",
        streaming=True
    )
    
    print("\nS7 (Symmetric group on 7 elements):")
    print(f"  Order: 5040 (7!)")
    
    # Count examples by length without loading all data
    length_counts = {}
    for i, example in enumerate(dataset["train"]):
        length = example["sequence_length"]
        length_counts[length] = length_counts.get(length, 0) + 1
        
        if i >= 1000:  # Sample first 1000
            break
    
    print(f"\nLength distribution (first 1000 examples):")
    for length in sorted(length_counts.keys())[:5]:
        print(f"  Length {length}: {length_counts[length]} examples")


def main():
    """Run all streaming examples."""
    print("Group Theory Collection - HuggingFace Streaming Examples")
    print("=" * 55)
    
    example_single_group_streaming()
    example_filter_by_length_streaming()
    example_multiple_groups_streaming()
    example_complexity_class_streaming()
    example_batch_processing_streaming()
    example_large_group_streaming()
    
    print("\n" + "=" * 55)
    print("All examples completed!")
    print("\nKey advantages of streaming:")
    print("- No need to download entire dataset")
    print("- Memory efficient for large groups")
    print("- Can filter and process on-the-fly")
    print("- Perfect for training neural networks")


if __name__ == "__main__":
    main()