#!/usr/bin/env python3
"""
Examples for working with local permutation group datasets.
This demonstrates loading and processing datasets from the local individual_datasets directory.
"""

from datasets import load_from_disk
from pathlib import Path

def example_single_group_local():
    """Load a single group dataset from local storage."""
    print("=== Example 1: Single Group Local Loading ===")
    
    # Load S5 (symmetric group) from local storage
    dataset_path = Path("individual_datasets/s5_data")
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    dataset = load_from_disk(str(dataset_path))
    
    # Access the train split
    train_data = dataset["train"]
    
    # Process first 5 examples
    print(f"\nFirst 5 examples from S5 (total: {len(train_data)} train examples):")
    for i in range(min(5, len(train_data))):
        example = train_data[i]
        print(f"\nExample {i+1}:")
        print(f"  Input: {example['input_sequence'][:50]}..." if len(example['input_sequence']) > 50 else f"  Input: {example['input_sequence']}")
        print(f"  Target: {example['target']}")
        print(f"  Length: {example['sequence_length']}")
        print(f"  Group: {example['group_type']} (order {example['group_order']})")


def example_filter_by_length_local():
    """Filter examples by sequence length from local dataset."""
    print("\n=== Example 2: Filter by Length Local ===")
    
    # Load C10 (cyclic group) from local storage
    dataset_path = Path("individual_datasets/c10_data")
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    dataset = load_from_disk(str(dataset_path))
    
    # Filter for sequences of length 16
    filtered_data = dataset["train"].filter(
        lambda x: x["sequence_length"] == 16
    )
    
    print(f"\nExamples with length 16 from C10 (found {len(filtered_data)} examples):")
    for i in range(min(3, len(filtered_data))):
        example = filtered_data[i]
        print(f"\nExample {i+1}:")
        print(f"  Input: {example['input_sequence']}")
        print(f"  Target: {example['target']}")


def example_multiple_groups_local():
    """Load multiple groups from local storage."""
    print("\n=== Example 3: Multiple Groups Local ===")
    
    groups = ["s3", "a4", "d5"]
    datasets = {}
    
    # Load each group
    for group in groups:
        dataset_path = Path(f"individual_datasets/{group}_data")
        if dataset_path.exists():
            datasets[group] = load_from_disk(str(dataset_path))
        else:
            print(f"Warning: {group} dataset not found")
    
    # Process one example from each group
    print("\nOne example from each group:")
    for group, dataset in datasets.items():
        if "train" in dataset:
            example = dataset["train"][0]
            print(f"\n{group.upper()}:")
            print(f"  Input: {example['input_sequence'][:30]}..." if len(example['input_sequence']) > 30 else f"  Input: {example['input_sequence']}")
            print(f"  Target: {example['target']}")
            print(f"  Order: {example['group_order']}")


def example_complexity_class_local():
    """Load groups by complexity class from local storage."""
    print("\n=== Example 4: Load by Complexity Class Local ===")
    
    # TC^0 (solvable) example
    print("\nTC^0 Group - F20 (Frobenius):")
    tc0_path = Path("individual_datasets/f20_data")
    if tc0_path.exists():
        tc0_dataset = load_from_disk(str(tc0_path))
        tc0_example = tc0_dataset["train"][0]
        print(f"  Order: {tc0_example['group_order']}")
        print(f"  Degree: {tc0_example['group_degree']}")
        print(f"  Type: {tc0_example['group_type']}")
    
    # NC^1 (non-solvable) example
    print("\nNC^1 Group - M11 (Mathieu):")
    nc1_path = Path("individual_datasets/m11_data")
    if nc1_path.exists():
        nc1_dataset = load_from_disk(str(nc1_path))
        nc1_example = nc1_dataset["train"][0]
        print(f"  Order: {nc1_example['group_order']}")
        print(f"  Degree: {nc1_example['group_degree']}")
        print(f"  Type: {nc1_example['group_type']}")


def example_batch_processing_local():
    """Process data in batches from local dataset."""
    print("\n=== Example 5: Batch Processing Local ===")
    
    # Load A5 (alternating group) from local storage
    dataset_path = Path("individual_datasets/a5_data")
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    dataset = load_from_disk(str(dataset_path))
    train_data = dataset["train"]
    
    # Process in batches of 10
    batch_size = 10
    
    print(f"\nProcessing A5 in batches of {batch_size}:")
    for batch_idx in range(3):  # Process first 3 batches
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(train_data))
        
        if start_idx >= len(train_data):
            break
            
        # Access individual elements instead of slicing
        batch = []
        for i in range(start_idx, end_idx):
            batch.append(train_data[i])
            
        avg_length = sum(ex["sequence_length"] for ex in batch) / len(batch)
        print(f"  Batch {batch_idx + 1}: avg length = {avg_length:.1f}")


def example_dataset_statistics():
    """Show statistics for available local datasets."""
    print("\n=== Example 6: Dataset Statistics ===")
    
    dataset_dir = Path("individual_datasets")
    if not dataset_dir.exists():
        print("Error: individual_datasets directory not found")
        return
    
    print("\nAvailable local datasets:")
    print(f"{'Group':<15} {'Order':<10} {'Degree':<10} {'Train':<10} {'Test':<10}")
    print("-" * 55)
    
    for dataset_path in sorted(dataset_dir.glob("*_data")):
        try:
            dataset = load_from_disk(str(dataset_path))
            if "train" in dataset and len(dataset["train"]) > 0:
                example = dataset["train"][0]
                group_name = dataset_path.name.replace("_data", "").upper()
                print(f"{group_name:<15} {example['group_order']:<10} {example['group_degree']:<10} {len(dataset['train']):<10} {len(dataset['test']):<10}")
        except Exception as e:
            print(f"Error loading {dataset_path.name}: {e}")


def main():
    """Run all local examples."""
    print("Group Theory Collection - Local Dataset Examples")
    print("=" * 50)
    
    example_single_group_local()
    example_filter_by_length_local()
    example_multiple_groups_local()
    example_complexity_class_local()
    example_batch_processing_local()
    example_dataset_statistics()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nThese examples demonstrate working with local datasets.")
    print("Once datasets are fully uploaded to HuggingFace, you can use")
    print("the streaming_examples.py script for cloud-based access.")


if __name__ == "__main__":
    main()