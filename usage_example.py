"""
Example usage of the permutation-groups dataset.

This shows how to load individual datasets or all datasets combined.
"""

from datasets import load_dataset

print("=" * 60)
print("Permutation Groups Dataset Usage Examples")
print("=" * 60)

# Example 1: Load a specific dataset
print("\n1. Loading individual datasets:")
print("-" * 40)

# Load S3 dataset
s3_dataset = load_dataset("BeeGass/permutation-groups", name="s3_data", trust_remote_code=True)
print(f"✓ S3 dataset loaded")
print(f"  - Train samples: {len(s3_dataset['train'])}")
print(f"  - Test samples: {len(s3_dataset['test'])}")
print(f"  - First example: {s3_dataset['train'][0]}")

# Load A5 dataset  
a5_dataset = load_dataset("BeeGass/permutation-groups", name="a5_data", trust_remote_code=True)
print(f"\n✓ A5 dataset loaded")
print(f"  - Train samples: {len(a5_dataset['train'])}")
print(f"  - Test samples: {len(a5_dataset['test'])}")

# Example 2: Load all datasets combined
print("\n2. Loading all datasets combined:")
print("-" * 40)
print("Note: This may take longer as it loads all 8 datasets")

all_datasets = load_dataset("BeeGass/permutation-groups", name="all", trust_remote_code=True)
print(f"✓ All datasets loaded")
print(f"  - Total train samples: {len(all_datasets['train'])}")
print(f"  - Total test samples: {len(all_datasets['test'])}")

# Example 3: Available configurations
print("\n3. Available configurations:")
print("-" * 40)
configs = ["s3_data", "s4_data", "s5_data", "s6_data", "s7_data", 
           "a5_data", "a6_data", "a7_data", "all"]
for config in configs:
    print(f"  - {config}")

print("\n" + "=" * 60)
print("✅ Dataset loading works correctly!")
print("=" * 60)