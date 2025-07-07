from datasets import load_dataset

# Test loading individual datasets
print("Testing S3 dataset...")
s3_dataset = load_dataset("BeeGass/permutation-groups", name="s3_data", trust_remote_code=True)
print(f"✓ S3 dataset loaded: {len(s3_dataset['train'])} train, {len(s3_dataset['test'])} test")

print("\nTesting S7 dataset...")
s7_dataset = load_dataset("BeeGass/permutation-groups", name="s7_data", trust_remote_code=True)
print(f"✓ S7 dataset loaded: {len(s7_dataset['train'])} train, {len(s7_dataset['test'])} test")

print("\nTesting A7 dataset...")
a7_dataset = load_dataset("BeeGass/permutation-groups", name="a7_data", trust_remote_code=True)
print(f"✓ A7 dataset loaded: {len(a7_dataset['train'])} train, {len(a7_dataset['test'])} test")

print("\nTesting 'all' configuration...")
all_datasets = load_dataset("BeeGass/permutation-groups", name="all", trust_remote_code=True)
print(f"✓ All datasets loaded: {len(all_datasets['train'])} train, {len(all_datasets['test'])} test")

print("\n✅ All tests passed! You can now load datasets using:")
print('  s3_dataset = load_dataset("BeeGass/permutation-groups", name="s3_data", trust_remote_code=True)')
print('  all_datasets = load_dataset("BeeGass/permutation-groups", name="all", trust_remote_code=True)')