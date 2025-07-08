"""Example usage of the permutation-groups dataset.

This script demonstrates:
1. Loading individual dataset configurations
2. Loading all datasets combined
3. Verifying permutation compositions
4. Performance benchmarking
"""

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json
import time
from sympy.combinatorics import Permutation
import argparse


def load_dataset_with_timing(repo_id, config_name):
    """Load a dataset and report timing."""
    print(f"\nLoading {config_name}...", end="", flush=True)
    start = time.time()
    dataset = load_dataset(repo_id, name=config_name, trust_remote_code=True)
    elapsed = time.time() - start
    print(f" ✓ ({elapsed:.2f}s)")
    print(f"  Train: {len(dataset['train']):,} samples")
    print(f"  Test: {len(dataset['test']):,} samples")
    return dataset


def verify_dataset_composition(repo_id: str, config_name: str, num_samples: int = 5):
    """Verify that permutation compositions are correct in a dataset."""
    print(f"\n{'='*60}")
    print(f"Verifying {config_name}")
    print(f"{'='*60}")
    
    # Load dataset
    dataset = load_dataset(repo_id, name=config_name, trust_remote_code=True, split="train")
    
    # Load metadata
    try:
        metadata_path = hf_hub_download(
            repo_id=repo_id, 
            filename=f"data/{config_name}/metadata.json", 
            repo_type="dataset"
        )
        with open(metadata_path, "r") as f:
            id_to_perm = json.load(f)
    except:
        print("❌ Could not load metadata.json")
        return False
    
    # Verify multiple samples
    all_correct = True
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_ids = [int(x) for x in sample['input_sequence'].split()]
        target_id = int(sample['target'])
        
        # Decode permutations
        try:
            input_perms = [eval(id_to_perm[str(i)]) for i in input_ids]
            target_perm = eval(id_to_perm[str(target_id)])
            
            # Compute composition
            group_degree = len(target_perm)
            composed = Permutation(group_degree - 1)  # Identity
            for perm in reversed([Permutation(p) for p in input_perms]):
                composed = perm * composed
            
            result = composed.array_form
            is_correct = result == target_perm
            
            if i < 3:  # Show details for first 3 samples
                print(f"\nSample {i}:")
                print(f"  Input IDs: {input_ids[:5]}{'...' if len(input_ids) > 5 else ''}")
                print(f"  Target ID: {target_id}")
                print(f"  Expected: {target_perm}")
                print(f"  Computed: {result}")
                print(f"  Result: {'✓ PASS' if is_correct else '❌ FAIL'}")
            
            if not is_correct:
                all_correct = False
                
        except Exception as e:
            print(f"❌ Error in sample {i}: {e}")
            all_correct = False
    
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_correct else '❌ SOME TESTS FAILED'}")
    return all_correct


def run_examples():
    """Run basic loading examples."""
    repo_id = "BeeGass/permutation-groups"
    
    print("="*60)
    print("Permutation Groups Dataset Examples")
    print("="*60)
    
    # Example 1: Load individual datasets
    print("\n1. Loading Individual Datasets:")
    print("-"*40)
    
    # Small datasets
    s3_dataset = load_dataset_with_timing(repo_id, "s3_data")
    s4_dataset = load_dataset_with_timing(repo_id, "s4_data")
    
    # Medium dataset
    a5_dataset = load_dataset_with_timing(repo_id, "a5_data")
    
    # Show sample data
    print("\n2. Sample Data:")
    print("-"*40)
    print(f"S3 first sample: {s3_dataset['train'][0]}")
    
    # Example 3: All configurations
    print("\n3. Available Configurations:")
    print("-"*40)
    configs = [
        ("s3_data", "Symmetric Group S3 (6 elements)"),
        ("s4_data", "Symmetric Group S4 (24 elements)"),
        ("s5_data", "Symmetric Group S5 (120 elements)"),
        ("s6_data", "Symmetric Group S6 (720 elements)"),
        ("s7_data", "Symmetric Group S7 (5040 elements)"),
        ("a5_data", "Alternating Group A5 (60 elements)"),
        ("a6_data", "Alternating Group A6 (360 elements)"),
        ("a7_data", "Alternating Group A7 (2520 elements)"),
        ("all", "All datasets combined")
    ]
    for config, desc in configs:
        print(f"  • {config}: {desc}")
    
    print("\n4. Usage Examples:")
    print("-"*40)
    print('# Load a specific dataset:')
    print('ds = load_dataset("BeeGass/permutation-groups", name="s5_data", trust_remote_code=True)')
    print('\n# Load all datasets:')
    print('all_ds = load_dataset("BeeGass/permutation-groups", name="all", trust_remote_code=True)')
    print('\n# Access splits:')
    print('train_data = ds["train"]')
    print('test_data = ds["test"]')


def run_verification():
    """Run verification on all datasets."""
    repo_id = "BeeGass/permutation-groups"
    
    print("="*60)
    print("Verifying All Permutation Group Datasets")
    print("="*60)
    
    configs = [
        "s3_data", "s4_data", "s5_data", "s6_data", "s7_data",
        "a5_data", "a6_data", "a7_data"
    ]
    
    results = {}
    for config in configs:
        results[config] = verify_dataset_composition(repo_id, config, num_samples=10)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    for config, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{config}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("✓ ALL DATASETS VERIFIED!" if all_passed else "❌ SOME DATASETS HAVE ERRORS!"))


def benchmark_loading():
    """Benchmark dataset loading times."""
    repo_id = "BeeGass/permutation-groups"
    
    print("="*60)
    print("Dataset Loading Benchmark")
    print("="*60)
    
    configs = [
        "s3_data", "s4_data", "s5_data", "s6_data", "s7_data",
        "a5_data", "a6_data", "a7_data", "all"
    ]
    
    times = {}
    for config in configs:
        start = time.time()
        dataset = load_dataset(repo_id, name=config, trust_remote_code=True)
        elapsed = time.time() - start
        times[config] = elapsed
        
        if config != "all":
            total_samples = len(dataset['train']) + len(dataset['test'])
        else:
            total_samples = len(dataset['train']) + len(dataset['test'])
        
        print(f"{config}: {elapsed:.2f}s ({total_samples:,} total samples)")
    
    print(f"\nFastest: {min(times, key=times.get)} ({times[min(times, key=times.get)]:.2f}s)")
    print(f"Slowest: {max(times, key=times.get)} ({times[max(times, key=times.get)]:.2f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examples for permutation-groups dataset")
    parser.add_argument("--verify", action="store_true", help="Run verification on all datasets")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark loading times")
    
    args = parser.parse_args()
    
    if args.verify:
        run_verification()
    elif args.benchmark:
        benchmark_loading()
    else:
        run_examples()