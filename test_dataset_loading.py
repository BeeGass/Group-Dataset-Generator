"""Test script to verify the dataset loading script works correctly."""

import os
import sys
from datasets import load_dataset

def test_local_loading():
    """Test loading the dataset from local files."""
    print("Testing local dataset loading...")
    
    # Test loading a single configuration
    print("\n1. Testing single configuration (s3_data):")
    try:
        dataset = load_dataset(
            "./permutation-groups.py",
            name="s3_data",
            data_dir=".",
            trust_remote_code=True
        )
        print(f"   Success! Loaded {len(dataset['train'])} train samples and {len(dataset['test'])} test samples")
        print(f"   First train sample: {dataset['train'][0]}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test loading another configuration
    print("\n2. Testing another configuration (s5_data):")
    try:
        dataset = load_dataset(
            "./permutation-groups.py",
            name="s5_data",
            data_dir=".",
            trust_remote_code=True
        )
        print(f"   Success! Loaded {len(dataset['train'])} train samples and {len(dataset['test'])} test samples")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test loading all configurations
    print("\n3. Testing 'all' configuration:")
    try:
        dataset = load_dataset(
            "./permutation-groups.py",
            name="all",
            data_dir=".",
            trust_remote_code=True
        )
        print(f"   Success! Loaded {len(dataset['train'])} train samples and {len(dataset['test'])} test samples")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_local_loading()