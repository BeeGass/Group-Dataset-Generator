#!/usr/bin/env python3
"""Script to generate all permutation group datasets and upload to HuggingFace."""

import subprocess
import sys
from pathlib import Path

# Define all groups and their configurations
GROUPS = [
    # Group name, num_samples, max_len
    ("S3", 10000, 512),
    ("S4", 20000, 512),
    ("S5", 50000, 512),
    ("S6", 100000, 512),
    ("S7", 200000, 512),
    ("A3", 5000, 512),      # Smaller dataset for A3 (only 3 elements)
    ("A4", 15000, 512),
    ("A5", 30000, 512),
    ("A6", 80000, 512),
    ("A7", 150000, 512),
]

HF_REPO = "BeeGass/permutation-groups"


def generate_dataset(group_name, num_samples, max_len):
    """Generate a single dataset."""
    print(f"\n{'='*60}")
    print(f"Generating {group_name} dataset...")
    print(f"{'='*60}")
    
    output_dir = f"./{group_name.lower()}_data"
    
    cmd = [
        sys.executable, "generate.py",
        "--group-name", group_name,
        "--num-samples", str(num_samples),
        "--min-len", "3",
        "--max-len", str(max_len),
        "--test-split-size", "0.2",
        "--output-dir", output_dir,
        "--hf-repo", HF_REPO
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully generated and uploaded {group_name} dataset")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate {group_name} dataset")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Generate all datasets."""
    print("Generating all permutation group datasets...")
    print(f"Target repository: {HF_REPO}")
    
    successful = []
    failed = []
    
    for group_name, num_samples, max_len in GROUPS:
        if generate_dataset(group_name, num_samples, max_len):
            successful.append(group_name)
        else:
            failed.append(group_name)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully generated: {', '.join(successful)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    else:
        print("All datasets generated successfully!")
    
    # Upload the dataset script
    if successful and not failed:
        print("\nUploading dataset script to HuggingFace...")
        try:
            subprocess.run([sys.executable, "upload_dataset_script.py"], check=True)
            print("✓ Dataset script uploaded successfully")
        except subprocess.CalledProcessError:
            print("✗ Failed to upload dataset script")


if __name__ == "__main__":
    main()