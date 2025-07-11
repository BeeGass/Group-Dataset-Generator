#!/usr/bin/env python3
"""
Generate all permutation group datasets.
"""

import subprocess
import os
from pathlib import Path

# Define all datasets to generate with higher orders where computationally feasible
DATASETS = [
    # Original groups - increase orders where possible
    {
        "group": "symmetric",
        "max_degree": 12,
        "samples": 500000,
    },  # S12 has order 479,001,600
    {
        "group": "alternating",
        "max_degree": 12,
        "samples": 400000,
    },  # A12 has order 239,500,800
    {"group": "cyclic", "max_degree": 100, "samples": 300000},  # C100 has order 100
    {"group": "dihedral", "max_degree": 50, "samples": 300000},  # D50 has order 100
    # Special groups
    {"group": "klein", "max_degree": 4, "samples": 40000},  # V4 has order 4 (fixed)
    {"group": "quaternion", "max_degree": 32, "samples": 100000},  # Q32 has order 32
    {
        "group": "elementary_abelian",
        "max_degree": 32,
        "samples": 150000,
    },  # Z_2^5 has order 32
    {"group": "psl", "max_degree": 8, "samples": 100000},  # PSL(2,7) has order 168
    {"group": "frobenius", "max_degree": 21, "samples": 80000},  # F21 has order 21
    {"group": "mathieu", "max_degree": 12, "samples": 120000},  # M12 has order 95,040
]

HF_REPO = "BeeGass/permutation-groups"


def run_command(cmd):
    """Run a command and handle errors."""
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def main():
    # Create output directory
    output_dir = Path("./data")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("GENERATING ALL PERMUTATION GROUP DATASETS")
    print("=" * 80)

    # Generate each dataset
    for dataset in DATASETS:
        print(f"\n{'=' * 60}")
        print(
            f"Generating {dataset['group']} superset (max degree {dataset['max_degree']})"
        )
        print(f"{'=' * 60}")

        cmd = (
            f"uv run python generate_superset.py "
            f"--group-type {dataset['group']} "
            f"--max-degree {dataset['max_degree']} "
            f"--num-samples {dataset['samples']} "
            f"--output-dir ./data/{dataset['group']}_superset "
            f"--hf-repo {HF_REPO}"
        )

        if not run_command(cmd):
            print(f"Failed to generate {dataset['group']} dataset")
            continue

        print(f"✓ Successfully generated {dataset['group']} dataset")

    print("\n" + "=" * 80)
    print("ALL DATASETS GENERATED")
    print("=" * 80)


if __name__ == "__main__":
    main()
