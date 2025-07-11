#!/usr/bin/env python3
"""
Master script to generate all permutation group datasets.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_generator(script_name: str, args: list):
    """Run a generator script with given arguments."""
    cmd = [sys.executable, script_name] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    print(result.stdout)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate all permutation group datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../individual_datasets",
        help="Output directory for all datasets",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        help="Specific groups to generate (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Define all groups to generate
    all_groups = {
        "symmetric": {
            "script": "symmetric_generator.py",
            "args": ["--degrees", "3", "4", "5", "6", "7", "8", "9"],
        },
        "alternating": {
            "script": "alternating_generator.py",
            "args": ["--degrees", "3", "4", "5", "6", "7", "8", "9"],
        },
        "cyclic": {
            "script": "cyclic_generator.py",
            "args": [
                "--degrees",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "12",
                "15",
                "20",
                "25",
                "30",
            ],
        },
        "dihedral": {
            "script": "dihedral_generator.py",
            "args": [
                "--degrees",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "12",
                "15",
                "20",
            ],
        },
        "quaternion": {
            "script": "quaternion_generator.py",
            "args": ["--powers", "3", "4", "5"],  # Q8, Q16, Q32
        },
        "klein": {"script": "klein_generator.py", "args": []},
        "elementary_abelian": {
            "script": "elementary_abelian_generator.py",
            "args": [
                "--groups",
                "2,1",
                "2,2",
                "2,3",
                "2,4",
                "2,5",
                "3,1",
                "3,2",
                "3,3",
                "5,1",
                "5,2",
            ],
        },
        "frobenius": {
            "script": "frobenius_generator.py",
            "args": ["--groups", "20", "21"],
        },
        "psl": {"script": "psl_generator.py", "args": ["--primes", "5", "7"]},
        "mathieu": {"script": "mathieu_generator.py", "args": ["--groups", "11", "12"]},
    }

    # Determine which groups to generate
    if args.groups:
        groups_to_generate = {k: v for k, v in all_groups.items() if k in args.groups}
    else:
        groups_to_generate = all_groups

    print(f"Generating {len(groups_to_generate)} group types")
    print("=" * 60)

    # Generate each group type
    success_count = 0
    for group_name, config in groups_to_generate.items():
        print(f"\n{'=' * 60}")
        print(f"Generating {group_name} groups")
        print("=" * 60)

        script_path = Path(__file__).parent / config["script"]
        generator_args = config["args"] + [
            "--output-dir",
            args.output_dir,
            "--seed",
            str(args.seed),
        ]

        if run_generator(str(script_path), generator_args):
            success_count += 1
            print(f"✓ Successfully generated {group_name} groups")
        else:
            print(f"✗ Failed to generate {group_name} groups")

    print(f"\n{'=' * 60}")
    print(f"Generation complete: {success_count}/{len(groups_to_generate)} successful")
    print("=" * 60)


if __name__ == "__main__":
    main()
