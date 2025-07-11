#!/usr/bin/env python3
"""
Split superset datasets into individual group configurations.
Since dynamic filtering is no longer supported, we need individual datasets.
"""

import argparse
from pathlib import Path
from datasets import load_from_disk, DatasetDict
import shutil
from tqdm import tqdm

# Define all the individual configurations we need
CONFIGURATIONS = {
    "symmetric": [
        {"name": "s3", "degree": 3, "order": 6},
        {"name": "s4", "degree": 4, "order": 24},
        {"name": "s5", "degree": 5, "order": 120},
        {"name": "s6", "degree": 6, "order": 720},
        {"name": "s7", "degree": 7, "order": 5040},
        {"name": "s8", "degree": 8, "order": 40320},
        {"name": "s9", "degree": 9, "order": 362880},
        {"name": "s10", "degree": 10, "order": 3628800},
    ],
    "alternating": [
        {"name": "a3", "degree": 3, "order": 3},
        {"name": "a4", "degree": 4, "order": 12},
        {"name": "a5", "degree": 5, "order": 60},
        {"name": "a6", "degree": 6, "order": 360},
        {"name": "a7", "degree": 7, "order": 2520},
        {"name": "a8", "degree": 8, "order": 20160},
        {"name": "a9", "degree": 9, "order": 181440},
        {"name": "a10", "degree": 10, "order": 1814400},
    ],
    "cyclic": [
        {"name": f"c{n}", "degree": n, "order": n}
        for n in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
    ],
    "dihedral": [
        {"name": f"d{n}", "degree": n, "order": 2 * n}
        for n in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    ],
    "klein": [
        {"name": "v4", "degree": 4, "order": 4},
    ],
    "quaternion": [
        {"name": "q8", "degree": 8, "order": 8},
        {"name": "q16", "degree": 16, "order": 16},
        {"name": "q32", "degree": 32, "order": 32},
    ],
    "elementary_abelian": [
        {"name": "z2_1", "degree": 2, "order": 2},
        {"name": "z2_2", "degree": 4, "order": 4},
        {"name": "z2_3", "degree": 8, "order": 8},
        {"name": "z2_4", "degree": 16, "order": 16},
        {"name": "z2_5", "degree": 32, "order": 32},
        {"name": "z3_1", "degree": 3, "order": 3},
        {"name": "z3_2", "degree": 9, "order": 9},
        {"name": "z3_3", "degree": 27, "order": 27},
        {"name": "z5_1", "degree": 5, "order": 5},
        {"name": "z5_2", "degree": 25, "order": 25},
    ],
    "psl": [
        {"name": "psl2_5", "degree": 6, "order": 60},
        {"name": "psl2_7", "degree": 8, "order": 168},
    ],
    "frobenius": [
        {"name": "f20", "degree": 5, "order": 20},
        {"name": "f21", "degree": 7, "order": 21},
    ],
    "mathieu": [
        {"name": "m11", "degree": 11, "order": 7920},
        {"name": "m12", "degree": 12, "order": 95040},
    ],
}


def split_dataset(superset_path, output_dir, group_type, config):
    """Split a superset into individual configuration."""
    print(
        f"Processing {config['name']} (degree={config['degree']}, order={config['order']})"
    )

    # Load the superset
    dataset_dict = load_from_disk(superset_path)

    # Filter for this specific configuration
    filtered_train = dataset_dict["train"].filter(
        lambda x: x["group_degree"] == config["degree"]
        and x["group_order"] == config["order"]
    )
    filtered_test = dataset_dict["test"].filter(
        lambda x: x["group_degree"] == config["degree"]
        and x["group_order"] == config["order"]
    )

    if len(filtered_train) == 0:
        print(f"  WARNING: No training samples found for {config['name']}")
        return False

    # Create new dataset dict
    new_dataset = DatasetDict({"train": filtered_train, "test": filtered_test})

    # Save to output directory
    dataset_name = f"{config['name']}_data"
    output_path = output_dir / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    new_dataset.save_to_disk(output_path)
    print(
        f"  Saved {len(filtered_train)} train, {len(filtered_test)} test samples to {output_path}"
    )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split supersets into individual datasets"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing superset datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("individual_datasets"),
        help="Output directory for individual datasets",
    )
    parser.add_argument("--group-type", type=str, help="Process only this group type")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which groups to process
    if args.group_type:
        groups_to_process = {args.group_type: CONFIGURATIONS[args.group_type]}
    else:
        groups_to_process = CONFIGURATIONS

    # Process each group type
    total_created = 0
    for group_type, configs in groups_to_process.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {group_type} group")
        print(f"{'=' * 60}")

        superset_path = args.input_dir / f"{group_type}_superset"
        if not superset_path.exists():
            print(f"WARNING: Superset not found at {superset_path}")
            continue

        for config in configs:
            if split_dataset(superset_path, args.output_dir, group_type, config):
                total_created += 1

    print(f"\n{'=' * 60}")
    print(f"Created {total_created} individual datasets")
    print(f"{'=' * 60}")

    # Create a summary
    summary_path = args.output_dir / "dataset_list.txt"
    with open(summary_path, "w") as f:
        f.write("Individual Permutation Group Datasets\n")
        f.write("=====================================\n\n")

        for group_type, configs in CONFIGURATIONS.items():
            f.write(f"{group_type.upper()} GROUPS:\n")
            for config in configs:
                f.write(
                    f"  - {config['name']}_data: degree={config['degree']}, order={config['order']}\n"
                )
            f.write("\n")

    print(f"\nDataset list saved to {summary_path}")


if __name__ == "__main__":
    main()
