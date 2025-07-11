#!/usr/bin/env python3
"""
Generate individual permutation group datasets directly (not from supersets).
This ensures each group dataset is generated correctly with proper parameters.
"""

import os
import numpy as np
from datasets import Dataset, DatasetDict
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

# Import the permutation group generation functions
from permutation_groups import (
    get_symmetric_group,
    get_alternating_group,
    get_cyclic_group,
    get_dihedral_group,
    get_quaternion_group,
    get_elementary_abelian_group,
    get_klein_four_group,
    get_frobenius_group,
    get_psl_group,
    get_mathieu_group,
)


class IndividualGroupGenerator:
    """Generate individual permutation group datasets."""

    def __init__(self, output_dir: str = "individual_datasets_v2", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.seed = seed
        np.random.seed(seed)

        # Define all groups to generate
        self.group_configs = self._get_group_configs()

    def _get_group_configs(self) -> List[Dict]:
        """Get configuration for all groups to generate."""
        configs = []

        # Symmetric groups
        for n in range(3, 10):  # S3 to S9
            configs.append(
                {
                    "type": "symmetric",
                    "degree": n,
                    "name": f"s{n}",
                    "generator": get_symmetric_group,
                    "params": {"n": n},
                }
            )

        # Alternating groups
        for n in range(3, 10):  # A3 to A9
            configs.append(
                {
                    "type": "alternating",
                    "degree": n,
                    "name": f"a{n}",
                    "generator": get_alternating_group,
                    "params": {"n": n},
                }
            )

        # Cyclic groups
        for n in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]:
            configs.append(
                {
                    "type": "cyclic",
                    "degree": n,
                    "name": f"c{n}",
                    "generator": get_cyclic_group,
                    "params": {"n": n},
                }
            )

        # Dihedral groups
        for n in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
            configs.append(
                {
                    "type": "dihedral",
                    "degree": n,
                    "name": f"d{n}",
                    "generator": get_dihedral_group,
                    "params": {"n": n},
                }
            )

        # Quaternion groups
        for k in [3, 4, 5]:  # Q8, Q16, Q32
            n = 2**k
            configs.append(
                {
                    "type": "quaternion",
                    "degree": n,
                    "name": f"q{n}",
                    "generator": get_quaternion_group,
                    "params": {"k": k},
                }
            )

        # Klein Four group
        configs.append(
            {
                "type": "klein",
                "degree": 4,
                "name": "v4",
                "generator": get_klein_four_group,
                "params": {},
            }
        )

        # Elementary abelian groups
        for p, k in [
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 1),
            (3, 2),
            (3, 3),
            (5, 1),
            (5, 2),
        ]:
            configs.append(
                {
                    "type": "elementary_abelian",
                    "degree": p**k,
                    "name": f"z{p}_{k}",
                    "generator": get_elementary_abelian_group,
                    "params": {"p": p, "k": k},
                }
            )

        # Frobenius groups
        for n in [20, 21]:
            configs.append(
                {
                    "type": "frobenius",
                    "degree": n,
                    "name": f"f{n}",
                    "generator": get_frobenius_group,
                    "params": {"n": n},
                }
            )

        # PSL groups
        for p in [5, 7]:
            configs.append(
                {
                    "type": "psl",
                    "degree": p,  # Note: actual order is different
                    "name": f"psl2_{p}",
                    "generator": get_psl_group,
                    "params": {"p": p},
                }
            )

        # Mathieu groups
        for n in [11, 12]:
            configs.append(
                {
                    "type": "mathieu",
                    "degree": n,
                    "name": f"m{n}",
                    "generator": get_mathieu_group,
                    "params": {"n": n},
                }
            )

        return configs

    def generate_dataset_for_group(
        self,
        config: Dict,
        num_train_samples: int = 100000,
        num_test_samples: int = 20000,
        min_seq_length: int = 3,
        max_seq_length: int = 1024,
    ) -> DatasetDict:
        """Generate dataset for a single group."""
        print(f"\nGenerating dataset for {config['name']}...")

        # Generate the group
        generator = config["generator"]
        permutations = generator(**config["params"])
        group_order = len(permutations)

        print(f"Group order: {group_order}")

        # Create permutation map for metadata
        perm_map = {}
        for i, perm in enumerate(permutations):
            perm_map[str(i)] = perm.tolist() if hasattr(perm, "tolist") else list(perm)

        # Generate samples
        def generate_samples(num_samples):
            samples = {
                "input_sequence": [],
                "target": [],
                "sequence_length": [],
                "group_degree": [],
                "group_order": [],
                "group_type": [],
            }

            for _ in tqdm(range(num_samples), desc="Generating samples"):
                # Random sequence length
                seq_length = np.random.randint(min_seq_length, max_seq_length + 1)

                # Random permutation indices
                input_indices = np.random.randint(0, group_order, size=seq_length)

                # Compute composition
                result = np.arange(len(permutations[0]))
                for idx in input_indices:
                    result = result[permutations[idx]]

                # Find target index
                target_idx = None
                for i, perm in enumerate(permutations):
                    if np.array_equal(result, perm):
                        target_idx = i
                        break

                if target_idx is None:
                    raise ValueError("Composition result not found in group!")

                # Store as strings to match original format
                samples["input_sequence"].append(" ".join(map(str, input_indices)))
                samples["target"].append(str(target_idx))
                samples["sequence_length"].append(seq_length)
                samples["group_degree"].append(config["degree"])
                samples["group_order"].append(group_order)
                samples["group_type"].append(config["type"])

            return samples

        # Generate train and test sets
        train_samples = generate_samples(num_train_samples)
        test_samples = generate_samples(num_test_samples)

        # Create datasets
        train_dataset = Dataset.from_dict(train_samples)
        test_dataset = Dataset.from_dict(test_samples)

        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

        # Save metadata
        metadata = {
            "group_type": config["type"],
            "group_name": config["name"],
            "group_order": group_order,
            "group_degree": config["degree"],
            "permutation_map": perm_map,
        }

        return dataset_dict, metadata

    def generate_all_groups(self, groups_to_generate: List[str] = None):
        """Generate datasets for all configured groups."""
        if groups_to_generate is None:
            configs = self.group_configs
        else:
            configs = [c for c in self.group_configs if c["name"] in groups_to_generate]

        print(f"Generating {len(configs)} group datasets...")

        for config in configs:
            try:
                # Generate dataset
                dataset_dict, metadata = self.generate_dataset_for_group(config)

                # Save dataset
                output_path = self.output_dir / f"{config['name']}_data"
                dataset_dict.save_to_disk(str(output_path))

                # Save metadata
                metadata_path = output_path / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"✓ Saved {config['name']} to {output_path}")

            except Exception as e:
                print(f"✗ Error generating {config['name']}: {e}")
                continue

    def verify_dataset(self, group_name: str):
        """Verify a generated dataset."""
        dataset_path = self.output_dir / f"{group_name}_data"
        if not dataset_path.exists():
            print(f"Dataset {group_name} not found at {dataset_path}")
            return False

        # Load dataset
        dataset = Dataset.load_from_disk(str(dataset_path))
        train_data = dataset["train"]

        # Load metadata
        metadata_path = dataset_path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        print(f"\nVerifying {group_name}:")
        print(f"  Group order: {metadata['group_order']}")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Test samples: {len(dataset['test'])}")

        # Check sample
        sample = train_data[0]
        input_ids = [int(x) for x in sample["input_sequence"].split()]
        target_id = int(sample["target"])

        print(f"  Sample input IDs range: [{min(input_ids)}, {max(input_ids)}]")
        print(f"  Expected range: [0, {metadata['group_order'] - 1}]")

        # Verify all IDs are in valid range
        valid = all(0 <= x < metadata["group_order"] for x in input_ids)
        valid &= 0 <= target_id < metadata["group_order"]

        if valid:
            print("  ✓ All IDs in valid range")
        else:
            print("  ✗ IDs out of range!")

        return valid


def main():
    parser = argparse.ArgumentParser(
        description="Generate individual permutation group datasets"
    )
    parser.add_argument(
        "--output-dir",
        default="individual_datasets_v2",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--groups", nargs="+", help="Specific groups to generate (e.g., q8 q16 q32)"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify generated datasets"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generator = IndividualGroupGenerator(output_dir=args.output_dir, seed=args.seed)

    if args.verify:
        # Verify specific groups or all
        groups = args.groups or [c["name"] for c in generator.group_configs]
        for group in groups:
            generator.verify_dataset(group)
    else:
        # Generate datasets
        generator.generate_all_groups(args.groups)


if __name__ == "__main__":
    main()
