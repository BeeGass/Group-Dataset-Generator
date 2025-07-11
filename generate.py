import argparse
import os
import random
from pathlib import Path

import pandas as pd
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
from sympy.combinatorics import Permutation
from sympy.combinatorics.named_groups import AlternatingGroup, SymmetricGroup
from tqdm import tqdm


def get_group(name: str):
    """Factory function to get a SymPy group object by name."""
    if name.upper().startswith("S"):
        degree = int(name[1:])
        return SymmetricGroup(degree)
    elif name.upper().startswith("A"):
        degree = int(name[1:])
        return AlternatingGroup(degree)
    else:
        raise ValueError(f"Unknown group name: {name}. Use format 'S<n>' or 'A<n>'.")


def generate_and_map_permutations(group):
    """Generates all permutations in a group and maps them to integer IDs."""
    print(f"Generating all {group.order()} elements for {group.degree}-degree group...")
    elements = list(group.generate())
    perm_to_id = {str(p.array_form): i for i, p in enumerate(elements)}
    id_to_perm = {i: p for i, p in enumerate(elements)}
    print("Element mapping complete.")
    return perm_to_id, id_to_perm


def generate_composition_sample(id_to_perm, perm_to_id, min_len, max_len, group_degree):
    """Generates a single permutation composition problem."""
    seq_len = random.randint(min_len, max_len)
    input_ids = [random.choice(range(len(id_to_perm))) for _ in range(seq_len)]
    input_perms = [id_to_perm[i] for i in input_ids]

    composed_perm = Permutation(group_degree - 1)
    for p in reversed(input_perms):
        composed_perm = p * composed_perm

    target_id = perm_to_id[str(composed_perm.array_form)]

    return {
        "input_sequence": " ".join(map(str, input_ids)),
        "target": str(target_id),
    }


def generate_readme_content(
    args, group_order, group_degree, num_train_samples, num_test_samples
):
    readme_content = f"""---
pretty_name: Permutation Composition Dataset ({args.group_name})
size_categories:
  - small
  - medium
  - large
  - xlarge
  - xxlarge
tags:
  - mathematics
  - group-theory
  - permutations
  - sequence-to-sequence
  - benchmark
  - generated
task_categories:
  - text-generation
  - sequence-modeling
annotations_creators:
  - no-annotations
language_creators:
  - other
language:
  - en
licenses:
  - mit
--- 

# Permutation Composition Dataset for {args.group_name}

This dataset contains sequences of permutation IDs and their compositions, designed for benchmarking sequence-to-sequence models on group theory tasks.

## Dataset Structure

The dataset is split into `train` and `test` sets. Each sample in the dataset has the following features:

- `input_sequence`: A space-separated string of integer IDs representing the sequence of permutations to be composed.
- `target`: An integer ID representing the composition of the `input_sequence` permutations.

## Group Details

- **Group Name**: {args.group_name}
- **Group Type**: {"Symmetric Group" if args.group_name.upper().startswith("S") else "Alternating Group"}
- **Degree**: {group_degree} (permutations act on {group_degree} elements)
- **Order**: {group_order} (total number of elements in the group)

## Data Generation

This dataset was generated using the `s5-data-gen` script. The generation process involves:

1.  Generating all unique permutations for the specified group.
2.  Mapping each unique permutation to a unique integer ID.
3.  Randomly sampling sequences of these permutation IDs.
4.  Composing the permutations in the sequence (from right to left: `p_n o ... o p_2 o p_1`).
5.  Mapping the resulting composed permutation to its integer ID as the target.

### Generation Parameters:

- **Total Samples**: {args.num_samples}
- **Minimum Sequence Length**: {args.min_len}
- **Maximum Sequence Length**: {args.max_len}
- **Test Split Size**: {args.test_split_size}

## Dataset Statistics

- **Train Samples**: {num_train_samples}
- **Test Samples**: {num_test_samples}

## Permutation Mapping

The mapping from integer IDs to their corresponding permutation array forms is provided in the `metadata.json` file alongside the dataset. This file is crucial for interpreting the `input_sequence` and `target` IDs.

Example of `metadata.json` content:

```json
{{
  "0": "[0, 1, 2, 3, 4]",
  "1": "[0, 1, 3, 2, 4]",
  "2": "[0, 1, 4, 3, 2]",
  "3": "[0, 2, 1, 3, 4]",
  "4": "[0, 2, 3, 1, 4]"
  // ... and so on for all {group_order} permutations
}}
```

## Usage

You can load this dataset using the Hugging Face `datasets` library:

```python
from datasets import load_dataset
import json
from huggingface_hub import hf_hub_download

# Load the dataset
dataset = load_dataset("{args.hf_repo}")

# Load the permutation mapping
metadata_path = hf_hub_download(repo_id="{args.hf_repo}", filename="metadata.json")
with open(metadata_path, "r") as f:
    id_to_perm_map = json.load(f)

# Example: Decode a sample
first_train_sample = dataset["train"][0]
input_ids = [int(x) for x in first_train_sample["input_sequence"].split(" ")]
target_id = int(first_train_sample["target"])

print(f"Input sequence IDs: {{input_ids}}")
print(f"Target ID: {{target_id}}")

# Convert IDs back to permutations (example for the first input permutation)
# Note: SymPy Permutation expects a list of integers, not a string representation
# You would need to parse the string representation from id_to_perm_map
# For example: eval(id_to_perm_map[str(input_ids[0])])

print(f"First input permutation (array form): {{id_to_perm_map[str(input_ids[0])]}}")
print(f"Target permutation (array form): {{id_to_perm_map[str(target_id)]}}")
```

## License

This dataset is licensed under the MIT License.
"""
    return readme_content


def main():
    """Main function to generate and process the dataset."""
    parser = argparse.ArgumentParser(
        description="Generate permutation composition datasets for groups like S5 and A5."
    )
    parser.add_argument(
        "--group-name",
        type=str,
        default="S5",
        help="Name of the permutation group (e.g., 'S5', 'A5').",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="Total number of samples to generate.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=3,
        help="Minimum length of permutation sequences.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Maximum length of permutation sequences.",
    )
    parser.add_argument(
        "--test-split-size",
        type=float,
        default=0.2,
        help="Fraction of data to reserve for the test set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Local directory to save the dataset.",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        help="Optional: Hugging Face repository ID to push the dataset to (e.g., 'your-username/s5-benchmark').",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face API token for authentication. Can also be set via env var HUGGING_FACE_HUB_TOKEN.",
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("--- Starting Dataset Generation ---")
    print(f"Configuration: {vars(args)}")

    group = get_group(args.group_name)
    group_degree = group.degree
    perm_to_id, id_to_perm = generate_and_map_permutations(group)

    print(f"Generating {args.num_samples} samples...")
    samples = [
        generate_composition_sample(
            id_to_perm, perm_to_id, args.min_len, args.max_len, group_degree
        )
        for _ in tqdm(range(args.num_samples), desc="Generating samples")
    ]

    df = pd.DataFrame(samples)
    raw_dataset = Dataset.from_pandas(df)

    dataset_dict = raw_dataset.train_test_split(test_size=args.test_split_size)

    # Add vocabulary metadata to the dataset for later use
    for split in dataset_dict.keys():
        dataset_dict[
            split
        ].info.description = (
            f"Permutation composition benchmark for the {args.group_name} group."
        )
        dataset_dict[
            split
        ].info.homepage = "https://github.com/your-repo"  # Optional: Link to your repo
        dataset_dict[split].info.license = "mit"

    # Save id_to_perm mapping separately
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({k: str(v.array_form) for k, v in id_to_perm.items()}, f, indent=4)

    print("--- Dataset Generation Complete ---")
    print(f"Train samples: {len(dataset_dict['train'])}")
    print(f"Test samples: {len(dataset_dict['test'])}")
    print(f"Dataset features: {dataset_dict['train'].features}")

    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_path)
    print(f"Dataset saved locally to: {output_path.resolve()}")

    if args.hf_repo:
        print(f"--- Uploading to Hugging Face Hub: {args.hf_repo} ---")
        try:
            if args.hf_token:
                login(token=args.hf_token)

            api = HfApi()
            api.upload_folder(
                folder_path=output_path,
                repo_id=args.hf_repo,
                repo_type="dataset",
                path_in_repo=f"data/{args.group_name.lower()}_data",
            )
        except Exception as e:
            print(f"Failed to upload to Hugging Face Hub: {e}")


if __name__ == "__main__":
    main()
