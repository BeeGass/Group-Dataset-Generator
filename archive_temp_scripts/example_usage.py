#!/usr/bin/env python3
"""Example usage of the new permutation groups dataset with dynamic filtering."""

from datasets import load_dataset, load_from_disk
from pathlib import Path
import sys

# Check if we're testing locally or from HuggingFace
data_dir = Path("./data")
use_local = data_dir.exists() and any(data_dir.iterdir())

if use_local:
    print("Testing with LOCAL datasets (direct loading)")
else:
    print("Testing with HUGGINGFACE datasets")
    repo_or_path = "BeeGass/permutation-groups"
    extra_args = {}

print("=" * 60)
print("SYMMETRIC GROUPS")
print("=" * 60)

# Load specific symmetric groups by degree
print("\n1. Loading specific symmetric groups by degree:")

if use_local:
    # Load full dataset and filter manually
    try:
        symmetric_dataset = load_from_disk(str(data_dir / "symmetric_superset"))
        s3_only = symmetric_dataset["train"].filter(lambda x: x["group_degree"] == 3)
        print(f"   S3 only (degree=3, order=6) - {len(s3_only)} samples")
    except Exception as e:
        print(f"   S3 failed: {e}")
else:
    try:
        s3_only = load_dataset(
            repo_or_path,
            name="symmetric",
            min_degree=3,
            max_degree=3,
            trust_remote_code=True,
            **extra_args,
        )
        print(f"   S3 only (degree=3, order=6) - {len(s3_only['train'])} samples")
    except Exception as e:
        print(f"   S3 failed: {e}")

if use_local:
    try:
        s4_only = symmetric_dataset["train"].filter(lambda x: x["group_degree"] == 4)
        print(f"   S4 only (degree=4, order=24) - {len(s4_only)} samples")
    except Exception as e:
        print(f"   S4 failed: {e}")
else:
    try:
        s4_only = load_dataset(
            repo_or_path,
            name="symmetric",
            min_degree=4,
            max_degree=4,
            trust_remote_code=True,
            **extra_args,
        )
        print(f"   S4 only (degree=4, order=24) - {len(s4_only['train'])} samples")
    except Exception as e:
        print(f"   S4 failed: {e}")

s5_only = load_dataset(
    "BeeGass/permutation-groups",
    name="symmetric",
    min_degree=5,
    max_degree=5,
    trust_remote_code=True,
)
print("   S5 only (degree=5, order=120)")

s6_only = load_dataset(
    "BeeGass/permutation-groups",
    name="symmetric",
    min_degree=6,
    max_degree=6,
    trust_remote_code=True,
)
print("   S6 only (degree=6, order=720)")

s7_only = load_dataset(
    "BeeGass/permutation-groups",
    name="symmetric",
    min_degree=7,
    max_degree=7,
    trust_remote_code=True,
)
print("   S7 only (degree=7, order=5040)")

s8_only = load_dataset(
    "BeeGass/permutation-groups",
    name="symmetric",
    min_degree=8,
    max_degree=8,
    trust_remote_code=True,
)
print("   S8 only (degree=8, order=40320)")

s9_only = load_dataset(
    "BeeGass/permutation-groups",
    name="symmetric",
    min_degree=9,
    max_degree=9,
    trust_remote_code=True,
)
print("   S9 only (degree=9, order=362880)")

s10_only = load_dataset(
    "BeeGass/permutation-groups",
    name="symmetric",
    min_degree=10,
    max_degree=10,
    trust_remote_code=True,
)
print("   S10 only (degree=10, order=3628800)")

# Load all symmetric groups by order
print("\n2. Loading all symmetric groups (S3-S10):")

if use_local:
    try:
        all_symmetric_train = symmetric_dataset["train"]
        print(
            f"   All symmetric groups S3-S10 loaded - {len(all_symmetric_train)} samples"
        )

        # Show sample distribution by degree
        degrees = {}
        for i in range(min(5000, len(all_symmetric_train))):
            deg = all_symmetric_train[i]["group_degree"]
            degrees[deg] = degrees.get(deg, 0) + 1
        print("   Distribution by degree (first 5000 samples):")
        for deg in sorted(degrees.keys()):
            print(f"     S{deg}: {degrees[deg]} samples")
    except Exception as e:
        print(f"   All symmetric failed: {e}")
else:
    try:
        all_symmetric = load_dataset(
            repo_or_path, name="symmetric", trust_remote_code=True, **extra_args
        )
        print(
            f"   All symmetric groups S3-S10 loaded - {len(all_symmetric['train'])} samples"
        )
    except Exception as e:
        print(f"   All symmetric failed: {e}")

# Load symmetric groups with different sequence lengths
print("\n3. Loading symmetric groups with different sequence lengths:")

if use_local:
    try:
        symmetric_short = symmetric_dataset["train"].filter(
            lambda x: x["sequence_length"] <= 32
        )
        print(
            f"   Symmetric groups with sequences ≤ 32 - {len(symmetric_short)} samples"
        )
    except Exception as e:
        print(f"   Symmetric short sequences failed: {e}")
else:
    try:
        symmetric_short = load_dataset(
            repo_or_path,
            name="symmetric",
            max_len=32,
            trust_remote_code=True,
            **extra_args,
        )
        print(
            f"   Symmetric groups with sequences ≤ 32 - {len(symmetric_short['train'])} samples"
        )
    except Exception as e:
        print(f"   Symmetric short sequences failed: {e}")

symmetric_medium = load_dataset(
    "BeeGass/permutation-groups",
    name="symmetric",
    min_len=128,
    max_len=256,
    trust_remote_code=True,
)
print("   Symmetric groups with sequences 128-256")

symmetric_long = load_dataset(
    "BeeGass/permutation-groups", name="symmetric", min_len=512, trust_remote_code=True
)
print("   Symmetric groups with sequences ≥ 512")

print("\n" + "=" * 60)
print("ALTERNATING GROUPS")
print("=" * 60)

# Load specific alternating groups by degree
print("\n1. Loading specific alternating groups by degree:")

a3_only = load_dataset(
    "BeeGass/permutation-groups",
    name="alternating",
    min_degree=3,
    max_degree=3,
    trust_remote_code=True,
)
print("   A3 only (degree=3, order=3)")

a4_only = load_dataset(
    "BeeGass/permutation-groups",
    name="alternating",
    min_degree=4,
    max_degree=4,
    trust_remote_code=True,
)
print("   A4 only (degree=4, order=12)")

a5_only = load_dataset(
    "BeeGass/permutation-groups",
    name="alternating",
    min_degree=5,
    max_degree=5,
    trust_remote_code=True,
)
print("   A5 only (degree=5, order=60)")

a6_only = load_dataset(
    "BeeGass/permutation-groups",
    name="alternating",
    min_degree=6,
    max_degree=6,
    trust_remote_code=True,
)
print("   A6 only (degree=6, order=360)")

a7_only = load_dataset(
    "BeeGass/permutation-groups",
    name="alternating",
    min_degree=7,
    max_degree=7,
    trust_remote_code=True,
)
print("   A7 only (degree=7, order=2520)")

a8_only = load_dataset(
    "BeeGass/permutation-groups",
    name="alternating",
    min_degree=8,
    max_degree=8,
    trust_remote_code=True,
)
print("   A8 only (degree=8, order=20160)")

a9_only = load_dataset(
    "BeeGass/permutation-groups",
    name="alternating",
    min_degree=9,
    max_degree=9,
    trust_remote_code=True,
)
print("   A9 only (degree=9, order=181440)")

a10_only = load_dataset(
    "BeeGass/permutation-groups",
    name="alternating",
    min_degree=10,
    max_degree=10,
    trust_remote_code=True,
)
print("   A10 only (degree=10, order=1814400)")

# Load all alternating groups by order
print("\n2. Loading all alternating groups (A3-A10):")

all_alternating = load_dataset(
    "BeeGass/permutation-groups", name="alternating", trust_remote_code=True
)
print("   All alternating groups A3-A10 loaded")

# Load alternating groups with different sequence lengths
print("\n3. Loading alternating groups with different sequence lengths:")

alternating_short = load_dataset(
    "BeeGass/permutation-groups", name="alternating", max_len=64, trust_remote_code=True
)
print("   Alternating groups with sequences ≤ 64")

alternating_medium = load_dataset(
    "BeeGass/permutation-groups",
    name="alternating",
    min_len=256,
    max_len=512,
    trust_remote_code=True,
)
print("   Alternating groups with sequences 256-512")

alternating_long = load_dataset(
    "BeeGass/permutation-groups",
    name="alternating",
    min_len=1024,
    trust_remote_code=True,
)
print("   Alternating groups with sequences ≥ 1024")

print("\n" + "=" * 60)
print("CYCLIC GROUPS")
print("=" * 60)

# Load specific cyclic groups by degree
print("\n1. Loading specific cyclic groups by degree:")

c3_only = load_dataset(
    "BeeGass/permutation-groups",
    name="cyclic",
    min_degree=3,
    max_degree=3,
    trust_remote_code=True,
)
print("   C3 only (degree=3, order=3)")

c5_only = load_dataset(
    "BeeGass/permutation-groups",
    name="cyclic",
    min_degree=5,
    max_degree=5,
    trust_remote_code=True,
)
print("   C5 only (degree=5, order=5)")

c7_only = load_dataset(
    "BeeGass/permutation-groups",
    name="cyclic",
    min_degree=7,
    max_degree=7,
    trust_remote_code=True,
)
print("   C7 only (degree=7, order=7)")

c10_only = load_dataset(
    "BeeGass/permutation-groups",
    name="cyclic",
    min_degree=10,
    max_degree=10,
    trust_remote_code=True,
)
print("   C10 only (degree=10, order=10)")

c15_only = load_dataset(
    "BeeGass/permutation-groups",
    name="cyclic",
    min_degree=15,
    max_degree=15,
    trust_remote_code=True,
)
print("   C15 only (degree=15, order=15)")

c20_only = load_dataset(
    "BeeGass/permutation-groups",
    name="cyclic",
    min_degree=20,
    max_degree=20,
    trust_remote_code=True,
)
print("   C20 only (degree=20, order=20)")

c25_only = load_dataset(
    "BeeGass/permutation-groups",
    name="cyclic",
    min_degree=25,
    max_degree=25,
    trust_remote_code=True,
)
print("   C25 only (degree=25, order=25)")

c30_only = load_dataset(
    "BeeGass/permutation-groups",
    name="cyclic",
    min_degree=30,
    max_degree=30,
    trust_remote_code=True,
)
print("   C30 only (degree=30, order=30)")

# Load all cyclic groups by order
print("\n2. Loading all cyclic groups (C3-C30):")

all_cyclic = load_dataset(
    "BeeGass/permutation-groups", name="cyclic", trust_remote_code=True
)
print("   All cyclic groups C3-C30 loaded")

# Load cyclic groups with different sequence lengths
print("\n3. Loading cyclic groups with different sequence lengths:")

cyclic_short = load_dataset(
    "BeeGass/permutation-groups", name="cyclic", max_len=16, trust_remote_code=True
)
print("   Cyclic groups with sequences ≤ 16")

cyclic_medium = load_dataset(
    "BeeGass/permutation-groups",
    name="cyclic",
    min_len=64,
    max_len=128,
    trust_remote_code=True,
)
print("   Cyclic groups with sequences 64-128")

cyclic_long = load_dataset(
    "BeeGass/permutation-groups", name="cyclic", min_len=512, trust_remote_code=True
)
print("   Cyclic groups with sequences ≥ 512")

print("\n" + "=" * 60)
print("DIHEDRAL GROUPS")
print("=" * 60)

# Load specific dihedral groups by degree
print("\n1. Loading specific dihedral groups by degree:")

d3_only = load_dataset(
    "BeeGass/permutation-groups",
    name="dihedral",
    min_degree=3,
    max_degree=3,
    trust_remote_code=True,
)
print("   D3 only (degree=3, order=6)")

d4_only = load_dataset(
    "BeeGass/permutation-groups",
    name="dihedral",
    min_degree=4,
    max_degree=4,
    trust_remote_code=True,
)
print("   D4 only (degree=4, order=8)")

d5_only = load_dataset(
    "BeeGass/permutation-groups",
    name="dihedral",
    min_degree=5,
    max_degree=5,
    trust_remote_code=True,
)
print("   D5 only (degree=5, order=10)")

d7_only = load_dataset(
    "BeeGass/permutation-groups",
    name="dihedral",
    min_degree=7,
    max_degree=7,
    trust_remote_code=True,
)
print("   D7 only (degree=7, order=14)")

d10_only = load_dataset(
    "BeeGass/permutation-groups",
    name="dihedral",
    min_degree=10,
    max_degree=10,
    trust_remote_code=True,
)
print("   D10 only (degree=10, order=20)")

d12_only = load_dataset(
    "BeeGass/permutation-groups",
    name="dihedral",
    min_degree=12,
    max_degree=12,
    trust_remote_code=True,
)
print("   D12 only (degree=12, order=24)")

d15_only = load_dataset(
    "BeeGass/permutation-groups",
    name="dihedral",
    min_degree=15,
    max_degree=15,
    trust_remote_code=True,
)
print("   D15 only (degree=15, order=30)")

d20_only = load_dataset(
    "BeeGass/permutation-groups",
    name="dihedral",
    min_degree=20,
    max_degree=20,
    trust_remote_code=True,
)
print("   D20 only (degree=20, order=40)")

# Load all dihedral groups by order
print("\n2. Loading all dihedral groups (D3-D20):")

all_dihedral = load_dataset(
    "BeeGass/permutation-groups", name="dihedral", trust_remote_code=True
)
print("   All dihedral groups D3-D20 loaded")

# Load dihedral groups with different sequence lengths
print("\n3. Loading dihedral groups with different sequence lengths:")

dihedral_short = load_dataset(
    "BeeGass/permutation-groups", name="dihedral", max_len=32, trust_remote_code=True
)
print("   Dihedral groups with sequences ≤ 32")

dihedral_medium = load_dataset(
    "BeeGass/permutation-groups",
    name="dihedral",
    min_len=128,
    max_len=256,
    trust_remote_code=True,
)
print("   Dihedral groups with sequences 128-256")

dihedral_long = load_dataset(
    "BeeGass/permutation-groups", name="dihedral", min_len=1024, trust_remote_code=True
)
print("   Dihedral groups with sequences ≥ 1024")

# Add a quick test for the new groups
print("\n" + "=" * 60)
print("NEW GROUPS QUICK TEST")
print("=" * 60)

new_groups = [
    "klein",
    "quaternion",
    "elementary_abelian",
    "psl",
    "frobenius",
    "mathieu",
]

if use_local:
    for group in new_groups:
        try:
            dataset = load_from_disk(str(data_dir / f"{group}_superset"))
            print(
                f"✓ {group.upper()}: {len(dataset['train'])} train, {len(dataset['test'])} test samples"
            )

            # Show first sample
            if len(dataset["train"]) > 0:
                sample = dataset["train"][0]
                print(
                    f"    First sample: degree={sample['group_degree']}, order={sample['group_order']}"
                )
        except Exception as e:
            print(f"✗ {group.upper()}: {e}")
else:
    for group in new_groups:
        try:
            dataset = load_dataset(
                repo_or_path, name=group, trust_remote_code=True, **extra_args
            )
            print(
                f"✓ {group.upper()}: {len(dataset['train'])} train, {len(dataset['test'])} test samples"
            )
        except Exception as e:
            print(f"✗ {group.upper()}: {e}")

print("\n" + "=" * 60)
print("All examples complete!")
print("=" * 60)
