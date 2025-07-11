#!/usr/bin/env python3
"""
Generate superset permutation datasets that contain multiple subgroups.
Each dataset includes metadata for filtering by degree and order.
"""

import argparse
import json
import random
from pathlib import Path
import os
from collections import defaultdict
import gc

from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi, login
from sympy.combinatorics import Permutation
from sympy.combinatorics.named_groups import (
    SymmetricGroup,
    AlternatingGroup,
    CyclicGroup,
    DihedralGroup,
)
from sympy.combinatorics.perm_groups import PermutationGroup
from tqdm import tqdm
import numpy as np


def get_klein_four_generators():
    """Get generators for Klein four-group V4 as permutations on 4 points."""
    # V4 = {e, (12)(34), (13)(24), (14)(23)}
    a = Permutation([[0, 1], [2, 3]])  # (12)(34)
    b = Permutation([[0, 2], [1, 3]])  # (13)(24)
    return [a, b]


def get_quaternion_generators(n=8):
    """Get generators for quaternion group Q_n."""
    if n == 8:
        # Q8 on 8 points
        a = Permutation([1, 2, 3, 0, 5, 6, 7, 4])  # (0 1 2 3)(4 5 6 7)
        b = Permutation([3, 0, 1, 2, 6, 7, 4, 5])  # (0 3 2 1)(4 6)(5 7)
    elif n == 16:
        # Q16 on 16 points - generalized quaternion
        a = Permutation(list(range(1, 8)) + [0] + list(range(9, 16)) + [8])
        b = Permutation([0, 7, 6, 5, 4, 3, 2, 1, 8, 15, 14, 13, 12, 11, 10, 9])
    elif n == 32:
        # Q32 on 32 points - generalized quaternion
        a = Permutation(list(range(1, 16)) + [0] + list(range(17, 32)) + [16])
        b = Permutation([0] + list(range(15, 0, -1)) + [16] + list(range(31, 16, -1)))
    else:
        raise ValueError(f"Quaternion group Q{n} not implemented")
    return [a, b]


def get_elementary_abelian_generators(p, k):
    """Get generators for elementary abelian group Z_p^k."""
    generators = []
    degree = p**k

    # Create k independent generators of order p
    for i in range(k):
        cycles = []
        for j in range(p**i):
            cycle = []
            for m in range(p):
                cycle.append(j + m * (p**i) + (p ** (i + 1)) * (j // (p**i)))
            if len(cycle) > 1:
                cycles.append(cycle)
        if cycles:
            generators.append(Permutation(cycles))

    return generators


def get_psl_generators(n, q):
    """Get generators for PSL(n,q) groups."""
    if n == 2 and q == 5:
        # PSL(2,5) ≅ A5 on 6 points
        a = Permutation([1, 2, 0, 4, 5, 3])  # (0 1 2)(3 4 5)
        b = Permutation([0, 2, 3, 1, 5, 4])  # (1 2 3)(4 5)
        return [a, b]
    elif n == 2 and q == 7:
        # PSL(2,7) on 8 points
        a = Permutation([1, 2, 3, 4, 5, 6, 0, 7])  # 7-cycle
        b = Permutation([0, 3, 1, 5, 2, 6, 4, 7])  # specific permutation
        return [a, b]
    else:
        raise ValueError(f"PSL({n},{q}) not implemented")


def get_frobenius_generators(order):
    """Get generators for Frobenius groups."""
    if order == 20:
        # F20 = C5 ⋊ C4 on 5 points
        a = Permutation([1, 2, 3, 4, 0])  # 5-cycle
        b = Permutation([0, 2, 4, 1, 3])  # order 4
        return [a, b]
    elif order == 21:
        # F21 = C7 ⋊ C3 on 7 points
        a = Permutation([1, 2, 3, 4, 5, 6, 0])  # 7-cycle
        b = Permutation([0, 2, 4, 6, 1, 3, 5])  # order 3
        return [a, b]
    else:
        raise ValueError(f"Frobenius group F{order} not implemented")


def get_mathieu_generators(n):
    """Get generators for Mathieu groups."""
    if n == 11:
        # M11 on 11 points
        a = Permutation([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0])  # 11-cycle
        b = Permutation([0, 1, 5, 8, 6, 9, 3, 7, 10, 4, 2])
        return [a, b]
    elif n == 12:
        # M12 on 12 points
        a = Permutation([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0])  # 12-cycle
        b = Permutation([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])  # reversal
        c = Permutation([0, 1, 5, 9, 3, 11, 2, 10, 4, 8, 7, 6])
        return [a, b, c]
    else:
        raise ValueError(f"Mathieu group M{n} not implemented")


def get_subgroup_info(group_type, max_degree):
    """Get information about all subgroups up to max_degree."""
    subgroups = []

    if group_type == "symmetric":
        for n in range(3, max_degree + 1):
            subgroups.append(
                {"degree": n, "order": int(SymmetricGroup(n).order()), "name": f"S{n}"}
            )
    elif group_type == "alternating":
        for n in range(3, max_degree + 1):
            subgroups.append(
                {
                    "degree": n,
                    "order": int(AlternatingGroup(n).order()),
                    "name": f"A{n}",
                }
            )
    elif group_type == "cyclic":
        for n in range(3, max_degree + 1):
            subgroups.append(
                {
                    "degree": n,
                    "order": n,  # Cyclic group Cn has order n
                    "name": f"C{n}",
                }
            )
    elif group_type == "dihedral":
        for n in range(3, max_degree + 1):
            subgroups.append(
                {
                    "degree": n,
                    "order": 2 * n,  # Dihedral group Dn has order 2n
                    "name": f"D{n}",
                }
            )
    elif group_type == "klein":
        # Klein four-group (only one size)
        subgroups.append({"degree": 4, "order": 4, "name": "V4"})
    elif group_type == "quaternion":
        # Quaternion groups Q8, Q16, Q32, etc.
        for k in [8, 16, 32]:
            if k <= max_degree:
                subgroups.append({"degree": k, "order": k, "name": f"Q{k}"})
    elif group_type == "elementary_abelian":
        # Elementary abelian groups Z_p^k
        for p in [2, 3, 5]:  # primes
            k = 1
            while p**k <= max_degree:
                subgroups.append(
                    {
                        "degree": p**k,
                        "order": p**k,
                        "name": f"Z_{p}^{k}",
                        "params": (p, k),
                    }
                )
                k += 1
    elif group_type == "psl":
        # PSL groups
        psl_groups = [(2, 5, 6, 60), (2, 7, 8, 168)]  # (n, q, degree, order)
        for n, q, deg, ord in psl_groups:
            if deg <= max_degree:
                subgroups.append(
                    {
                        "degree": deg,
                        "order": ord,
                        "name": f"PSL({n},{q})",
                        "params": (n, q),
                    }
                )
    elif group_type == "frobenius":
        # Frobenius groups
        frob_groups = [(20, 5, 20), (21, 7, 21)]  # (order, degree, order)
        for ord, deg, _ in frob_groups:
            if deg <= max_degree:
                subgroups.append(
                    {"degree": deg, "order": ord, "name": f"F{ord}", "params": ord}
                )
    elif group_type == "mathieu":
        # Mathieu groups
        mathieu_groups = [(11, 11, 7920), (12, 12, 95040)]  # (n, degree, order)
        for n, deg, ord in mathieu_groups:
            if deg <= max_degree:
                subgroups.append(
                    {"degree": deg, "order": ord, "name": f"M{n}", "params": n}
                )

    return subgroups


def generate_subgroup_elements_lazy(group_type, degree, params=None, batch_size=1000):
    """Generate elements of a specific subgroup in batches to save memory."""
    if group_type == "symmetric":
        group = SymmetricGroup(degree)
    elif group_type == "alternating":
        group = AlternatingGroup(degree)
    elif group_type == "cyclic":
        group = CyclicGroup(degree)
    elif group_type == "dihedral":
        group = DihedralGroup(degree)
    elif group_type == "klein":
        generators = get_klein_four_generators()
        group = PermutationGroup(generators)
    elif group_type == "quaternion":
        generators = get_quaternion_generators(degree)
        group = PermutationGroup(generators)
    elif group_type == "elementary_abelian":
        if params:
            p, k = params
            generators = get_elementary_abelian_generators(p, k)
            group = PermutationGroup(generators)
        else:
            raise ValueError("Elementary abelian groups need params (p, k)")
    elif group_type == "psl":
        if params:
            n, q = params
            generators = get_psl_generators(n, q)
            group = PermutationGroup(generators)
        else:
            raise ValueError("PSL groups need params (n, q)")
    elif group_type == "frobenius":
        if params:
            generators = get_frobenius_generators(params)
            group = PermutationGroup(generators)
        else:
            raise ValueError("Frobenius groups need order param")
    elif group_type == "mathieu":
        if params:
            generators = get_mathieu_generators(params)
            group = PermutationGroup(generators)
        else:
            raise ValueError("Mathieu groups need n param")
    else:
        raise ValueError(f"Unknown group type: {group_type}")

    # For large groups, use a more memory-efficient approach
    if int(group.order()) > 10000:
        # Use generators to create elements on demand
        elements = []
        seen = set()
        queue = [Permutation(degree - 1)]  # Identity

        while queue and len(elements) < int(group.order()):
            current = queue.pop(0)
            current_str = str(current.array_form)

            if current_str not in seen:
                seen.add(current_str)
                elements.append(current)

                # Generate new elements
                for gen in group.generators:
                    new_elem = gen * current
                    if str(new_elem.array_form) not in seen:
                        queue.append(new_elem)

            # Yield batch when ready
            if len(elements) >= batch_size:
                yield elements[:batch_size], group
                elements = elements[batch_size:]

        # Yield remaining elements
        if elements:
            yield elements, group
    else:
        # For smaller groups, generate all at once
        yield list(group.generate()), group


def generate_permutation_mappings_efficient(group_type, max_degree):
    """Generate mappings for all subgroups up to max_degree with memory efficiency."""
    all_mappings = {}

    subgroups = get_subgroup_info(group_type, max_degree)

    for subgroup in subgroups:
        degree = subgroup["degree"]
        params = subgroup.get("params", None)

        # For large groups, store only essential data
        if subgroup["order"] > 10000:
            # Store generators instead of all elements
            if group_type == "symmetric":
                group = SymmetricGroup(degree)
            elif group_type == "alternating":
                group = AlternatingGroup(degree)
            else:
                # Get first batch to establish the group
                for elements, group in generate_subgroup_elements_lazy(
                    group_type, degree, params
                ):
                    break

            all_mappings[degree] = {
                "type": "lazy",
                "generators": group.generators,
                "order": subgroup["order"],
                "name": subgroup["name"],
                "group_type": group_type,
                "params": params,
            }
        else:
            # For smaller groups, keep the full mapping
            elements = []
            for batch, _ in generate_subgroup_elements_lazy(group_type, degree, params):
                elements.extend(batch)

            perm_to_id = {str(p.array_form): i for i, p in enumerate(elements)}
            id_to_perm = {i: p for i, p in enumerate(elements)}

            all_mappings[degree] = {
                "type": "full",
                "perm_to_id": perm_to_id,
                "id_to_perm": id_to_perm,
                "order": subgroup["order"],
                "name": subgroup["name"],
            }

    return all_mappings


def get_random_element_from_group(group_type, degree, order, params=None):
    """Get a random element from a group efficiently."""
    # For large groups, use random walk on generators
    if group_type == "symmetric":
        # Random permutation
        perm_array = list(range(degree))
        random.shuffle(perm_array)
        return Permutation(perm_array)
    elif group_type == "alternating":
        # Random even permutation
        while True:
            perm_array = list(range(degree))
            random.shuffle(perm_array)
            perm = Permutation(perm_array)
            if perm.is_even:
                return perm
    elif group_type == "cyclic":
        # Random power of generator
        k = random.randint(0, degree - 1)
        return Permutation([(i + k) % degree for i in range(degree)])
    elif group_type == "dihedral":
        # Random element: r^i or sr^i
        r = random.randint(0, degree - 1)
        flip = random.choice([True, False])
        if flip:
            # Reflection followed by rotation
            return Permutation([degree - 1 - (i - r) % degree for i in range(degree)])
        else:
            # Just rotation
            return Permutation([(i + r) % degree for i in range(degree)])
    else:
        # For other groups, use generator walk
        if group_type == "klein":
            generators = get_klein_four_generators()
        elif group_type == "quaternion":
            generators = get_quaternion_generators(degree)
        elif group_type == "elementary_abelian":
            p, k = params
            generators = get_elementary_abelian_generators(p, k)
        elif group_type == "psl":
            n, q = params
            generators = get_psl_generators(n, q)
        elif group_type == "frobenius":
            generators = get_frobenius_generators(params)
        elif group_type == "mathieu":
            generators = get_mathieu_generators(params)
        else:
            raise ValueError(f"Unknown group type: {group_type}")

        # Random walk on generators
        result = Permutation(degree - 1)  # Identity
        steps = random.randint(5, 20)
        for _ in range(steps):
            gen = random.choice(generators)
            if random.choice([True, False]):
                result = gen * result
            else:
                result = result * gen
        return result


def generate_composition_sample_efficient(
    mappings, group_type, min_degree, max_degree, min_len, max_len, samples_per_degree
):
    """Generate composition samples with metadata, using memory-efficient methods."""
    samples = []

    for degree in tqdm(range(min_degree, max_degree + 1), desc="Generating samples"):
        if degree not in mappings:
            continue

        mapping = mappings[degree]
        group_order = mapping["order"]

        # Generate samples for this degree
        for _ in range(samples_per_degree[degree]):
            seq_len = random.randint(min_len, max_len)

            if mapping["type"] == "full":
                # Use existing mapping
                id_to_perm = mapping["id_to_perm"]
                perm_to_id = mapping["perm_to_id"]

                input_ids = [
                    random.randint(0, len(id_to_perm) - 1) for _ in range(seq_len)
                ]

                # Compute composition (right to left)
                composed_perm = Permutation(degree - 1)  # Identity
                for id in reversed(input_ids):
                    composed_perm = id_to_perm[id] * composed_perm

                target_id = perm_to_id[str(composed_perm.array_form)]
            else:
                # For lazy groups, work directly with permutations
                params = mapping.get("params", None)

                # Generate random permutations
                input_perms = []
                for _ in range(seq_len):
                    input_perms.append(
                        get_random_element_from_group(
                            group_type, degree, group_order, params
                        )
                    )

                # Compute composition
                composed_perm = Permutation(degree - 1)  # Identity
                for perm in reversed(input_perms):
                    composed_perm = perm * composed_perm

                # For large groups, use hash as ID
                input_ids = [hash(str(p.array_form)) % group_order for p in input_perms]
                target_id = hash(str(composed_perm.array_form)) % group_order

            samples.append(
                {
                    "input_sequence": " ".join(map(str, input_ids)),
                    "target": str(target_id),
                    "group_type": group_type,
                    "group_degree": degree,
                    "group_order": group_order,
                    "sequence_length": seq_len,
                }
            )

            # Periodically clean up memory
            if len(samples) % 10000 == 0:
                gc.collect()

    return samples


def calculate_samples_distribution(subgroups, total_samples):
    """Distribute samples across subgroups proportionally."""
    # Use sqrt of order for more balanced distribution
    weights = [
        min(s["order"] ** 0.5, 1000) for s in subgroups
    ]  # Cap weight to avoid too much bias
    total_weight = sum(weights)

    distribution = {}
    allocated = 0

    for i, subgroup in enumerate(subgroups):
        degree = subgroup["degree"]
        # Allocate proportionally with minimum samples
        samples = max(100, int(total_samples * weights[i] / total_weight))
        distribution[degree] = samples
        allocated += samples

    # Distribute remaining samples
    remaining = total_samples - allocated
    if remaining > 0:
        # Add to smaller groups first
        sorted_degrees = sorted(distribution.keys(), key=lambda d: distribution[d])
        for i in range(remaining):
            degree = sorted_degrees[i % len(sorted_degrees)]
            distribution[degree] += 1
    elif remaining < 0:
        # Remove from larger groups first
        sorted_degrees = sorted(
            distribution.keys(), key=lambda d: distribution[d], reverse=True
        )
        for i in range(-remaining):
            degree = sorted_degrees[i % len(sorted_degrees)]
            distribution[degree] = max(10, distribution[degree] - 1)

    return distribution


def save_dataset_in_chunks(
    samples, features, output_path, test_split_size=0.2, chunk_size=50000
):
    """Save dataset in chunks to manage memory."""
    # Shuffle samples
    random.shuffle(samples)

    # Split into train/test
    split_idx = int(len(samples) * (1 - test_split_size))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    # Process in chunks
    train_datasets = []
    test_datasets = []

    # Process train samples
    for i in range(0, len(train_samples), chunk_size):
        chunk = train_samples[i : i + chunk_size]
        ds = Dataset.from_list(chunk, features=features)
        train_datasets.append(ds)
        gc.collect()

    # Process test samples
    for i in range(0, len(test_samples), chunk_size):
        chunk = test_samples[i : i + chunk_size]
        ds = Dataset.from_list(chunk, features=features)
        test_datasets.append(ds)
        gc.collect()

    # Concatenate datasets
    from datasets import concatenate_datasets

    if len(train_datasets) > 1:
        train_dataset = concatenate_datasets(train_datasets)
    else:
        train_dataset = (
            train_datasets[0]
            if train_datasets
            else Dataset.from_list([], features=features)
        )

    if len(test_datasets) > 1:
        test_dataset = concatenate_datasets(test_datasets)
    else:
        test_dataset = (
            test_datasets[0]
            if test_datasets
            else Dataset.from_list([], features=features)
        )

    # Create dataset dict
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Save
    dataset_dict.save_to_disk(output_path)

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate superset permutation datasets with subgroup metadata."
    )
    parser.add_argument(
        "--group-type",
        type=str,
        required=True,
        choices=[
            "symmetric",
            "alternating",
            "cyclic",
            "dihedral",
            "klein",
            "quaternion",
            "elementary_abelian",
            "psl",
            "frobenius",
            "mathieu",
        ],
        help="Type of permutation group",
    )
    parser.add_argument(
        "--max-degree", type=int, required=True, help="Maximum degree for the superset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200000,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--min-len", type=int, default=3, help="Minimum sequence length"
    )
    parser.add_argument(
        "--max-len", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--test-split-size", type=float, default=0.2, help="Fraction for test set"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data", help="Output directory"
    )
    parser.add_argument("--hf-repo", type=str, help="HuggingFace repository ID")

    args = parser.parse_args()

    print(f"Generating {args.group_type} superset up to degree {args.max_degree}")

    # Get subgroup information
    subgroups = get_subgroup_info(args.group_type, args.max_degree)
    print(f"Will include {len(subgroups)} subgroups:")
    for sg in subgroups:
        print(f"  - {sg['name']}: degree={sg['degree']}, order={sg['order']}")

    # Generate mappings for all subgroups
    print("\nGenerating permutation mappings...")
    mappings = generate_permutation_mappings_efficient(args.group_type, args.max_degree)

    # Calculate sample distribution
    samples_dist = calculate_samples_distribution(subgroups, args.num_samples)
    print("\nSample distribution:")
    for degree, count in sorted(samples_dist.items()):
        print(f"  - Degree {degree}: {count} samples")

    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    samples = generate_composition_sample_efficient(
        mappings,
        args.group_type,
        3,  # min_degree
        args.max_degree,
        args.min_len,
        args.max_len,
        samples_dist,
    )

    # Create dataset with proper features
    features = Features(
        {
            "input_sequence": Value("string"),
            "target": Value("string"),
            "group_type": Value("string"),
            "group_degree": Value("int32"),
            "group_order": Value("int32"),
            "sequence_length": Value("int32"),
        }
    )

    # Save dataset efficiently
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nSaving dataset...")
    dataset_dict = save_dataset_in_chunks(
        samples, features, output_path, args.test_split_size
    )

    # Save lightweight metadata (don't store all permutations for large groups)
    metadata = {
        "group_type": args.group_type,
        "max_degree": args.max_degree,
        "subgroups": subgroups,
        "mappings_info": {
            degree: {"order": int(m["order"]), "name": m["name"], "type": m["type"]}
            for degree, m in mappings.items()
        },
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to: {output_path}")
    print(f"Train samples: {len(dataset_dict['train'])}")
    print(f"Test samples: {len(dataset_dict['test'])}")

    # Clean up memory
    del samples
    del mappings
    gc.collect()

    # Upload to HuggingFace if requested
    if args.hf_repo:
        print(f"\nUploading to HuggingFace: {args.hf_repo}")
        api = HfApi()
        api.upload_folder(
            folder_path=output_path,
            repo_id=args.hf_repo,
            repo_type="dataset",
            path_in_repo=f"data/{args.group_type}_superset",
        )
        print("✓ Upload complete")


if __name__ == "__main__":
    main()
