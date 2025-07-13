#!/usr/bin/env python3
"""
Master script to generate all permutation group datasets.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set

# Import all generators
from gdg.generators.symmetric import SymmetricGroupGenerator
from gdg.generators.alternating import AlternatingGroupGenerator
from gdg.generators.cyclic import CyclicGroupGenerator
from gdg.generators.dihedral import DihedralGroupGenerator
from gdg.generators.quaternion import QuaternionGroupGenerator
from gdg.generators.klein import KleinFourGroupGenerator
from gdg.generators.elementary_abelian import ElementaryAbelianGroupGenerator
from gdg.generators.frobenius import FrobeniusGroupGenerator
from gdg.generators.psl import PSLGroupGenerator
from gdg.generators.mathieu import MathieuGroupGenerator


# Define all available groups and their complexity classes
TC0_GROUPS = {
    # Solvable groups
    's3', 's4', 'a3', 'a4',
    'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10',
    'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20',
    'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30',
    'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10',
    'd11', 'd12', 'd13', 'd14', 'd15', 'd16', 'd17', 'd18', 'd19', 'd20',
    'v4', 'q8', 'q16', 'q32',
    'z2_1', 'z2_2', 'z2_3', 'z2_4', 'z2_5',
    'z3_1', 'z3_2', 'z3_3', 'z3_4',
    'z5_1', 'z5_2', 'z5_3', 'z5_4',
    'f20', 'f21', 'psl2_2', 'psl2_3'
}

NC1_GROUPS = {
    # Non-solvable groups
    's5', 's6', 's7', 's8', 's9',
    'a5', 'a6', 'a7', 'a8', 'a9',
    'psl2_4', 'psl2_5', 'psl2_7', 'psl2_8', 'psl2_9', 'psl2_11',
    'psl3_2', 'psl3_3', 'psl3_4', 'psl3_5',
    'm11', 'm12'
}

ALL_GROUPS = TC0_GROUPS | NC1_GROUPS


def parse_group_spec(spec: str) -> Set[str]:
    """Parse a group specification and return set of group names."""
    spec_lower = spec.lower()
    
    if spec_lower == 'all':
        return ALL_GROUPS
    elif spec_lower == 'tc0':
        return TC0_GROUPS
    elif spec_lower == 'nc1':
        return NC1_GROUPS
    else:
        # Individual group name
        return {spec_lower} if spec_lower in ALL_GROUPS else set()


def generate_groups(groups: Set[str], output_dir: str, seed: int = 42):
    """Generate datasets for the specified groups."""
    # Group the requested groups by type
    symmetric_degrees = []
    alternating_degrees = []
    cyclic_degrees = []
    dihedral_degrees = []
    quaternion_powers = []
    elementary_abelian_groups = []
    frobenius_groups = []
    psl_groups = []
    mathieu_groups = []
    
    for group in groups:
        if group.startswith('s') and group[1:].isdigit():
            symmetric_degrees.append(int(group[1:]))
        elif group.startswith('a') and group[1:].isdigit():
            alternating_degrees.append(int(group[1:]))
        elif group.startswith('c') and group[1:].isdigit():
            cyclic_degrees.append(int(group[1:]))
        elif group.startswith('d') and group[1:].isdigit():
            dihedral_degrees.append(int(group[1:]))
        elif group == 'q8':
            quaternion_powers.append(3)
        elif group == 'q16':
            quaternion_powers.append(4)
        elif group == 'q32':
            quaternion_powers.append(5)
        elif group.startswith('z') and '_' in group:
            parts = group.split('_')
            p = int(parts[0][1:])
            k = int(parts[1])
            elementary_abelian_groups.append((p, k))
        elif group.startswith('f') and group[1:].isdigit():
            frobenius_groups.append(int(group[1:]))
        elif group.startswith('psl'):
            if '_' in group:
                parts = group.split('_')
                n = int(parts[0][3:])
                q = int(parts[1])
                psl_groups.append((n, q))
        elif group.startswith('m') and group[1:].isdigit():
            mathieu_groups.append(int(group[1:]))
        elif group == 'v4':
            # Klein four-group is handled separately
            pass
    
    success_count = 0
    total_count = 0
    
    # Generate each type of group
    if symmetric_degrees:
        print(f"\nGenerating symmetric groups: {symmetric_degrees}")
        gen = SymmetricGroupGenerator(seed)
        for degree in sorted(symmetric_degrees):
            total_count += 1
            try:
                dataset_dict, metadata = gen.generate_dataset(
                    params={"n": degree},
                    num_train_samples=100000,
                    num_test_samples=20000,
                    output_dir=Path(output_dir) / f"s{degree}_data"
                )
                print(f"✓ Generated S{degree}")
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to generate S{degree}: {e}")
    
    if alternating_degrees:
        print(f"\nGenerating alternating groups: {alternating_degrees}")
        gen = AlternatingGroupGenerator(seed)
        for degree in sorted(alternating_degrees):
            total_count += 1
            try:
                dataset_dict, metadata = gen.generate_dataset(
                    params={"n": degree},
                    num_train_samples=100000,
                    num_test_samples=20000,
                    output_dir=Path(output_dir) / f"a{degree}_data"
                )
                print(f"✓ Generated A{degree}")
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to generate A{degree}: {e}")
    
    if cyclic_degrees:
        print(f"\nGenerating cyclic groups: {cyclic_degrees}")
        gen = CyclicGroupGenerator(seed)
        for degree in sorted(cyclic_degrees):
            total_count += 1
            try:
                dataset_dict, metadata = gen.generate_dataset(
                    params={"n": degree},
                    num_train_samples=100000,
                    num_test_samples=20000,
                    output_dir=Path(output_dir) / f"c{degree}_data"
                )
                print(f"✓ Generated C{degree}")
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to generate C{degree}: {e}")
    
    if dihedral_degrees:
        print(f"\nGenerating dihedral groups: {dihedral_degrees}")
        gen = DihedralGroupGenerator(seed)
        for degree in sorted(dihedral_degrees):
            total_count += 1
            try:
                dataset_dict, metadata = gen.generate_dataset(
                    params={"n": degree},
                    num_train_samples=100000,
                    num_test_samples=20000,
                    output_dir=Path(output_dir) / f"d{degree}_data"
                )
                print(f"✓ Generated D{degree}")
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to generate D{degree}: {e}")
    
    if quaternion_powers:
        print(f"\nGenerating quaternion groups: {[2**p for p in quaternion_powers]}")
        gen = QuaternionGroupGenerator(seed)
        for power in sorted(quaternion_powers):
            total_count += 1
            try:
                dataset_dict, metadata = gen.generate_dataset(
                    params={"k": power},
                    num_train_samples=100000,
                    num_test_samples=20000,
                    output_dir=Path(output_dir) / f"q{2**power}_data"
                )
                print(f"✓ Generated Q{2**power}")
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to generate Q{2**power}: {e}")
    
    if 'v4' in groups:
        print(f"\nGenerating Klein four-group")
        gen = KleinFourGroupGenerator(seed)
        total_count += 1
        try:
            dataset_dict, metadata = gen.generate_dataset(
                params={},
                num_train_samples=100000,
                num_test_samples=20000,
                output_dir=Path(output_dir) / "v4_data"
            )
            print(f"✓ Generated V4")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to generate V4: {e}")
    
    if elementary_abelian_groups:
        print(f"\nGenerating elementary abelian groups: {elementary_abelian_groups}")
        gen = ElementaryAbelianGroupGenerator(seed)
        for p, k in sorted(elementary_abelian_groups):
            total_count += 1
            try:
                dataset_dict, metadata = gen.generate_dataset(
                    params={"p": p, "k": k},
                    num_train_samples=100000,
                    num_test_samples=20000,
                    output_dir=Path(output_dir) / f"z{p}_{k}_data"
                )
                print(f"✓ Generated Z{p}^{k}")
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to generate Z{p}^{k}: {e}")
    
    if frobenius_groups:
        print(f"\nGenerating Frobenius groups: {frobenius_groups}")
        gen = FrobeniusGroupGenerator(seed)
        for order in sorted(frobenius_groups):
            total_count += 1
            try:
                dataset_dict, metadata = gen.generate_dataset(
                    params={"n": order},
                    num_train_samples=100000,
                    num_test_samples=20000,
                    output_dir=Path(output_dir) / f"f{order}_data"
                )
                print(f"✓ Generated F{order}")
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to generate F{order}: {e}")
    
    if psl_groups:
        print(f"\nGenerating PSL groups: {psl_groups}")
        gen = PSLGroupGenerator(seed)
        # Import the helper function from PSL module
        from gdg.generators.psl import get_prime_power
        
        for dim, q in sorted(psl_groups):
            total_count += 1
            try:
                # Convert q to p and n where q = p^n
                p_n_pair = get_prime_power(q)
                if not p_n_pair:
                    print(f"✗ Error: q={q} is not a valid prime power for PSL({dim},{q})")
                    continue
                p, n = p_n_pair
                
                dataset_dict, metadata = gen.generate_dataset(
                    params={"dim": dim, "p": p, "n": n},
                    num_train_samples=100000,
                    num_test_samples=20000,
                    output_dir=Path(output_dir) / f"psl{dim}_{q}_data"
                )
                print(f"✓ Generated PSL({dim},{q})")
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to generate PSL({dim},{q}): {e}")
    
    if mathieu_groups:
        print(f"\nGenerating Mathieu groups: {mathieu_groups}")
        gen = MathieuGroupGenerator(seed)
        for degree in sorted(mathieu_groups):
            total_count += 1
            try:
                dataset_dict, metadata = gen.generate_dataset(
                    params={"n": degree},
                    num_train_samples=100000,
                    num_test_samples=20000,
                    output_dir=Path(output_dir) / f"m{degree}_data"
                )
                print(f"✓ Generated M{degree}")
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to generate M{degree}: {e}")
    
    return success_count, total_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate permutation group datasets"
    )
    parser.add_argument(
        "groups",
        nargs="*",
        default=["all"],
        help="Groups to generate: specific groups (e.g., s5, c10), 'tc0', 'nc1', or 'all' (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Output directory for all datasets (default: datasets)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()
    
    # Parse all group specifications
    requested_groups = set()
    for spec in args.groups:
        groups = parse_group_spec(spec)
        if not groups:
            print(f"Warning: Unknown group specification '{spec}'")
        requested_groups.update(groups)
    
    if not requested_groups:
        print("Error: No valid groups specified")
        sys.exit(1)
    
    print(f"Generating {len(requested_groups)} groups")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate the groups
    success_count, total_count = generate_groups(
        requested_groups, 
        str(output_path), 
        args.seed
    )
    
    print(f"\n{'=' * 60}")
    print(f"Generation complete: {success_count}/{total_count} successful")
    print("=" * 60)


if __name__ == "__main__":
    main()