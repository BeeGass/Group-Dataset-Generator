#!/usr/bin/env python3
"""Configuration for all supported permutation groups."""

# Length variants we support (powers of 2)
LENGTHS = [4, 8, 16, 32, 64, 128, 256, 512]

# All groups with their properties
GROUPS = {
    # Symmetric Groups
    "S3": {"type": "Symmetric", "degree": 3, "order": 6, "samples": 10000},
    "S4": {"type": "Symmetric", "degree": 4, "order": 24, "samples": 20000},
    "S5": {"type": "Symmetric", "degree": 5, "order": 120, "samples": 50000},
    "S6": {"type": "Symmetric", "degree": 6, "order": 720, "samples": 80000},
    "S7": {"type": "Symmetric", "degree": 7, "order": 5040, "samples": 100000},
    # Alternating Groups
    "A3": {"type": "Alternating", "degree": 3, "order": 3, "samples": 5000},
    "A4": {"type": "Alternating", "degree": 4, "order": 12, "samples": 15000},
    "A5": {"type": "Alternating", "degree": 5, "order": 60, "samples": 30000},
    "A6": {"type": "Alternating", "degree": 6, "order": 360, "samples": 50000},
    "A7": {"type": "Alternating", "degree": 7, "order": 2520, "samples": 80000},
    # Cyclic Groups (C notation)
    "C3": {"type": "Cyclic", "degree": 3, "order": 3, "samples": 5000},
    "C4": {"type": "Cyclic", "degree": 4, "order": 4, "samples": 8000},
    "C5": {"type": "Cyclic", "degree": 5, "order": 5, "samples": 10000},
    "C6": {"type": "Cyclic", "degree": 6, "order": 6, "samples": 12000},
    "C7": {"type": "Cyclic", "degree": 7, "order": 7, "samples": 14000},
    "C8": {"type": "Cyclic", "degree": 8, "order": 8, "samples": 16000},
    "C10": {"type": "Cyclic", "degree": 10, "order": 10, "samples": 20000},
    "C12": {"type": "Cyclic", "degree": 12, "order": 12, "samples": 24000},
    # Cyclic Groups (Z notation - same as C)
    "Z3": {"type": "Cyclic", "degree": 3, "order": 3, "samples": 5000},
    "Z4": {"type": "Cyclic", "degree": 4, "order": 4, "samples": 8000},
    "Z5": {"type": "Cyclic", "degree": 5, "order": 5, "samples": 10000},
    "Z6": {"type": "Cyclic", "degree": 6, "order": 6, "samples": 12000},
    # Dihedral Groups (symmetries of n-gon)
    "D3": {"type": "Dihedral", "degree": 3, "order": 6, "samples": 10000},  # Same as S3
    "D4": {"type": "Dihedral", "degree": 4, "order": 8, "samples": 16000},
    "D5": {"type": "Dihedral", "degree": 5, "order": 10, "samples": 20000},
    "D6": {"type": "Dihedral", "degree": 6, "order": 12, "samples": 24000},
    "D7": {"type": "Dihedral", "degree": 7, "order": 14, "samples": 28000},
    "D8": {"type": "Dihedral", "degree": 8, "order": 16, "samples": 32000},
    # Mathieu Sporadic Groups (commented out due to computational complexity)
    # "M11": {"type": "Mathieu", "degree": 11, "order": 7920, "samples": 50000},
    # "M12": {"type": "Mathieu", "degree": 12, "order": 95040, "samples": 80000},
    # "M22": {"type": "Mathieu", "degree": 22, "order": 443520, "samples": 100000},
    # "M23": {"type": "Mathieu", "degree": 23, "order": 10200960, "samples": 100000},
    # "M24": {"type": "Mathieu", "degree": 24, "order": 244823040, "samples": 100000},
    # Projective Special Linear Group
    "PSL25": {"type": "PSL", "degree": 6, "order": 60, "samples": 30000},  # PSL(2,5)
    # Frobenius Group
    "F20": {"type": "Frobenius", "degree": 5, "order": 20, "samples": 20000},  # F(5,4)
}

# Sample scaling factors for different lengths
# (shorter sequences can have more samples since they're faster)
SAMPLE_SCALE = {
    4: 2.0,  # 100% more samples
    8: 1.8,  # 80% more samples
    16: 1.6,  # 60% more samples
    32: 1.4,  # 40% more samples
    64: 1.2,  # 20% more samples
    128: 1.0,  # baseline
    256: 0.9,  # 10% fewer samples
    512: 0.8,  # 20% fewer samples
}


def get_samples_for_length(base_samples, length):
    """Calculate number of samples for a given length."""
    return int(base_samples * SAMPLE_SCALE[length])


def generate_all_configs():
    """Generate all configuration combinations."""
    configs = []

    # Original configs (backwards compatibility)
    for group_name, info in GROUPS.items():
        configs.append(
            {
                "name": f"{group_name.lower()}_data",
                "group": group_name,
                "samples": info["samples"],
                "max_len": 512,  # default
            }
        )

    # Length-specific configs
    for length in LENGTHS:
        for group_name, info in GROUPS.items():
            configs.append(
                {
                    "name": f"{group_name.lower()}_len{length}",
                    "group": group_name,
                    "samples": get_samples_for_length(info["samples"], length),
                    "max_len": length,
                }
            )

    return configs


if __name__ == "__main__":
    # Print summary
    print(f"Total groups: {len(GROUPS)}")
    print(f"Length variants: {LENGTHS}")
    print(f"Total configurations: {len(GROUPS) * (len(LENGTHS) + 1)}")

    print("\nGroups by type:")
    by_type = {}
    for name, info in GROUPS.items():
        group_type = info["type"]
        if group_type not in by_type:
            by_type[group_type] = []
        by_type[group_type].append(name)

    for group_type, names in sorted(by_type.items()):
        print(f"  {group_type}: {', '.join(names)}")
