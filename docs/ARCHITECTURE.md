# Architecture Overview

## Project Structure

```
s5-data-gen/
├── gdg/                        # Core package (Group Dataset Generator)
│   ├── base_generator.py       # Base class for all generators
│   ├── generators/             # All specific group generators
│   │   ├── alternating.py      # Alternating groups (A3-A9)
│   │   ├── cyclic.py          # Cyclic groups (C2-C30)
│   │   ├── dihedral.py        # Dihedral groups (D3-D20)
│   │   ├── elementary_abelian.py # Elementary abelian groups (Z_p^k)
│   │   ├── frobenius.py       # Frobenius groups (F20, F21)
│   │   ├── klein.py           # Klein four-group (V4)
│   │   ├── mathieu.py         # Mathieu groups (M11, M12)
│   │   ├── psl.py             # Projective special linear groups
│   │   ├── quaternion.py      # Quaternion groups (Q8, Q16, Q32)
│   │   └── symmetric.py       # Symmetric groups (S3-S9)
│   └── generate_all.py        # Master generation script
│
├── tests/                     # Test suite
│   ├── test_base_individual.py # Base test class
│   ├── test_data_correctness.py # Data validation tests
│   ├── test_generate.py       # Generation tests
│   ├── test_hf_streaming.py   # HuggingFace streaming tests
│   └── generators/            # Group-specific tests
│
├── scripts/                   # Utility scripts
│   ├── upload_datasets.sh     # Upload to HuggingFace
│   └── delete_hf_datasets.sh  # Delete from HuggingFace
│
├── examples/                  # Usage examples
│   ├── local_examples.py      # Local loading examples
│   └── streaming_examples.py  # HuggingFace streaming examples
│
├── docs/                      # Documentation
│   ├── HF_README.md          # HuggingFace dataset card
│   └── ARCHITECTURE.md       # This file
│
└── datasets/                  # Generated datasets (git-ignored)
```

## Key Components

### BaseGroupGenerator

The `BaseGroupGenerator` class in `gdg/base_generator.py` provides the foundation for all group generators:

- **Permutation Management**: Handles permutation-to-ID mapping
- **Dataset Generation**: Creates train/test splits with configurable parameters
- **Sequence Generation**: Generates variable-length sequences (3-1024)
- **Metadata Tracking**: Records group properties and generation parameters

### Group Generators

Each generator in `gdg/generators/` implements:
- `get_elements()`: Returns all group elements as permutation arrays
- `get_group_name()`: Returns the group type name
- `compose()`: Implements group multiplication

### Data Format

Each dataset contains:
- `input_sequence`: Space-separated permutation IDs
- `target`: Result of composing all permutations
- `sequence_length`: Number of permutations in sequence
- `group_degree`: Size of permutation (e.g., S7 has degree 7)
- `group_order`: Total number of elements in group
- `group_type`: Name of the group family

## Design Principles

1. **Modularity**: Each group type has its own generator
2. **Consistency**: All generators follow the same interface
3. **Efficiency**: Uses numpy arrays for fast computation
4. **Scalability**: Supports groups up to ~10^5 elements
5. **Reproducibility**: Seeded random generation

## Testing Strategy

- **Unit Tests**: Each generator has comprehensive tests
- **Integration Tests**: End-to-end dataset generation
- **Data Validation**: Ensures mathematical correctness
- **HuggingFace Tests**: Verifies streaming functionality