# Group Theory Dataset Generator

A comprehensive Python framework for generating permutation composition datasets across 94 mathematical groups, organized by computational complexity classes (TC⁰/NC¹).


**Dataset Repository**: [BeeGass/Group-Theory-Collection](https://huggingface.co/datasets/BeeGass/Group-Theory-Collection)

## Overview

This project generates datasets for studying the computational boundaries between solvable (TC⁰) and non-solvable (NC¹) groups, providing:

### Dataset Composition: 94 Groups Across 10 Families

**TC⁰ (Solvable) - 58 Groups:**
- **Symmetric Groups**: $S_3$, $S_4$
- **Alternating Groups**: $A_3$, $A_4$
- **Cyclic Groups**: $C_2$ through $C_{30}$ (29 groups)
- **Dihedral Groups**: $D_3$ through $D_{20}$ (18 groups)
- **Klein Four-Group**: $V_4$ (isomorphic to $\mathbb{Z}_2^2$)
- **Quaternion Groups**: $Q_8$, $Q_{16}$, $Q_{32}$
- **Elementary Abelian**: $\mathbb{Z}_2^k$ for $k \in \{1,2,3,4,5\}$, $\mathbb{Z}_3^k$ for $k \in \{1,2,3,4\}$, $\mathbb{Z}_5^k$ for $k \in \{1,2,3,4\}$ (13 groups)
- **Frobenius Groups**: $F_{20}$, $F_{21}$
- **Projective Special Linear**: $\text{PSL}(2,2)$, $\text{PSL}(2,3)$

**NC¹ (Non-Solvable) - 36 Groups:**
- **Symmetric Groups**: $S_5$ through $S_9$
- **Alternating Groups**: $A_5$ through $A_9$
- **Projective Special Linear**: $\text{PSL}(2,4)$, $\text{PSL}(2,5)$, $\text{PSL}(2,7)$, $\text{PSL}(2,8)$, $\text{PSL}(2,9)$, $\text{PSL}(2,11)$, $\text{PSL}(3,2)$, $\text{PSL}(3,3)$, $\text{PSL}(3,4)$, $\text{PSL}(3,5)$
- **Mathieu Groups**: $M_{11}$, $M_{12}$

All datasets contain variable sequence lengths (uniformly distributed between 3 and 1024 permutations), suitable for studying state-space models and transformer architectures across different context lengths.

## Features

- Fast dataset generation using parallel processing
- Direct upload to HuggingFace Hub
- Configurable sequence lengths and dataset sizes
- Comprehensive test suite
- Multiple dataset loading options

## Installation

```bash
# Clone the repository
git clone https://github.com/BeeGass/Group-Dataset-Generator.git
cd Group-Dataset-Generator

# Install dependencies using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

## Quick Start

### Generate a Dataset

```bash
# Generate S5 dataset
uv run python gdg/generate_all.py s5

# Generate all TC⁰ groups
uv run python gdg/generate_all.py tc0

# Generate all NC¹ groups  
uv run python gdg/generate_all.py nc1

# Generate specific groups
uv run python gdg/generate_all.py c5 d6 q8 m11

# Generate all 94 groups
uv run python gdg/generate_all.py all
```

### Load Datasets from HuggingFace

```python
from datasets import load_dataset

# Load specific group datasets
s5_dataset = load_dataset("BeeGass/Group-Theory-Collection", name="s5")
a4_dataset = load_dataset("BeeGass/Group-Theory-Collection", name="a4")
m11_dataset = load_dataset("BeeGass/Group-Theory-Collection", name="m11")

# Access train/test splits
train_data = s5_dataset["train"]
test_data = s5_dataset["test"]

# Example data point (sequences have variable lengths)
print(train_data[0])
# {
#     'input_sequence': '23 45 12 ... (variable number of permutation IDs)',
#     'target': '67',
#     'sequence_length': 512,  # Varies from 3 to 1024
#     'group_degree': 5,
#     'group_order': 120,
#     'group_type': 'symmetric'
# }

# Load datasets from complexity class directories
tc0_cyclic = load_dataset("BeeGass/Group-Theory-Collection", data_dir="TC0/c10")
nc1_symmetric = load_dataset("BeeGass/Group-Theory-Collection", data_dir="NC1/s7")
```

## Dataset Structure

Each dataset contains:
- `input_sequence`: Space-separated permutation IDs (variable length)
- `target`: The ID of the resulting permutation after composition
- `sequence_length`: Number of permutations in the sequence (3 to 1024)
- `group_degree`: Number of elements the group acts on
- `group_order`: Total number of elements in the group
- `group_type`: Type of mathematical group

The composition follows the standard mathematical convention: for input $[p_1, p_2, p_3]$, the result is $p_3 \circ p_2 \circ p_1$.

## Dataset Organization

The datasets are organized in three ways on HuggingFace:

1. **`data/`** - All 94 individual group datasets in flat structure
2. **`TC0/`** - 58 solvable groups (TC⁰ complexity class)
3. **`NC1/`** - 36 non-solvable groups (NC¹ complexity class)

### Complete Group Inventory

**TC⁰ Groups (58 total):**
- Cyclic: $C_2$ through $C_{30}$ (29 groups)
- Dihedral: $D_3$ through $D_{20}$ (18 groups)
- Symmetric (solvable): $S_3$, $S_4$
- Alternating (solvable): $A_3$, $A_4$
- Klein: $V_4$ (isomorphic to $\mathbb{Z}_2^2$)
- Quaternion: $Q_8$, $Q_{16}$, $Q_{32}$
- Elementary Abelian: $\mathbb{Z}_p^k$ for various primes and powers (13 groups)
- Frobenius: $F_{20}$, $F_{21}$
- PSL (solvable): $\text{PSL}(2,2)$, $\text{PSL}(2,3)$

**NC¹ Groups (36 total):**
- Symmetric (non-solvable): $S_5$, $S_6$, $S_7$, $S_8$, $S_9$
- Alternating (non-solvable): $A_5$, $A_6$, $A_7$, $A_8$, $A_9$
- PSL (non-solvable): $\text{PSL}(2,q)$ for $q \in \{4,5,7,8,9,11\}$, $\text{PSL}(3,q)$ for $q \in \{2,3,4,5\}$
- Mathieu: $M_{11}$, $M_{12}$

## Examples

Run the streaming examples:

```bash
# Test streaming from HuggingFace
uv run python examples/streaming_examples.py
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=.

# Run specific test categories
uv run pytest tests/test_generate.py -v
```

## Project Structure

```
Group-Dataset-Generator/
├── gdg/                           # Core package (Group Dataset Generator)
│   ├── base_generator.py          # Base class for all generators
│   ├── generators/                # All specific group generators
│   │   ├── __init__.py           # Package exports
│   │   ├── symmetric.py          # Symmetric groups (Sn)
│   │   ├── alternating.py        # Alternating groups (An)
│   │   ├── cyclic.py             # Cyclic groups (Cn)
│   │   ├── dihedral.py           # Dihedral groups (Dn)
│   │   ├── quaternion.py         # Quaternion groups (Qn)
│   │   ├── elementary_abelian.py # Elementary abelian (Zp^k)
│   │   ├── frobenius.py          # Frobenius groups (Fn)
│   │   ├── klein.py              # Klein four-group (V4)
│   │   ├── psl.py                # Projective special linear groups
│   │   └── mathieu.py            # Mathieu groups (M11, M12)
│   └── generate_all.py           # Master generation script
├── scripts/                      # Utility scripts
│   ├── upload_datasets.sh        # Upload to HuggingFace
│   └── delete_hf_datasets.sh     # Delete from HuggingFace
├── examples/                     # Usage examples
│   └── streaming_examples.py     # Streaming usage examples
├── tests/                        # Comprehensive test suite
│   └── generators/               # Group-specific tests
├── docs/                         # Documentation
│   └── HF_README.md             # HuggingFace dataset card
├── datasets/                     # Generated datasets (git-ignored)
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Methodology

1. **Generation**: The framework generates all permutations for a given group and assigns unique identifiers
2. **Composition**: Random sequences of permutations are created with their composed result
3. **Storage**: Data is saved in Arrow format for efficient loading
4. **Distribution**: Datasets are uploaded to HuggingFace for accessibility

## Contributing

Contributions are welcome. Please submit a Pull Request for any proposed changes.

## License

This project is licensed under the MIT License.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{gass2024groupthoery,
  author = {Bryan Gass},
  title = {Group Theory Collection},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/BeeGass/Group-Theory-Collection}
}

@software{gass2024generator,
  author = {Bryan Gass},
  title = {Group Theory Dataset Generator},
  year = {2024},
  url = {https://github.com/BeeGass/Group-Dataset-Generator}
}
```

## Acknowledgments

This project was inspired by the work of [William Merrill](https://github.com/viking-sudo-rm) and his paper ["The Illusion of State in State-Space Models"](https://arxiv.org/abs/2404.08819), which explores the computational properties of state-space models through group theory.

Built with:
- [SymPy](https://www.sympy.org/) for group theory computations
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/) for data handling
- [PyArrow](https://arrow.apache.org/docs/python/) for efficient storage