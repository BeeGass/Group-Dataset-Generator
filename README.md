# Group Theory Dataset Generator

A comprehensive Python framework for generating permutation composition datasets across 94 mathematical groups, organized by computational complexity classes (TC^0/NC^1).


🤗 **Dataset available on HuggingFace**: [BeeGass/Group-Theory-Collection](https://huggingface.co/datasets/BeeGass/Group-Theory-Collection)

## Overview

This project generates datasets for studying the computational boundaries between solvable (TC^0) and non-solvable (NC^1) groups, providing:

### 94 Total Groups across 10 Families:

**TC^0 (Solvable) - 58 Groups:**
- **Symmetric Groups**: S3, S4
- **Alternating Groups**: A3, A4
- **Cyclic Groups**: C2-C30 (29 groups)
- **Dihedral Groups**: D3-D20 (18 groups)
- **Klein Four-Group**: V4 (isomorphic to Z₂²)
- **Quaternion Groups**: Q8, Q16, Q32
- **Elementary Abelian**: Z₂^[1-5], Z₃^[1-4], Z₅^[1-4] (13 groups)
- **Frobenius Groups**: F20, F21
- **Projective Special Linear**: PSL(2,2), PSL(2,3)

**NC^1 (Non-Solvable) - 36 Groups:**
- **Symmetric Groups**: S5-S9
- **Alternating Groups**: A5-A9
- **Projective Special Linear**: PSL(2,4), PSL(2,5), PSL(2,7), PSL(2,8), PSL(2,9), PSL(2,11), PSL(3,2), PSL(3,3), PSL(3,4), PSL(3,5)
- **Mathieu Groups**: M11, M12

All datasets contain variable sequence lengths (uniformly distributed between 3 and 1024 permutations), suitable for studying state-space models and transformer architectures across different context lengths.

## Features

- 🚀 Fast dataset generation using parallel processing
- 📦 Direct upload to HuggingFace Hub
- 🔧 Configurable sequence lengths and dataset sizes
- ✅ Comprehensive test suite
- 📊 Multiple dataset loading options

## Installation

```bash
# Clone the repository
git clone https://github.com/BeeGass/s5-data-gen.git
cd s5-data-gen

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

# Generate all TC^0 groups
uv run python gdg/generate_all.py tc0

# Generate all NC^1 groups  
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

The composition follows the standard mathematical convention: for input `[p1, p2, p3]`, the result is `p3 ∘ p2 ∘ p1`.

## Dataset Organization

The datasets are organized in three ways on HuggingFace:

1. **`data/`** - All 94 individual group datasets in flat structure
2. **`TC0/`** - 58 solvable groups (TC^0 complexity class)
3. **`NC1/`** - 36 non-solvable groups (NC^1 complexity class)

### Complete Group Inventory

**TC^0 Groups (58 total):**
- Cyclic: c2-c30 (29 groups)
- Dihedral: d3-d20 (18 groups)
- Symmetric (solvable): s3, s4
- Alternating (solvable): a3, a4
- Klein: v4 (same as z2_2)
- Quaternion: q8, q16, q32
- Elementary Abelian: z2_[1-5], z3_[1-4], z5_[1-4] (13 groups)
- Frobenius: f20, f21
- PSL (solvable): psl2_2, psl2_3

**NC^1 Groups (36 total):**
- Symmetric (non-solvable): s5, s6, s7, s8, s9
- Alternating (non-solvable): a5, a6, a7, a8, a9
- PSL (non-solvable): psl2_4, psl2_5, psl2_7, psl2_8, psl2_9, psl2_11, psl3_2, psl3_3, psl3_4, psl3_5
- Mathieu: m11, m12

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
s5-data-gen/
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

## How It Works

1. **Generation**: The tool generates all permutations for a given group and assigns unique IDs
2. **Composition**: Random sequences of permutations are created with their composed result
3. **Storage**: Data is saved in Arrow format for efficient loading
4. **Distribution**: Datasets can be uploaded to HuggingFace for easy sharing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

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
  url = {https://github.com/BeeGass/s5-data-gen}
}
```

## Acknowledgments

This project was inspired by the work of [William Merrill](https://github.com/viking-sudo-rm) and his paper ["The Illusion of State in State-Space Models"](https://arxiv.org/abs/2404.08819), which explores the computational properties of state-space models through group theory.

Built with:
- [SymPy](https://www.sympy.org/) for group theory computations
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/) for data handling
- [PyArrow](https://arrow.apache.org/docs/python/) for efficient storage