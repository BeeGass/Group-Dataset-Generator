# Permutation Groups Dataset Generator

A Python tool for generating permutation composition datasets for symmetric and alternating groups, with seamless HuggingFace integration.

🤗 **Dataset available on HuggingFace**: [BeeGass/permutation-groups](https://huggingface.co/datasets/BeeGass/permutation-groups)

## Overview

This project generates datasets of permutation compositions for various mathematical groups:
- **Symmetric Groups**: S3, S4, S5, S6, S7
- **Alternating Groups**: A3, A4, A5, A6, A7
- **Cyclic Groups**: C3-C12, Z3-Z6 (alternative notation)
- **Dihedral Groups**: D3-D8
- **Special Groups**: PSL(2,5), F20 (Frobenius)

Each dataset consists of sequences of permutations and their compositions, useful for training models on group theory operations. All datasets are available with multiple sequence length variants (4, 8, 16, 32, 64, 128, 256, 512).

## Features

- 🚀 Fast dataset generation using parallel processing
- 📦 Direct upload to HuggingFace Hub
- 🔧 Configurable sequence lengths and dataset sizes
- ✅ Comprehensive test suite
- 📊 Multiple dataset loading options

## Installation

```bash
# Clone the repository
git clone https://github.com/BeeGass/permutation-groups.git
cd permutation-groups

# Install dependencies using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

## Quick Start

### Generate a Dataset

```bash
# Generate S5 dataset with 100,000 samples
uv run python generate_enhanced.py \
    --group-name S5 \
    --num-samples 100000 \
    --min-len 3 \
    --max-len 512 \
    --test-split-size 0.2 \
    --output-dir ./s5_data

# Generate a cyclic group dataset
uv run python generate_enhanced.py \
    --group-name C8 \
    --num-samples 20000 \
    --max-len 64 \
    --output-dir ./c8_data

# Generate all datasets with various lengths
uv run python generate_all_groups.py --lengths 4 8 16 32 64 128 256 512
```

### Load Datasets from HuggingFace

```python
from datasets import load_dataset

# Load a specific group dataset
s5_dataset = load_dataset("BeeGass/permutation-groups", name="s5_data", trust_remote_code=True)

# Load a specific length variant
s5_len32 = load_dataset("BeeGass/permutation-groups", name="s5_len32", trust_remote_code=True)

# Load cyclic group dataset
c8_dataset = load_dataset("BeeGass/permutation-groups", name="c8_data", trust_remote_code=True)

# Load all datasets combined
all_datasets = load_dataset("BeeGass/permutation-groups", name="all", trust_remote_code=True)

# Load all datasets with specific length
all_len64 = load_dataset("BeeGass/permutation-groups", name="all_len64", trust_remote_code=True)

# Access the data
train_data = s5_dataset["train"]
test_data = s5_dataset["test"]

# Example data point
print(train_data[0])
# {'input_sequence': '23 45 12', 'target': '67'}
```

## Dataset Structure

Each dataset contains:
- `input_sequence`: Space-separated permutation IDs to be composed
- `target`: The ID of the resulting permutation after composition

The composition follows the standard mathematical convention: for input `[p1, p2, p3]`, the result is `p3 ∘ p2 ∘ p1`.

## Available Configurations

### Base Configurations (max_len=512)

| Configuration | Group Type | Group Order | Description |
|--------------|------------|-------------|-------------|
| `s3_data` - `s7_data` | Symmetric | 6-5040 | Permutations of 3-7 elements |
| `a3_data` - `a7_data` | Alternating | 3-2520 | Even permutations of 3-7 elements |
| `c3_data` - `c12_data` | Cyclic | 3-12 | Cyclic permutations |
| `z3_data` - `z6_data` | Cyclic | 3-6 | Cyclic (alternative notation) |
| `d3_data` - `d8_data` | Dihedral | 6-16 | Symmetries of n-gons |
| `psl25_data` | PSL(2,5) | 60 | Projective special linear group |
| `f20_data` | Frobenius | 20 | Frobenius group F(5,4) |
| `all` | Combined | - | All datasets combined |

### Length Variants

Each group is also available with specific maximum sequence lengths:
- `{group}_len4` - Maximum sequence length 4
- `{group}_len8` - Maximum sequence length 8  
- `{group}_len16` - Maximum sequence length 16
- `{group}_len32` - Maximum sequence length 32
- `{group}_len64` - Maximum sequence length 64
- `{group}_len128` - Maximum sequence length 128
- `{group}_len256` - Maximum sequence length 256
- `{group}_len512` - Maximum sequence length 512

Example: `s5_len32`, `c8_len64`, `d4_len128`, `all_len16`

## Examples

Run the examples script to see various usage patterns:

```bash
# Basic examples
uv run python examples.py

# Verify dataset correctness
uv run python examples.py --verify

# Benchmark loading times
uv run python examples.py --benchmark
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
permutation-groups/
├── generate.py              # Basic dataset generation script
├── generate_enhanced.py     # Enhanced generator with all group types
├── generate_all_groups.py   # Batch generation for all groups
├── all_groups_config.py     # Configuration for all supported groups
├── permutation-groups.py    # HuggingFace dataset loading script
├── upload_dataset_script.py # Script to upload dataset config to HuggingFace
├── upload_datasets.py       # Batch upload with rate limiting
├── examples.py             # Usage examples and verification
├── tests/
│   └── test_generate.py    # Comprehensive test suite
├── pyproject.toml          # Project configuration
└── README.md               # This file
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
@software{permutation_groups_dataset,
  author = {Bryan Gass},
  title = {Permutation Groups Dataset Generator},
  year = {2024},
  url = {https://github.com/BeeGass/permutation-groups}
}
```

## Acknowledgments

This project was inspired by the work of [William Merrill](https://github.com/viking-sudo-rm) and his paper ["The Illusion of State in State-Space Models"](https://arxiv.org/abs/2404.08819), which explores the computational properties of state-space models through group theory.

Built with:
- [SymPy](https://www.sympy.org/) for group theory computations
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/) for data handling
- [PyArrow](https://arrow.apache.org/docs/python/) for efficient storage