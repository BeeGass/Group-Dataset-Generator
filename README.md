# Permutation Groups Dataset Generator

A Python tool for generating permutation composition datasets for symmetric and alternating groups, with seamless HuggingFace integration.

## Overview

This project generates datasets of permutation compositions for various mathematical groups:
- **Symmetric Groups**: S3, S4, S5, S6, S7
- **Alternating Groups**: A5, A6, A7

Each dataset consists of sequences of permutations and their compositions, useful for training models on group theory operations.

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
uv run python generate.py \
    --group-name S5 \
    --num-samples 100000 \
    --min-len 3 \
    --max-len 512 \
    --test-split-size 0.2 \
    --output-dir ./s5_data

# Upload to HuggingFace (requires authentication)
uv run python generate.py \
    --group-name S5 \
    --num-samples 100000 \
    --hf-repo YourUsername/permutation-groups \
    --hf-upload
```

### Load Datasets from HuggingFace

```python
from datasets import load_dataset

# Load a specific group dataset
s5_dataset = load_dataset("BeeGass/permutation-groups", name="s5_data", trust_remote_code=True)

# Load all datasets combined
all_datasets = load_dataset("BeeGass/permutation-groups", name="all", trust_remote_code=True)

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

| Configuration | Group Type | Group Order | Description |
|--------------|------------|-------------|-------------|
| `s3_data` | Symmetric | 6 | Permutations of 3 elements |
| `s4_data` | Symmetric | 24 | Permutations of 4 elements |
| `s5_data` | Symmetric | 120 | Permutations of 5 elements |
| `s6_data` | Symmetric | 720 | Permutations of 6 elements |
| `s7_data` | Symmetric | 5040 | Permutations of 7 elements |
| `a5_data` | Alternating | 60 | Even permutations of 5 elements |
| `a6_data` | Alternating | 360 | Even permutations of 6 elements |
| `a7_data` | Alternating | 2520 | Even permutations of 7 elements |
| `all` | Combined | - | All datasets combined |

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
├── generate.py              # Main dataset generation script
├── permutation-groups.py    # HuggingFace dataset loading script
├── upload_dataset_script.py # Script to upload dataset config to HuggingFace
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
  author = {BeeGass},
  title = {Permutation Groups Dataset Generator},
  year = {2024},
  url = {https://github.com/BeeGass/permutation-groups}
}
```

## Acknowledgments

Built with:
- [SymPy](https://www.sympy.org/) for group theory computations
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/) for data handling
- [PyArrow](https://arrow.apache.org/docs/python/) for efficient storage