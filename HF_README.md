---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- mathematics
- group-theory
- permutations
- symbolic-reasoning
pretty_name: Permutation Groups Dataset
size_categories:
- 100K<n<1M
configs:
- config_name: s3_data
  data_files:
    - split: train
      path: data/s3_data/train/*
    - split: test
      path: data/s3_data/test/*
- config_name: s4_data
  data_files:
    - split: train
      path: data/s4_data/train/*
    - split: test
      path: data/s4_data/test/*
- config_name: s5_data
  data_files:
    - split: train
      path: data/s5_data/train/*
    - split: test
      path: data/s5_data/test/*
- config_name: s6_data
  data_files:
    - split: train
      path: data/s6_data/train/*
    - split: test
      path: data/s6_data/test/*
- config_name: s7_data
  data_files:
    - split: train
      path: data/s7_data/train/*
    - split: test
      path: data/s7_data/test/*
- config_name: a3_data
  data_files:
    - split: train
      path: data/a3_data/train/*
    - split: test
      path: data/a3_data/test/*
- config_name: a4_data
  data_files:
    - split: train
      path: data/a4_data/train/*
    - split: test
      path: data/a4_data/test/*
- config_name: a5_data
  data_files:
    - split: train
      path: data/a5_data/train/*
    - split: test
      path: data/a5_data/test/*
- config_name: a6_data
  data_files:
    - split: train
      path: data/a6_data/train/*
    - split: test
      path: data/a6_data/test/*
- config_name: a7_data
  data_files:
    - split: train
      path: data/a7_data/train/*
    - split: test
      path: data/a7_data/test/*
---

# Permutation Groups Dataset

A comprehensive collection of permutation composition datasets for symmetric and alternating groups, designed for training and evaluating models on group theory operations.

## Dataset Description

This dataset contains permutation composition problems for various mathematical groups:
- **Symmetric Groups**: S3, S4, S5, S6, S7
- **Alternating Groups**: A3, A4, A5, A6, A7

Each dataset consists of sequences of permutations that need to be composed to produce a target permutation. This is useful for:
- Training models on symbolic reasoning
- Evaluating mathematical understanding
- Testing compositional generalization
- Studying group theory properties in neural networks

## Usage

```python
from datasets import load_dataset

# Load a specific group dataset
s5_dataset = load_dataset("BeeGass/permutation-groups", name="s5_data", trust_remote_code=True)

# Load alternating group A5
a5_dataset = load_dataset("BeeGass/permutation-groups", name="a5_data", trust_remote_code=True)

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

Each example contains:
- `input_sequence`: A space-separated sequence of permutation IDs to be composed
- `target`: The ID of the resulting permutation after composition

The composition follows standard mathematical convention: for input `[p1, p2, p3]`, the result is `p3 ∘ p2 ∘ p1`.

## Available Configurations

| Configuration | Group Type | Group Order | Elements | Train Samples | Test Samples |
|--------------|------------|-------------|----------|---------------|--------------|
| `s3_data` | Symmetric | S3 | 6 | 8,000 | 2,000 |
| `s4_data` | Symmetric | S4 | 24 | 16,000 | 4,000 |
| `s5_data` | Symmetric | S5 | 120 | 40,000 | 10,000 |
| `s6_data` | Symmetric | S6 | 720 | 80,000 | 20,000 |
| `s7_data` | Symmetric | S7 | 5,040 | 160,000 | 40,000 |
| `a3_data` | Alternating | A3 | 3 | 4,000 | 1,000 |
| `a4_data` | Alternating | A4 | 12 | 12,000 | 3,000 |
| `a5_data` | Alternating | A5 | 60 | 24,000 | 6,000 |
| `a6_data` | Alternating | A6 | 360 | 64,000 | 16,000 |
| `a7_data` | Alternating | A7 | 2,520 | 120,000 | 30,000 |
| `all` | Combined | - | - | 528,000 | 132,000 |

## Dataset Features

- **Variable sequence length**: Input sequences range from 3 to 512 permutations
- **Consistent formatting**: All permutations use space-separated integer IDs
- **Metadata included**: Each dataset includes a `metadata.json` file mapping IDs to permutation array forms
- **Train/test split**: 80/20 split for all configurations

## Understanding the Data

Each permutation is represented by a unique integer ID. The `metadata.json` file in each dataset folder provides the mapping from IDs to permutation array forms.

For example, in S3:
- ID 0 might map to `[0, 1, 2]` (identity)
- ID 1 might map to `[0, 2, 1]` (transpose elements 1 and 2)
- etc.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@software{permutation_groups_dataset,
  author = {Bryan Gass},
  title = {Permutation Groups Dataset},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/BeeGass/permutation-groups}
}
```

## Acknowledgments

This dataset was inspired by the work of [William Merrill](https://github.com/viking-sudo-rm) and his paper ["The Illusion of State in State-Space Models"](https://arxiv.org/abs/2404.08819), which explores the computational properties of state-space models through group theory.

## License

This dataset is released under the MIT License.

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/BeeGass/permutation-groups).