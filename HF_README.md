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
- algebra
- sequence-modeling
- state-space-models
- computational-complexity
pretty_name: Group Theory Collection
size_categories:
- 10M<n<100M
---

# Group Theory Collection

A comprehensive collection of permutation composition datasets for various mathematical groups, organized by computational complexity classes. This dataset is designed for studying the "Illusion of State" phenomenon in state-space models and transformer architectures.

## Overview

This dataset provides 94 individual permutation group datasets spanning 10 different group families, systematically organized to facilitate research on the computational boundaries between solvable and non-solvable groups. The organization reflects the fundamental distinction between TC⁰-computable (solvable groups) and NC¹-complete (non-solvable groups) problems.

### Research Motivation

Recent theoretical work demonstrates that TC⁰ models, including Transformers and standard State-Space Models (SSMs), cannot solve NC¹-complete problems such as composing permutations in non-solvable groups. This dataset enables researchers to:

- Empirically verify theoretical computational complexity boundaries
- Study the "Illusion of State" phenomenon in neural architectures
- Benchmark mathematical reasoning capabilities of sequence models
- Investigate generalization patterns across different group structures
- Analyze the relationship between model architecture and algebraic computation

## Dataset Structure

The dataset is organized in three complementary ways to support different research approaches:

### 1. Flat Organization (data/)
All 59 individual group datasets are available for direct access in a flat structure, facilitating straightforward loading and comparison across groups.

### 2. TC⁰ Complexity Class (TC0/)
Contains 43 solvable groups that can theoretically be computed by constant-depth threshold circuits. These groups serve as positive controls where current neural architectures should succeed.

### 3. NC¹ Complexity Class (NC1/)
Contains 14 non-solvable groups requiring logarithmic-depth circuits for computation. These groups represent problems that are provably beyond the computational capacity of TC⁰ models.

## Usage

### Basic Loading

```python
from datasets import load_dataset

# Load specific group datasets
s5_data = load_dataset("BeeGass/Group-Theory-Collection", data_dir="data/s5")
a4_data = load_dataset("BeeGass/Group-Theory-Collection", data_dir="data/a4")

# Load from complexity-organized directories
tc0_cyclic = load_dataset("BeeGass/Group-Theory-Collection", data_dir="TC0/c10")
nc1_symmetric = load_dataset("BeeGass/Group-Theory-Collection", data_dir="NC1/s7")

# Access train/test splits
train_data = s5_data["train"]
test_data = s5_data["test"]
```

### Data Format

Each example contains the following fields:

```python
{
    'input_sequence': "123 456 789",     # Space-separated permutation IDs to compose
    'target': "234",                      # Result of composition as string
    'sequence_length': 3,                 # Number of permutations in this sequence
    'group_degree': 7,                    # Degree of the permutation group (e.g., S7 acts on 7 elements)
    'group_order': 5040,                  # Order (size) of the group (e.g., |S7| = 7!)
    'group_type': "symmetric"             # Type of the group
}
```

Note: Each dataset contains sequences of varying lengths. The 'sequence_length' field indicates how many permutations are in that particular example's input sequence (ranging from 3 to 1024).

### Filtering by Sequence Length

Since each dataset contains sequences of all lengths from 3 to 1024, researchers often need to filter for specific length ranges:

```python
# Load full dataset
dataset = load_dataset("BeeGass/Group-Theory-Collection", data_dir="data/s5")

# Filter for sequences of specific lengths
short_sequences = dataset.filter(lambda x: x['sequence_length'] <= 32)
medium_sequences = dataset.filter(lambda x: 32 < x['sequence_length'] <= 128)
length_16_only = dataset.filter(lambda x: x['sequence_length'] == 16)
```

## Group Inventory

### TC⁰ Groups (Solvable) - 75 Groups

| Group Family | Groups | Orders | Mathematical Properties |
|--------------|--------|--------|------------------------|
| Symmetric | S3, S4 | 6, 24 | Solvable for n ≤ 4 |
| Alternating | A3, A4 | 3, 12 | Solvable for n ≤ 4 |
| Cyclic | C2-C30 (all) | 2-30 | Abelian groups |
| Dihedral | D3-D20 (all) | 6-40 | Symmetries of regular polygons |
| Klein | V4 | 4 | Smallest non-cyclic abelian group |
| Quaternion | Q8, Q16, Q32 | 8, 16, 32 | Non-abelian 2-groups |
| Elementary Abelian | Z2^[1-5], Z3^[1-4], Z5^[1-4] | Various | Direct products of cyclic groups |
| Frobenius | F20, F21 | 20, 21 | Transitive permutation groups |
| Projective Special Linear | PSL(2,2), PSL(2,3), PSL(2,4), PSL(2,8), PSL(2,9), PSL(3,4) | Various | Some solvable PSL groups |

### NC¹ Groups (Non-Solvable) - 19 Groups

| Group Family | Groups | Orders | Mathematical Properties |
|--------------|--------|--------|------------------------|
| Symmetric | S5, S6, S7, S8, S9 | 120-362,880 | Non-solvable for n ≥ 5 |
| Alternating | A5, A6, A7, A8, A9 | 60-181,440 | Simple groups for n ≥ 5 |
| Projective Special Linear | PSL(2,5), PSL(2,7), PSL(2,11), PSL(3,2), PSL(3,3), PSL(3,5) | Various | Simple groups |
| Mathieu | M11, M12 | 7,920, 95,040 | Sporadic simple groups |

## Technical Specifications

### Permutation Representation
- Each permutation is assigned a unique integer identifier within its group
- Mappings between IDs and permutation arrays are consistent across train/test splits
- Permutation composition follows right-to-left convention (standard in mathematics)

### Dataset Statistics
- **Train/Test Split**: 80/20 ratio for all groups
- **Sequence Lengths**: Variable lengths from 3 to 1024 permutations per example
- **File Format**: Apache Arrow for efficient data loading and memory mapping
- **Total Size**: Varies by group order and maximum sequence length

### Composition Convention
For an input sequence [p₁, p₂, p₃], the target is computed as:
- Mathematical notation: p₃ ∘ p₂ ∘ p₁
- Operational interpretation: First apply p₁, then p₂, then p₃

## Dataset Generation

The code used to generate this dataset is available at [https://github.com/BeeGass/Group-Dataset-Generator](https://github.com/BeeGass/Group-Dataset-Generator). The repository includes:

- Complete implementation of all permutation groups
- Dataset generation scripts with configurable parameters
- Verification and testing utilities
- Documentation for extending the dataset with additional groups

## Research Applications

This dataset supports various research directions:

1. **Computational Complexity Theory**: Empirical validation of TC⁰/NC¹ separation in neural networks
2. **State-Space Model Analysis**: Testing fundamental limitations of linear recurrent architectures
3. **Transformer Architecture Studies**: Investigating attention mechanism constraints
4. **Mathematical Reasoning**: Benchmarking symbolic manipulation capabilities
5. **Generalization Studies**: Cross-length and cross-group generalization patterns
6. **Representation Learning**: Understanding how models encode algebraic structures

## Citation

When using this dataset in academic work, please cite:

```bibtex
@dataset{gass2024permutation,
  author = {Gass, Bryan},
  title = {Group Theory Collection},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/BeeGass/Group-Theory-Collection},
  note = {Organized by computational complexity classes (TC⁰/NC¹)}
}

@software{gass2024generator,
  author = {Gass, Bryan},
  title = {Group Dataset Generator},
  year = {2024},
  url = {https://github.com/BeeGass/Group-Dataset-Generator}
}

@article{merrill2024illusion,
  title = {The Illusion of State in State-Space Models},
  author = {Merrill, William and Jackson, Ashish and Goldstein, Yoav and Weiss, Gail and Angluin, Dana},
  journal = {arXiv preprint arXiv:2404.08819},
  year = {2024}
}
```

## Acknowledgments

This dataset was inspired by the theoretical work of William Merrill and colleagues on "The Illusion of State in State-Space Models" (arXiv:2404.08819), which establishes fundamental computational limitations of state-space models through group-theoretic analysis.

## License

This dataset is released under the MIT License.

## Contact

For questions, issues, or contributions, please use the Hugging Face dataset repository's discussion forum or contact Bryan Gass directly.