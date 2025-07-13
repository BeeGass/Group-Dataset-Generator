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
configs:
- config_name: default
  data_files:
    - split: train
      path: data/*/train/data-*
    - split: test
      path: data/*/test/data-*
- config_name: s3
  data_files:
    - split: train
      path: data/s3/train/data-*
    - split: test
      path: data/s3/test/data-*
- config_name: s4
  data_files:
    - split: train
      path: data/s4/train/data-*
    - split: test
      path: data/s4/test/data-*
- config_name: s5
  data_files:
    - split: train
      path: data/s5/train/data-*
    - split: test
      path: data/s5/test/data-*
- config_name: s6
  data_files:
    - split: train
      path: data/s6/train/data-*
    - split: test
      path: data/s6/test/data-*
- config_name: s7
  data_files:
    - split: train
      path: data/s7/train/data-*
    - split: test
      path: data/s7/test/data-*
- config_name: s8
  data_files:
    - split: train
      path: data/s8/train/data-*
    - split: test
      path: data/s8/test/data-*
- config_name: s9
  data_files:
    - split: train
      path: data/s9/train/data-*
    - split: test
      path: data/s9/test/data-*
- config_name: a3
  data_files:
    - split: train
      path: data/a3/train/data-*
    - split: test
      path: data/a3/test/data-*
- config_name: a4
  data_files:
    - split: train
      path: data/a4/train/data-*
    - split: test
      path: data/a4/test/data-*
- config_name: a5
  data_files:
    - split: train
      path: data/a5/train/data-*
    - split: test
      path: data/a5/test/data-*
- config_name: a6
  data_files:
    - split: train
      path: data/a6/train/data-*
    - split: test
      path: data/a6/test/data-*
- config_name: a7
  data_files:
    - split: train
      path: data/a7/train/data-*
    - split: test
      path: data/a7/test/data-*
- config_name: a8
  data_files:
    - split: train
      path: data/a8/train/data-*
    - split: test
      path: data/a8/test/data-*
- config_name: a9
  data_files:
    - split: train
      path: data/a9/train/data-*
    - split: test
      path: data/a9/test/data-*
- config_name: c2
  data_files:
    - split: train
      path: data/c2/train/data-*
    - split: test
      path: data/c2/test/data-*
- config_name: c3
  data_files:
    - split: train
      path: data/c3/train/data-*
    - split: test
      path: data/c3/test/data-*
- config_name: c4
  data_files:
    - split: train
      path: data/c4/train/data-*
    - split: test
      path: data/c4/test/data-*
- config_name: c5
  data_files:
    - split: train
      path: data/c5/train/data-*
    - split: test
      path: data/c5/test/data-*
- config_name: c6
  data_files:
    - split: train
      path: data/c6/train/data-*
    - split: test
      path: data/c6/test/data-*
- config_name: c7
  data_files:
    - split: train
      path: data/c7/train/data-*
    - split: test
      path: data/c7/test/data-*
- config_name: c8
  data_files:
    - split: train
      path: data/c8/train/data-*
    - split: test
      path: data/c8/test/data-*
- config_name: c9
  data_files:
    - split: train
      path: data/c9/train/data-*
    - split: test
      path: data/c9/test/data-*
- config_name: c10
  data_files:
    - split: train
      path: data/c10/train/data-*
    - split: test
      path: data/c10/test/data-*
- config_name: c11
  data_files:
    - split: train
      path: data/c11/train/data-*
    - split: test
      path: data/c11/test/data-*
- config_name: c12
  data_files:
    - split: train
      path: data/c12/train/data-*
    - split: test
      path: data/c12/test/data-*
- config_name: c13
  data_files:
    - split: train
      path: data/c13/train/data-*
    - split: test
      path: data/c13/test/data-*
- config_name: c14
  data_files:
    - split: train
      path: data/c14/train/data-*
    - split: test
      path: data/c14/test/data-*
- config_name: c15
  data_files:
    - split: train
      path: data/c15/train/data-*
    - split: test
      path: data/c15/test/data-*
- config_name: c16
  data_files:
    - split: train
      path: data/c16/train/data-*
    - split: test
      path: data/c16/test/data-*
- config_name: c17
  data_files:
    - split: train
      path: data/c17/train/data-*
    - split: test
      path: data/c17/test/data-*
- config_name: c18
  data_files:
    - split: train
      path: data/c18/train/data-*
    - split: test
      path: data/c18/test/data-*
- config_name: c19
  data_files:
    - split: train
      path: data/c19/train/data-*
    - split: test
      path: data/c19/test/data-*
- config_name: c20
  data_files:
    - split: train
      path: data/c20/train/data-*
    - split: test
      path: data/c20/test/data-*
- config_name: c21
  data_files:
    - split: train
      path: data/c21/train/data-*
    - split: test
      path: data/c21/test/data-*
- config_name: c22
  data_files:
    - split: train
      path: data/c22/train/data-*
    - split: test
      path: data/c22/test/data-*
- config_name: c23
  data_files:
    - split: train
      path: data/c23/train/data-*
    - split: test
      path: data/c23/test/data-*
- config_name: c24
  data_files:
    - split: train
      path: data/c24/train/data-*
    - split: test
      path: data/c24/test/data-*
- config_name: c25
  data_files:
    - split: train
      path: data/c25/train/data-*
    - split: test
      path: data/c25/test/data-*
- config_name: c26
  data_files:
    - split: train
      path: data/c26/train/data-*
    - split: test
      path: data/c26/test/data-*
- config_name: c27
  data_files:
    - split: train
      path: data/c27/train/data-*
    - split: test
      path: data/c27/test/data-*
- config_name: c28
  data_files:
    - split: train
      path: data/c28/train/data-*
    - split: test
      path: data/c28/test/data-*
- config_name: c29
  data_files:
    - split: train
      path: data/c29/train/data-*
    - split: test
      path: data/c29/test/data-*
- config_name: c30
  data_files:
    - split: train
      path: data/c30/train/data-*
    - split: test
      path: data/c30/test/data-*
- config_name: d3
  data_files:
    - split: train
      path: data/d3/train/data-*
    - split: test
      path: data/d3/test/data-*
- config_name: d4
  data_files:
    - split: train
      path: data/d4/train/data-*
    - split: test
      path: data/d4/test/data-*
- config_name: d5
  data_files:
    - split: train
      path: data/d5/train/data-*
    - split: test
      path: data/d5/test/data-*
- config_name: d6
  data_files:
    - split: train
      path: data/d6/train/data-*
    - split: test
      path: data/d6/test/data-*
- config_name: d7
  data_files:
    - split: train
      path: data/d7/train/data-*
    - split: test
      path: data/d7/test/data-*
- config_name: d8
  data_files:
    - split: train
      path: data/d8/train/data-*
    - split: test
      path: data/d8/test/data-*
- config_name: d9
  data_files:
    - split: train
      path: data/d9/train/data-*
    - split: test
      path: data/d9/test/data-*
- config_name: d10
  data_files:
    - split: train
      path: data/d10/train/data-*
    - split: test
      path: data/d10/test/data-*
- config_name: d11
  data_files:
    - split: train
      path: data/d11/train/data-*
    - split: test
      path: data/d11/test/data-*
- config_name: d12
  data_files:
    - split: train
      path: data/d12/train/data-*
    - split: test
      path: data/d12/test/data-*
- config_name: d13
  data_files:
    - split: train
      path: data/d13/train/data-*
    - split: test
      path: data/d13/test/data-*
- config_name: d14
  data_files:
    - split: train
      path: data/d14/train/data-*
    - split: test
      path: data/d14/test/data-*
- config_name: d15
  data_files:
    - split: train
      path: data/d15/train/data-*
    - split: test
      path: data/d15/test/data-*
- config_name: d16
  data_files:
    - split: train
      path: data/d16/train/data-*
    - split: test
      path: data/d16/test/data-*
- config_name: d17
  data_files:
    - split: train
      path: data/d17/train/data-*
    - split: test
      path: data/d17/test/data-*
- config_name: d18
  data_files:
    - split: train
      path: data/d18/train/data-*
    - split: test
      path: data/d18/test/data-*
- config_name: d19
  data_files:
    - split: train
      path: data/d19/train/data-*
    - split: test
      path: data/d19/test/data-*
- config_name: d20
  data_files:
    - split: train
      path: data/d20/train/data-*
    - split: test
      path: data/d20/test/data-*
- config_name: q8
  data_files:
    - split: train
      path: data/q8/train/data-*
    - split: test
      path: data/q8/test/data-*
- config_name: q16
  data_files:
    - split: train
      path: data/q16/train/data-*
    - split: test
      path: data/q16/test/data-*
- config_name: q32
  data_files:
    - split: train
      path: data/q32/train/data-*
    - split: test
      path: data/q32/test/data-*
- config_name: f20
  data_files:
    - split: train
      path: data/f20/train/data-*
    - split: test
      path: data/f20/test/data-*
- config_name: f21
  data_files:
    - split: train
      path: data/f21/train/data-*
    - split: test
      path: data/f21/test/data-*
- config_name: v4
  data_files:
    - split: train
      path: data/v4/train/data-*
    - split: test
      path: data/v4/test/data-*
- config_name: z2_1
  data_files:
    - split: train
      path: data/z2_1/train/data-*
    - split: test
      path: data/z2_1/test/data-*
- config_name: z2_2
  data_files:
    - split: train
      path: data/z2_2/train/data-*
    - split: test
      path: data/z2_2/test/data-*
- config_name: z2_3
  data_files:
    - split: train
      path: data/z2_3/train/data-*
    - split: test
      path: data/z2_3/test/data-*
- config_name: z2_4
  data_files:
    - split: train
      path: data/z2_4/train/data-*
    - split: test
      path: data/z2_4/test/data-*
- config_name: z2_5
  data_files:
    - split: train
      path: data/z2_5/train/data-*
    - split: test
      path: data/z2_5/test/data-*
- config_name: z3_1
  data_files:
    - split: train
      path: data/z3_1/train/data-*
    - split: test
      path: data/z3_1/test/data-*
- config_name: z3_2
  data_files:
    - split: train
      path: data/z3_2/train/data-*
    - split: test
      path: data/z3_2/test/data-*
- config_name: z3_3
  data_files:
    - split: train
      path: data/z3_3/train/data-*
    - split: test
      path: data/z3_3/test/data-*
- config_name: z3_4
  data_files:
    - split: train
      path: data/z3_4/train/data-*
    - split: test
      path: data/z3_4/test/data-*
- config_name: z5_1
  data_files:
    - split: train
      path: data/z5_1/train/data-*
    - split: test
      path: data/z5_1/test/data-*
- config_name: z5_2
  data_files:
    - split: train
      path: data/z5_2/train/data-*
    - split: test
      path: data/z5_2/test/data-*
- config_name: z5_3
  data_files:
    - split: train
      path: data/z5_3/train/data-*
    - split: test
      path: data/z5_3/test/data-*
- config_name: z5_4
  data_files:
    - split: train
      path: data/z5_4/train/data-*
    - split: test
      path: data/z5_4/test/data-*
- config_name: psl2_2
  data_files:
    - split: train
      path: data/psl2_2/train/data-*
    - split: test
      path: data/psl2_2/test/data-*
- config_name: psl2_3
  data_files:
    - split: train
      path: data/psl2_3/train/data-*
    - split: test
      path: data/psl2_3/test/data-*
- config_name: psl2_4
  data_files:
    - split: train
      path: data/psl2_4/train/data-*
    - split: test
      path: data/psl2_4/test/data-*
- config_name: psl2_5
  data_files:
    - split: train
      path: data/psl2_5/train/data-*
    - split: test
      path: data/psl2_5/test/data-*
- config_name: psl2_7
  data_files:
    - split: train
      path: data/psl2_7/train/data-*
    - split: test
      path: data/psl2_7/test/data-*
- config_name: psl2_8
  data_files:
    - split: train
      path: data/psl2_8/train/data-*
    - split: test
      path: data/psl2_8/test/data-*
- config_name: psl2_9
  data_files:
    - split: train
      path: data/psl2_9/train/data-*
    - split: test
      path: data/psl2_9/test/data-*
- config_name: psl2_11
  data_files:
    - split: train
      path: data/psl2_11/train/data-*
    - split: test
      path: data/psl2_11/test/data-*
- config_name: psl3_2
  data_files:
    - split: train
      path: data/psl3_2/train/data-*
    - split: test
      path: data/psl3_2/test/data-*
- config_name: psl3_3
  data_files:
    - split: train
      path: data/psl3_3/train/data-*
    - split: test
      path: data/psl3_3/test/data-*
- config_name: psl3_4
  data_files:
    - split: train
      path: data/psl3_4/train/data-*
    - split: test
      path: data/psl3_4/test/data-*
- config_name: psl3_5
  data_files:
    - split: train
      path: data/psl3_5/train/data-*
    - split: test
      path: data/psl3_5/test/data-*
- config_name: m11
  data_files:
    - split: train
      path: data/m11/train/data-*
    - split: test
      path: data/m11/test/data-*
- config_name: m12
  data_files:
    - split: train
      path: data/m12/train/data-*
    - split: test
      path: data/m12/test/data-*
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
All 94 individual group datasets are available for direct access in a flat structure, facilitating straightforward loading and comparison across groups.

### 2. TC⁰ Complexity Class (TC0/)
Contains 58 solvable groups that can theoretically be computed by constant-depth threshold circuits. These groups serve as positive controls where current neural architectures should succeed.

### 3. NC¹ Complexity Class (NC1/)
Contains 36 non-solvable groups requiring logarithmic-depth circuits for computation. These groups represent problems that are provably beyond the computational capacity of TC⁰ models.

## Usage

### Basic Loading

```python
from datasets import load_dataset

# Load specific group datasets using config names
s5_data = load_dataset("BeeGass/Group-Theory-Collection", name="s5")
a4_data = load_dataset("BeeGass/Group-Theory-Collection", name="a4")
m11_data = load_dataset("BeeGass/Group-Theory-Collection", name="m11")

# Alternative: Load from data directories
s5_data = load_dataset("BeeGass/Group-Theory-Collection", data_dir="data/s5")
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
    'input_sequence': "123 456 789 ...",             # Space-separated permutation IDs (variable length)
    'target': "234",                                  # Result of composition as string
    'sequence_length': 512,                           # Length of input sequence (varies from 3 to 1024)
    'group_degree': 7,                                # Degree of the permutation group (e.g., S7 acts on 7 elements)
    'group_order': 5040,                              # Order (size) of the group (e.g., |S7| = 7!)
    'group_type': "symmetric"                         # Type of the group
}
```

Note: Sequences contain a variable number of permutation IDs (uniformly distributed between 3 and 1024). The provided target is the composition of all permutations in the input sequence.

### Working with Different Sequence Lengths

The dataset already contains sequences of varying lengths (3 to 1024). You can filter or analyze based on sequence length:

```python
# Load full dataset
dataset = load_dataset("BeeGass/Group-Theory-Collection", name="s5")

# Example: Filter for specific sequence lengths
short_sequences = dataset['train'].filter(lambda x: x['sequence_length'] <= 32)
medium_sequences = dataset['train'].filter(lambda x: 32 < x['sequence_length'] <= 256)
long_sequences = dataset['train'].filter(lambda x: x['sequence_length'] > 256)

# Analyze sequence length distribution
import numpy as np
lengths = np.array(dataset['train']['sequence_length'])
print(f"Min length: {lengths.min()}, Max length: {lengths.max()}")
print(f"Mean length: {lengths.mean():.1f}, Std: {lengths.std():.1f}")
```

## Group Inventory

### TC⁰ Groups (Solvable) - 58 Groups

| Group Family | Groups | Orders | Mathematical Properties |
|--------------|--------|--------|------------------------|
| Symmetric | S3, S4 | 6, 24 | Solvable for n ≤ 4 |
| Alternating | A3, A4 | 3, 12 | Solvable for n ≤ 4 |
| Cyclic | C2-C30 (all) | 2-30 | Abelian groups |
| Dihedral | D3-D20 (all) | 6-40 | Symmetries of regular polygons |
| Klein | V4 | 4 | Smallest non-cyclic abelian group (isomorphic to Z₂²) |
| Quaternion | Q8, Q16, Q32 | 8, 16, 32 | Non-abelian 2-groups |
| Elementary Abelian | Z2^[1-5], Z3^[1-4], Z5^[1-4] | Various | Direct products of cyclic groups |
| Frobenius | F20, F21 | 20, 21 | Transitive permutation groups |
| Projective Special Linear | PSL(2,2), PSL(2,3) | 6, 12 | Solvable PSL groups |

### NC¹ Groups (Non-Solvable) - 36 Groups

| Group Family | Groups | Orders | Mathematical Properties |
|--------------|--------|--------|------------------------|
| Symmetric | S5, S6, S7, S8, S9 | 120-362,880 | Non-solvable for n ≥ 5 |
| Alternating | A5, A6, A7, A8, A9 | 60-181,440 | Simple groups for n ≥ 5 |
| Projective Special Linear | PSL(2,4), PSL(2,5), PSL(2,7), PSL(2,8), PSL(2,9), PSL(2,11), PSL(3,2), PSL(3,3), PSL(3,4), PSL(3,5) | Various | Simple groups (PSL(2,4) ≅ A5) |
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