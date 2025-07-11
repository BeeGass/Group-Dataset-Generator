# Group Theory Collection - Examples

This directory contains example scripts demonstrating how to use the Group Theory Collection dataset from HuggingFace.

## Available Examples

### streaming_examples.py
Demonstrates how to efficiently load and process group datasets using HuggingFace's streaming mode:
- Loading individual groups
- Filtering by sequence length
- Processing multiple groups simultaneously
- Batch processing for training
- Handling large groups efficiently

**Run with:**
```bash
python examples/streaming_examples.py
```

**Key Features:**
- Memory efficient - no need to download entire datasets
- Perfect for training neural networks
- Real-time filtering and processing
- Works with all 94 available groups

## Dataset Information

The Group Theory Collection contains 94 permutation group datasets organized by computational complexity:
- **TC⁰ Groups** (75 solvable groups): Cyclic, Dihedral, Elementary Abelian, Quaternion, Frobenius, and others
- **NC¹ Groups** (19 non-solvable groups): Large Symmetric/Alternating groups, PSL groups, Mathieu groups

## Quick Start

```python
from datasets import load_dataset

# Load a specific group
dataset = load_dataset("BeeGass/Group-Theory-Collection", data_dir="data/s5")

# Stream data without downloading
dataset = load_dataset("BeeGass/Group-Theory-Collection", data_dir="data/s5", streaming=True)

# Filter by sequence length
filtered = dataset.filter(lambda x: x['sequence_length'] == 16)
```

## Additional Resources

- Full dataset: https://huggingface.co/datasets/BeeGass/Group-Theory-Collection
- Generator code: https://github.com/BeeGass/Group-Dataset-Generator
- Research paper: "The Illusion of State in State-Space Models" (arXiv:2404.08819)