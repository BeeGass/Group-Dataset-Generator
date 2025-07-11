#!/usr/bin/env python3
"""
Base class for generating permutation group datasets.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import json
from typing import List, Tuple, Dict, Optional


class BaseGroupGenerator(ABC):
    """Base class for generating permutation group datasets."""

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.seed = seed
        np.random.seed(seed)
        self._permutations = None
        self._perm_to_idx = None

    @abstractmethod
    def get_group_name(self) -> str:
        """Return the name of the group type."""
        pass

    @abstractmethod
    def generate_group(self, **params) -> List[np.ndarray]:
        """Generate the permutation group with given parameters."""
        pass

    @abstractmethod
    def get_valid_parameters(self) -> List[Dict]:
        """Return list of valid parameter sets for this group type."""
        pass

    def _compute_composition(self, perm1: np.ndarray, perm2: np.ndarray) -> np.ndarray:
        """Compute composition of two permutations."""
        return perm1[perm2]

    def _find_permutation_index(self, perm: np.ndarray) -> int:
        """Find index of permutation in the group."""
        perm_tuple = tuple(perm)
        if perm_tuple in self._perm_to_idx:
            return self._perm_to_idx[perm_tuple]

        # If not found, search linearly (shouldn't happen if group is complete)
        for i, p in enumerate(self._permutations):
            if np.array_equal(p, perm):
                return i
        raise ValueError(f"Permutation {perm} not found in group")

    def generate_dataset(
        self,
        params: Dict,
        num_train_samples: int = 100000,
        num_test_samples: int = 20000,
        min_seq_length: int = 3,
        max_seq_length: int = 1024,
        output_dir: Optional[Path] = None,
    ) -> Tuple[DatasetDict, Dict]:
        """Generate dataset for specific group parameters."""
        # Generate the group
        self._permutations = self.generate_group(**params)
        group_order = len(self._permutations)

        # Create mapping for fast lookup
        self._perm_to_idx = {tuple(p): i for i, p in enumerate(self._permutations)}

        print(f"Generated {self.get_group_name()} with parameters {params}")
        print(f"Group order: {group_order}")

        # Generate samples
        def generate_samples(num_samples):
            samples = {
                "input_sequence": [],
                "target": [],
                "sequence_length": [],
                "group_degree": [],
                "group_order": [],
                "group_type": [],
            }

            for _ in tqdm(range(num_samples), desc="Generating samples"):
                # Random sequence length
                seq_length = np.random.randint(min_seq_length, max_seq_length + 1)

                # Random permutation indices
                input_indices = np.random.randint(0, group_order, size=seq_length)

                # Compute composition
                result = np.arange(len(self._permutations[0]))
                for idx in input_indices:
                    result = self._compute_composition(result, self._permutations[idx])

                # Find target index
                target_idx = self._find_permutation_index(result)

                # Store as strings to match expected format
                samples["input_sequence"].append(" ".join(map(str, input_indices)))
                samples["target"].append(str(target_idx))
                samples["sequence_length"].append(seq_length)
                samples["group_degree"].append(len(self._permutations[0]))
                samples["group_order"].append(group_order)
                samples["group_type"].append(self.get_group_name())

            return samples

        # Generate train and test sets
        print("\nGenerating training samples...")
        train_samples = generate_samples(num_train_samples)

        print("\nGenerating test samples...")
        test_samples = generate_samples(num_test_samples)

        # Create datasets
        train_dataset = Dataset.from_dict(train_samples)
        test_dataset = Dataset.from_dict(test_samples)

        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

        # Create metadata
        metadata = {
            "group_type": self.get_group_name(),
            "group_parameters": params,
            "group_order": group_order,
            "group_degree": len(self._permutations[0]),
            "num_train_samples": num_train_samples,
            "num_test_samples": num_test_samples,
            "min_seq_length": min_seq_length,
            "max_seq_length": max_seq_length,
            "permutation_map": {
                str(i): p.tolist() for i, p in enumerate(self._permutations)
            },
        }

        # Save if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            dataset_dict.save_to_disk(str(output_dir))

            with open(output_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"\nDataset saved to {output_dir}")

        return dataset_dict, metadata
