"""Tests for the permutation dataset generation script."""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate import (
    get_group, 
    generate_and_map_permutations, 
    generate_composition_sample,
    generate_readme_content
)
from datasets import Dataset, DatasetDict
from sympy.combinatorics import Permutation
from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup


class TestGroupGeneration:
    """Test group generation functions."""
    
    def test_get_symmetric_group(self):
        """Test symmetric group creation."""
        s3 = get_group("S3")
        assert type(s3).__name__ == "PermutationGroup"
        assert s3.degree == 3
        assert s3.order() == 6
        
        s5 = get_group("S5")
        assert type(s5).__name__ == "PermutationGroup"
        assert s5.degree == 5
        assert s5.order() == 120
    
    def test_get_alternating_group(self):
        """Test alternating group creation."""
        a4 = get_group("A4")
        assert type(a4).__name__ == "PermutationGroup"
        assert a4.degree == 4
        assert a4.order() == 12
        
        a5 = get_group("A5")
        assert type(a5).__name__ == "PermutationGroup"
        assert a5.degree == 5
        assert a5.order() == 60
    
    def test_invalid_group_name(self):
        """Test error handling for invalid group names."""
        with pytest.raises(ValueError, match="Unknown group name"):
            get_group("X5")
        
        with pytest.raises(ValueError, match="Unknown group name"):
            get_group("Invalid")


class TestPermutationMapping:
    """Test permutation mapping functions."""
    
    def test_generate_and_map_permutations_s3(self):
        """Test permutation mapping for S3."""
        group = SymmetricGroup(3)
        perm_to_id, id_to_perm = generate_and_map_permutations(group)
        
        # Check correct number of permutations
        assert len(perm_to_id) == 6
        assert len(id_to_perm) == 6
        
        # Check that identity is mapped
        identity = Permutation(2)  # Identity for degree 3
        assert str(identity.array_form) in perm_to_id
        
        # Check bijection
        for perm_str, id_val in perm_to_id.items():
            assert id_to_perm[id_val].array_form == eval(perm_str)
    
    def test_generate_and_map_permutations_a4(self):
        """Test permutation mapping for A4."""
        group = AlternatingGroup(4)
        perm_to_id, id_to_perm = generate_and_map_permutations(group)
        
        # Check correct number of permutations
        assert len(perm_to_id) == 12
        assert len(id_to_perm) == 12
        
        # Check all permutations are even
        for perm in id_to_perm.values():
            assert perm.is_even


class TestCompositionSample:
    """Test composition sample generation."""
    
    def test_generate_composition_sample(self):
        """Test single composition sample generation."""
        group = SymmetricGroup(3)
        perm_to_id, id_to_perm = generate_and_map_permutations(group)
        
        sample = generate_composition_sample(
            id_to_perm, perm_to_id, 
            min_len=2, max_len=5, 
            group_degree=3
        )
        
        # Check sample structure
        assert "input_sequence" in sample
        assert "target" in sample
        
        # Check input sequence
        input_ids = [int(x) for x in sample["input_sequence"].split()]
        assert 2 <= len(input_ids) <= 5
        assert all(0 <= id_val < 6 for id_val in input_ids)
        
        # Check target
        target_id = int(sample["target"])
        assert 0 <= target_id < 6
    
    def test_composition_correctness(self):
        """Test that composition is computed correctly."""
        group = SymmetricGroup(3)
        perm_to_id, id_to_perm = generate_and_map_permutations(group)
        
        # Generate multiple samples and verify
        for _ in range(10):
            sample = generate_composition_sample(
                id_to_perm, perm_to_id,
                min_len=3, max_len=3,  # Fixed length for easier testing
                group_degree=3
            )
            
            # Parse sample
            input_ids = [int(x) for x in sample["input_sequence"].split()]
            target_id = int(sample["target"])
            
            # Compute composition manually
            composed = Permutation(2)  # Identity
            for id_val in reversed(input_ids):
                composed = id_to_perm[id_val] * composed
            
            # Check result
            expected_id = perm_to_id[str(composed.array_form)]
            assert target_id == expected_id


class TestDatasetGeneration:
    """Test full dataset generation."""
    
    def test_dataset_creation(self):
        """Test creating a dataset with proper structure."""
        group = SymmetricGroup(3)
        perm_to_id, id_to_perm = generate_and_map_permutations(group)
        
        # Generate samples
        samples = []
        for _ in range(100):
            sample = generate_composition_sample(
                id_to_perm, perm_to_id,
                min_len=2, max_len=5,
                group_degree=3
            )
            samples.append(sample)
        
        # Create dataset
        dataset = Dataset.from_list(samples)
        
        # Check dataset structure
        assert len(dataset) == 100
        assert set(dataset.column_names) == {"input_sequence", "target"}
        
        # Check split
        dataset_dict = dataset.train_test_split(test_size=0.2)
        assert len(dataset_dict["train"]) == 80
        assert len(dataset_dict["test"]) == 20
    
    def test_dataset_saving(self):
        """Test saving dataset to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dataset"
            
            # Create small dataset
            group = SymmetricGroup(3)
            perm_to_id, id_to_perm = generate_and_map_permutations(group)
            
            samples = []
            for _ in range(50):
                sample = generate_composition_sample(
                    id_to_perm, perm_to_id,
                    min_len=2, max_len=5,
                    group_degree=3
                )
                samples.append(sample)
            
            dataset = Dataset.from_list(samples)
            dataset_dict = dataset.train_test_split(test_size=0.2)
            
            # Save dataset
            dataset_dict.save_to_disk(output_path)
            
            # Check files exist
            assert output_path.exists()
            assert (output_path / "dataset_dict.json").exists()
            assert (output_path / "train").exists()
            assert (output_path / "test").exists()
            
            # Save metadata
            metadata_path = output_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump({k: str(v.array_form) for k, v in id_to_perm.items()}, f)
            
            assert metadata_path.exists()
            
            # Load and verify
            loaded_dataset = DatasetDict.load_from_disk(output_path)
            assert len(loaded_dataset["train"]) == 40
            assert len(loaded_dataset["test"]) == 10


class TestReadmeGeneration:
    """Test README content generation."""
    
    def test_readme_content(self):
        """Test README generation includes all required information."""
        from types import SimpleNamespace
        
        args = SimpleNamespace(
            group_name="S5",
            num_samples=100000,
            min_len=3,
            max_len=512,
            test_split_size=0.2,
            hf_repo="test/repo"
        )
        
        readme = generate_readme_content(
            args, 
            group_order=120,
            group_degree=5,
            num_train_samples=80000,
            num_test_samples=20000
        )
        
        # Check key information is present
        assert "S5" in readme
        assert "Symmetric Group" in readme
        assert "120" in readme  # group order
        assert "100000" in readme  # total samples
        assert "80000" in readme  # train samples
        assert "20000" in readme  # test samples
        assert "test/repo" in readme
        assert "metadata.json" in readme


class TestIntegration:
    """Integration tests for the full generation process."""
    
    @pytest.mark.slow
    def test_full_generation_small_group(self):
        """Test full generation process for a small group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from types import SimpleNamespace
            import subprocess
            
            # Run generation script
            result = subprocess.run([
                sys.executable, "generate.py",
                "--group-name", "S3",
                "--num-samples", "1000",
                "--min-len", "2",
                "--max-len", "10",
                "--test-split-size", "0.2",
                "--output-dir", tmpdir
            ], capture_output=True, text=True)
            
            assert result.returncode == 0
            
            # Check output files
            output_path = Path(tmpdir)
            assert (output_path / "dataset_dict.json").exists()
            assert (output_path / "metadata.json").exists()
            assert (output_path / "train").exists()
            assert (output_path / "test").exists()
            
            # Load and verify dataset
            dataset = DatasetDict.load_from_disk(output_path)
            assert len(dataset["train"]) == 800
            assert len(dataset["test"]) == 200
            
            # Load metadata
            with open(output_path / "metadata.json", "r") as f:
                metadata = json.load(f)
            assert len(metadata) == 6  # S3 has 6 elements


if __name__ == "__main__":
    pytest.main([__file__, "-v"])