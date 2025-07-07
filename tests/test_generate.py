import pytest
from unittest.mock import patch
from pathlib import Path
import shutil
import json

from sympy.combinatorics import Permutation
from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup
from datasets import DatasetDict, Dataset

# Add the parent directory to the path to import generate
import sys
sys.path.append(str(Path(__file__).parent.parent))

from generate import (
    get_group,
    generate_and_map_permutations,
    generate_composition_sample,
    main as generate_main,
)


def test_get_group():
    """Test that the group factory function works correctly."""
    s3 = get_group('S3')
    assert s3.degree == 3
    assert s3.order() == 6

    a4 = get_group('A4')
    assert a4.degree == 4
    assert a4.order() == 12
    with pytest.raises(ValueError):
        get_group('Z5')  # Invalid group name


def test_generate_and_map_permutations():
    """Test the permutation mapping for a small, known group (S3)."""
    group = SymmetricGroup(3)
    perm_to_id, id_to_perm = generate_and_map_permutations(group)

    assert len(perm_to_id) == 6  # S3 has 3! = 6 elements
    assert len(id_to_perm) == 6

    # Check for a specific, known permutation
    p = Permutation([1, 2, 0])  # A 3-cycle
    assert str(p.array_form) in perm_to_id
    p_id = perm_to_id[str(p.array_form)]
    assert id_to_perm[p_id] == p


def test_generate_composition_sample():
    """Test the composition logic for a single sample."""
    group = SymmetricGroup(3)
    group_degree = group.degree
    perm_to_id, id_to_perm = generate_and_map_permutations(group)

    # Mock random.randint and random.choice to get a predictable sample
    with patch('random.randint', return_value=3), patch('random.choice', side_effect=[0, 1, 2]):  # Sample IDs 0, 1, 2

        sample = generate_composition_sample(id_to_perm, perm_to_id, 3, 5, group_degree)

        # Expected sequence: id_0, id_1, id_2
        p0 = id_to_perm[0]
        p1 = id_to_perm[1]
        p2 = id_to_perm[2]

        # Composition is p2 * p1 * p0
        expected_composition = p2 * p1 * p0
        expected_target_id = perm_to_id[str(expected_composition.array_form)]

        assert sample['input_sequence'] == "0 1 2"
        assert sample['target'] == str(expected_target_id)


@patch('generate.login')
@patch('datasets.DatasetDict.push_to_hub')
def test_end_to_end_script(mock_push_to_hub, mock_login, tmp_path):
    """Run a small, end-to-end test of the main script logic."""
    group_name = 'S3'
    num_samples = 100
    output_dir = tmp_path / 'test_s3_data'

    # Use a context manager to temporarily change sys.argv
    with patch.object(sys, 'argv', [
        'generate.py',
        '--group-name', group_name,
        '--num-samples', str(num_samples),
        '--output-dir', str(output_dir),
        '--test-split-size', '0.2'
    ]):
        # We need to import the script's main execution block
        # to run it under the patched argv.
        generate_main()

    # Verify outputs
    assert output_dir.exists()
    assert (output_dir / "dataset_dict.json").exists()

    # Load the dataset and check its properties
    loaded_dataset = DatasetDict.load_from_disk(output_dir)
    assert 'train' in loaded_dataset
    assert 'test' in loaded_dataset
    assert len(loaded_dataset['train']) == 80
    assert len(loaded_dataset['test']) == 20

    # Check metadata
    assert loaded_dataset['train'].info.description == f"Permutation composition benchmark for the {group_name} group."
    assert loaded_dataset['train'].info.homepage == "https://github.com/your-repo"
    assert loaded_dataset['train'].info.license == "mit"

    # Check for metadata.json
    metadata_path = output_dir / "metadata.json"
    assert metadata_path.exists()
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    assert "0" in metadata # Check for a known key from id_to_perm


def test_dataset_integrity(tmp_path):
    """Test the integrity of the generated dataset by re-calculating compositions."""
    group_name = 'S3'
    num_samples = 10
    output_dir = tmp_path / 'integrity_test_s3_data'

    # Generate a small dataset for integrity check
    with patch.object(sys, 'argv', [
        'generate.py',
        '--group-name', group_name,
        '--num-samples', str(num_samples),
        '--output-dir', str(output_dir),
        '--test-split-size', '0.5'
    ]):
        generate_main()

    loaded_dataset = DatasetDict.load_from_disk(output_dir)
    group = get_group(group_name)
    perm_to_id, id_to_perm = generate_and_map_permutations(group)

    for split in ['train', 'test']:
        for sample in loaded_dataset[split]:
            input_ids = [int(x) for x in sample['input_sequence'].split(' ')]
            target_id = int(sample['target'])

            input_perms = [id_to_perm[i] for i in input_ids]

            # Re-calculate composition
            composed_perm = Permutation(group.degree - 1) # Start with identity
            for p in reversed(input_perms):
                composed_perm = p * composed_perm
            
            recalculated_target_id = perm_to_id[str(composed_perm.array_form)]

            assert recalculated_target_id == target_id, \
                f"Composition mismatch in {split} split: expected {target_id}, got {recalculated_target_id}"

    # Clean up the created directory
    shutil.rmtree(output_dir)