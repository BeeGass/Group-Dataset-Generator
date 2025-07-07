#!/usr/bin/env python3
"""
Script to upload the dataset loading script to Hugging Face.
This should be run after all datasets have been uploaded to enable
configuration-based loading.
"""

import argparse
from huggingface_hub import HfApi, login
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Upload the dataset loading script to Hugging Face"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="BeeGass/permutation-groups",
        help="Hugging Face repository ID",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face API token for authentication",
    )
    
    args = parser.parse_args()
    
    print(f"Uploading dataset script to {args.hf_repo}")
    
    # Login to Hugging Face
    if args.hf_token:
        login(token=args.hf_token)
    
    # Upload the dataset script
    api = HfApi()
    
    # Upload permutation-groups.py
    script_path = Path("permutation-groups.py")
    if script_path.exists():
        api.upload_file(
            path_or_fileobj=str(script_path),
            path_in_repo="permutation-groups.py",
            repo_id=args.hf_repo,
            repo_type="dataset",
        )
        print("Dataset script uploaded successfully!")
    else:
        print("Error: permutation-groups.py not found!")
        return
    
    # Also create/update a README if needed
    readme_content = """# Permutation Groups Datasets

This repository contains permutation composition datasets for various symmetric and alternating groups.

## Usage

You can load individual datasets or all datasets combined:

```python
from datasets import load_dataset

# Load a specific dataset
s3_dataset = load_dataset("BeeGass/permutation-groups", name="s3_data", trust_remote_code=True)
s7_dataset = load_dataset("BeeGass/permutation-groups", name="s7_data", trust_remote_code=True)
a7_dataset = load_dataset("BeeGass/permutation-groups", name="a7_data", trust_remote_code=True)

# Load all datasets combined
all_datasets = load_dataset("BeeGass/permutation-groups", name="all", trust_remote_code=True)
```

## Available Configurations

- `s3_data`: Symmetric Group S3
- `s4_data`: Symmetric Group S4
- `s5_data`: Symmetric Group S5
- `s6_data`: Symmetric Group S6
- `s7_data`: Symmetric Group S7
- `a5_data`: Alternating Group A5
- `a6_data`: Alternating Group A6
- `a7_data`: Alternating Group A7
- `all`: All datasets combined

## Dataset Structure

Each dataset contains:
- `input_sequence`: Space-separated sequence of permutation IDs
- `target`: The ID of the composed permutation

## License

MIT
"""
    
    # Upload README
    readme_path = Path("README_DATASET.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.hf_repo,
        repo_type="dataset",
    )
    print("README uploaded successfully!")
    
    # Clean up temporary file
    readme_path.unlink()
    
    print("\nDataset is now ready to use with:")
    print(f'  load_dataset("{args.hf_repo}", name="<config_name>", trust_remote_code=True)')

if __name__ == "__main__":
    main()