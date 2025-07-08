#!/usr/bin/env python3
"""Upload the HuggingFace README."""

from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="HF_README.md",
    path_in_repo="README.md",
    repo_id="BeeGass/permutation-groups",
    repo_type="dataset",
)
print("✓ HuggingFace README uploaded successfully!")