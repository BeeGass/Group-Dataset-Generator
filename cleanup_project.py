#!/usr/bin/env python3
"""
Cleanup script to organize temporary files and identify unused scripts.
"""

import os
import shutil
from pathlib import Path

def main():
    """Organize and clean up project files."""
    
    # Create archive directory for old/temporary scripts
    archive_dir = Path("archive_temp_scripts")
    archive_dir.mkdir(exist_ok=True)
    
    # Files to keep in root (essential scripts)
    keep_in_root = {
        "generate.py",  # Main generation script
        "generate_individual_groups.py",  # Individual group generation
        "generate_superset.py",  # Superset generation
        "generate_superset_optimized.py",  # Optimized superset generation
        "pyproject.toml",  # Project configuration
        "README.md",  # Project documentation
        "HF_README.md",  # HuggingFace documentation
        ".gitignore",  # Git configuration
        "upload_individual_datasets.sh",  # Active upload script
    }
    
    # Files to archive (temporary/old scripts)
    files_to_archive = [
        "all_groups_config.py",
        "check_quaternion_data.py", 
        "cleanup_remaining.py",
        "example_usage.py",
        "generate_all_datasets.py",
        "list_hf_files.py",
        "permutation-groups.py",
        "split_supersets.py",
        "test_direct_loading.py",
        "upload_clean_datasets.sh",
        "upload_with_complexity_classes.sh",
        "upload_dataset_script.py",
        "upload_hf_readme.py",
    ]
    
    print("Project Cleanup Report")
    print("=" * 50)
    
    # Archive old scripts
    print("\nArchiving temporary/old scripts:")
    archived_count = 0
    for filename in files_to_archive:
        if Path(filename).exists():
            print(f"  - Archiving: {filename}")
            shutil.move(filename, archive_dir / filename)
            archived_count += 1
    
    if archived_count == 0:
        print("  (No files to archive)")
    
    # Report on project structure
    print("\nCurrent project structure:")
    print("  /gdg                 - Group data generators")
    print("  /tests               - Test suite")
    print("  /examples            - Usage examples")
    print("  /individual_datasets - Generated datasets")
    
    if archive_dir.exists() and any(archive_dir.iterdir()):
        print(f"  /{archive_dir}      - Archived temporary scripts")
    
    # Check for additional cleanup opportunities
    print("\nAdditional cleanup suggestions:")
    
    # Check for __pycache__ directories
    pycache_dirs = list(Path(".").rglob("__pycache__"))
    if pycache_dirs:
        print(f"  - Found {len(pycache_dirs)} __pycache__ directories (can be removed)")
    
    # Check for .pyc files
    pyc_files = list(Path(".").rglob("*.pyc"))
    if pyc_files:
        print(f"  - Found {len(pyc_files)} .pyc files (can be removed)")
    
    # Check for .DS_Store files (macOS)
    ds_store_files = list(Path(".").rglob(".DS_Store"))
    if ds_store_files:
        print(f"  - Found {len(ds_store_files)} .DS_Store files (can be removed)")
    
    print("\n" + "=" * 50)
    print("Cleanup complete!")
    print(f"\nArchived {archived_count} temporary scripts to '{archive_dir}/'")
    print("\nTo fully clean the project, you can:")
    print("  1. Remove the archive directory: rm -rf archive_temp_scripts/")
    print("  2. Clean Python cache: find . -type d -name __pycache__ -exec rm -rf {} +")
    print("  3. Clean .pyc files: find . -name '*.pyc' -delete")
    print("  4. Clean .DS_Store: find . -name '.DS_Store' -delete")


if __name__ == "__main__":
    main()