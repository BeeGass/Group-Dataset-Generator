#!/usr/bin/env python3
"""Clean up remaining old data directories from HuggingFace."""

from huggingface_hub import HfApi
import time

# The only dataset repository to keep
MAIN_DATASET = "permutation-groups"

# Within the main dataset, these are the data directories to keep
KEEP_DATA_DIRS = [
    "symmetric_superset",
    "alternating_superset",
    "cyclic_superset",
    "dihedral_superset",
    "psl25_data",  # Special groups that don't fit superset pattern
    "f20_data",
]


def main():
    api = HfApi()

    print("=== Cleaning up remaining old data directories ===")

    # List all files in the main dataset
    main_repo_id = f"BeeGass/{MAIN_DATASET}"
    try:
        all_files = api.list_repo_files(repo_id=main_repo_id, repo_type="dataset")

        # Find data directories to delete
        data_dirs_to_delete = set()
        files_to_delete = []

        for file_path in all_files:
            if file_path.startswith("data/") and "/" in file_path[5:]:
                # Extract the data directory name
                data_dir = file_path.split("/")[1]

                # Check if it's an old individual dataset (not a superset or special group)
                if data_dir not in KEEP_DATA_DIRS and not data_dir.endswith(
                    "_superset"
                ):
                    data_dirs_to_delete.add(data_dir)
                    files_to_delete.append(file_path)

        if files_to_delete:
            print(
                f"\nFound {len(data_dirs_to_delete)} old data directories with {len(files_to_delete)} files to delete:"
            )
            for data_dir in sorted(data_dirs_to_delete):
                print(f"  - data/{data_dir}")

            print(f"\nWill keep these data directories:")
            for keep_dir in KEEP_DATA_DIRS:
                print(f"  - data/{keep_dir}")

            # Delete each file
            deleted_count = 0
            failed_count = 0

            for i, file_path in enumerate(files_to_delete):
                try:
                    print(f"[{i + 1}/{len(files_to_delete)}] Deleting {file_path}...")
                    api.delete_file(
                        path_in_repo=file_path,
                        repo_id=main_repo_id,
                        repo_type="dataset",
                    )
                    deleted_count += 1

                    # Rate limiting - pause every 10 files
                    if deleted_count % 10 == 0:
                        time.sleep(2)
                except Exception as e:
                    print(f"✗ Error deleting {file_path}: {e}")
                    failed_count += 1
                    # If we hit rate limit, wait longer
                    if "429" in str(e):
                        print("Rate limited - waiting 30 seconds...")
                        time.sleep(30)

            print(f"\n✓ Deleted {deleted_count} files")
            if failed_count > 0:
                print(f"✗ Failed to delete {failed_count} files")
        else:
            print("No old data directories found to delete.")

    except Exception as e:
        print(f"Error accessing repository: {e}")

    print("\n=== Cleanup complete! ===")


if __name__ == "__main__":
    main()
