#!/usr/bin/env python3
"""List all files in the HuggingFace dataset repository."""

from huggingface_hub import HfApi


def main():
    api = HfApi()

    # List all files in the repository
    try:
        files = api.list_repo_files(
            repo_id="BeeGass/permutation-groups", repo_type="dataset"
        )

        # Organize files by directory
        organized = {}

        for file in files:
            parts = file.split("/")
            if len(parts) == 1:
                # Root level file
                if "root" not in organized:
                    organized["root"] = []
                organized["root"].append(file)
            else:
                # File in a directory
                dir_name = parts[0]
                if dir_name not in organized:
                    organized[dir_name] = []
                organized[dir_name].append("/".join(parts[1:]))

        # Print organized structure
        print("=== Files in BeeGass/permutation-groups ===")
        print()

        # First show root files
        if "root" in organized:
            print("Root level files:")
            for file in sorted(organized["root"]):
                print(f"  - {file}")
            print()

        # Then show directories and their contents
        for dir_name in sorted(organized.keys()):
            if dir_name != "root":
                print(f"Directory: {dir_name}/")
                for file in sorted(organized[dir_name]):
                    print(f"  - {file}")
                print()

        # Check for specific directories mentioned
        print("=== Analysis ===")

        # Check for z6_data
        if "z6_data" in organized:
            print("⚠️  WARNING: Found z6_data directory (old data)")
        else:
            print("✅ GOOD: No z6_data directory found")

        # Check for superset directories
        superset_dirs = [
            d for d in organized.keys() if "superset" in d.lower() and d != "root"
        ]
        if superset_dirs:
            print(f"\nSuperset directories found: {superset_dirs}")
        else:
            print("\nNo superset directories found")

        # Check for dataset script
        dataset_scripts = []
        for dir_name, files in organized.items():
            if dir_name == "root":
                for file in files:
                    if file.endswith(".py"):
                        dataset_scripts.append(file)
            else:
                for file in files:
                    if file.endswith(".py"):
                        dataset_scripts.append(f"{dir_name}/{file}")

        if dataset_scripts:
            print(f"\nDataset scripts found:")
            for script in dataset_scripts:
                print(f"  - {script}")

        # Count total files
        total_files = sum(len(files) for files in organized.values())
        print(f"\nTotal files: {total_files}")

    except Exception as e:
        print(f"Error accessing repository: {e}")


if __name__ == "__main__":
    main()
