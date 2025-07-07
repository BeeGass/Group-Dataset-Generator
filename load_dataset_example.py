from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json
import os
from sympy.combinatorics import Permutation

def load_and_inspect_dataset_config(repo_id: str, config_name: str):
    print(f"\n--- Loading dataset config '{config_name}' from {repo_id} ---")
    try:
        # Load the dataset configuration
        dataset_dict = load_dataset(repo_id, name=config_name, trust_remote_code=True)
        print("Dataset config loaded successfully!")

        # The metadata.json and README.md are now located within the config's subfolder
        # e.g., BeeGass/permutation-groups/data/s3_data/metadata.json
        subfolder_path = f"data/{config_name}"

        # Download and load the metadata.json file
        id_to_perm_map = None
        print("\n--- Attempting to load metadata.json ---")
        try:
            metadata_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder_path}/metadata.json", repo_type="dataset")
            with open(metadata_path, "r") as f:
                id_to_perm_map = json.load(f)
            print("metadata.json loaded successfully!")
            print(f"  Sample from metadata.json (first 5 keys): {dict(list(id_to_perm_map.items())[:5])}")
        except Exception as e:
            print(f"Could not load metadata.json: {e}")

        # Download and load the README.md file
        print("\n--- Attempting to load README.md ---")
        try:
            readme_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder_path}/README.md", repo_type="dataset")
            with open(readme_path, "r") as f:
                readme_content = f.read()
            print("README.md loaded successfully!")
            print(f"  First 500 characters of README.md:\n{readme_content[:500]}...")
        except Exception as e:
            print(f"Could not load README.md: {e}")

        # Print information about the dataset and interpret a sample
        for split, dataset in dataset_dict.items():
            print(f"\nSplit: {split}")
            print(f"  Number of samples: {len(dataset)}")
            print(f"  Features: {dataset.features}")

            # Access metadata from info attribute (if any)
            if hasattr(dataset.info, 'description'):
                print(f"  Description: {dataset.info.description}")
            if hasattr(dataset.info, 'homepage'):
                print(f"  Homepage: {dataset.info.homepage}")
            if hasattr(dataset.info, 'license'):
                print(f"  License: {dataset.info.license}")

            if len(dataset) > 0 and id_to_perm_map:
                print("\n  --- Interpreting a sample ---")
                first_sample = dataset[0]
                input_ids_str = first_sample['input_sequence']
                target_id_str = first_sample['target']

                input_ids = [int(x) for x in input_ids_str.split(' ')]
                target_id = int(target_id_str)

                print(f"    Raw input sequence IDs: {input_ids_str}")
                print(f"    Raw target ID: {target_id_str}")

                # Decode input permutations
                decoded_input_perms = []
                for perm_id in input_ids:
                    decoded_input_perms.append(id_to_perm_map.get(str(perm_id), "Unknown Permutation"))
                print(f"    Decoded input permutations (array form): {decoded_input_perms}")

                # Decode target permutation
                decoded_target_perm = id_to_perm_map.get(str(target_id), "Unknown Permutation")
                print(f"    Decoded target permutation (array form): {decoded_target_perm}")

                # Optional: Verify composition (requires sympy)
                try:
                    # Need to get the degree from the metadata or infer it
                    # For S5, degree is 5, so permutations act on 0,1,2,3,4
                    # The array form is 0-indexed, so Permutation(array_form) is correct
                    group_degree = len(eval(id_to_perm_map['0'])) # Infer degree from first permutation
                    
                    composed_perm_obj = Permutation(group_degree - 1) # Start with identity
                    for p_str in reversed(decoded_input_perms):
                        if p_str != "Unknown Permutation":
                            composed_perm_obj = Permutation(eval(p_str)) * composed_perm_obj
                        else:
                            composed_perm_obj = None # Cannot verify if unknown
                            break
                    
                    if composed_perm_obj:
                        recalculated_target_perm_str = str(composed_perm_obj.array_form)
                        print(f"    Recalculated target permutation (array form): {recalculated_target_perm_str}")
                        if recalculated_target_perm_str == decoded_target_perm:
                            print("    Composition verification: SUCCESS")
                        else:
                            print("    Composition verification: FAILED")
                    else:
                        print("    Composition verification: Skipped due to unknown permutations.")

                except Exception as e:
                    print(f"    Error during composition verification: {e}")

    except Exception as e:
        print(f"Failed to load dataset: {e}")

if __name__ == "__main__":
    central_repo_id = "BeeGass/permutation-groups"

    # Example 1: Load specific configurations using the name parameter
    print("\n=== Example 1: Loading individual datasets ===")
    s3_dataset = load_dataset(central_repo_id, name="s3_data", trust_remote_code=True)
    print(f"S3 dataset loaded: {s3_dataset}")
    
    s7_dataset = load_dataset(central_repo_id, name="s7_data", trust_remote_code=True)
    print(f"S7 dataset loaded: {s7_dataset}")
    
    a7_dataset = load_dataset(central_repo_id, name="a7_data", trust_remote_code=True)
    print(f"A7 dataset loaded: {a7_dataset}")

    # Example 2: Load all datasets combined using the "all" configuration
    print("\n=== Example 2: Loading all datasets combined ===")
    all_datasets = load_dataset(central_repo_id, name="all", trust_remote_code=True)
    print(f"All datasets loaded: {all_datasets}")
    print(f"Total train samples: {len(all_datasets['train'])}")
    print(f"Total test samples: {len(all_datasets['test'])}")

    # Example 3: Detailed inspection of a specific dataset
    print("\n=== Example 3: Detailed inspection of S3 dataset ===")
    load_and_inspect_dataset_config(central_repo_id, "s3_data")

    # Example 4: Show how to use the datasets
    print("\n=== Example 4: Using the datasets ===")
    print("# Load a specific dataset:")
    print('s5_dataset = load_dataset("BeeGass/permutation-groups", name="s5", trust_remote_code=True)')
    print("\n# Load all datasets combined:")
    print('all_datasets = load_dataset("BeeGass/permutation-groups", name="all", trust_remote_code=True)')
    print("\n# Access train/test splits:")
    print('train_data = s5_dataset["train"]')
    print('test_data = s5_dataset["test"]')
