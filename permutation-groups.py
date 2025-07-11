import datasets
import json
import os
import pandas as pd

_DESCRIPTION = "Permutation composition datasets with dynamic filtering by group degree, order, and sequence length."
_HOMEPAGE = "https://huggingface.co/datasets/BeeGass/permutation-groups"
_LICENSE = "MIT"


class PermutationGroupsConfig(datasets.BuilderConfig):
    def __init__(
        self,
        group_type=None,
        min_degree=None,
        max_degree=None,
        min_order=None,
        max_order=None,
        min_len=3,
        max_len=1024,
        **kwargs,
    ):
        """
        Configuration for loading permutation groups.

        Args:
            group_type: Type of group (symmetric, alternating, cyclic, dihedral, klein,
                       quaternion, elementary_abelian, psl, frobenius, mathieu)
            min_degree: Minimum group degree to include
            max_degree: Maximum group degree to include
            min_order: Minimum group order to include
            max_order: Maximum group order to include
            min_len: Minimum sequence length
            max_len: Maximum sequence length
        """
        # Set name based on parameters
        if "name" not in kwargs:
            if group_type:
                kwargs["name"] = group_type
            else:
                kwargs["name"] = "all"

        super().__init__(**kwargs)
        self.group_type = group_type
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.min_order = min_order
        self.max_order = max_order
        self.min_len = min_len
        self.max_len = max_len


class PermutationGroups(datasets.GeneratorBasedBuilder):
    """Permutation groups dataset with dynamic filtering."""

    VERSION = datasets.Version("5.0.0")

    # Define all available group types
    GROUP_TYPES = [
        "symmetric",
        "alternating",
        "cyclic",
        "dihedral",
        "klein",
        "quaternion",
        "elementary_abelian",
        "psl",
        "frobenius",
        "mathieu",
    ]

    BUILDER_CONFIGS = []

    # Add configs for each group type
    for group_type in GROUP_TYPES:
        BUILDER_CONFIGS.append(
            PermutationGroupsConfig(
                name=group_type,
                description=f"{group_type.capitalize()} permutation groups",
                group_type=group_type,
            )
        )

    # Add "all" configuration
    BUILDER_CONFIGS.append(
        PermutationGroupsConfig(
            name="all",
            description="All permutation groups",
            group_type=None,  # Will load all types
        )
    )

    DEFAULT_CONFIG_NAME = "symmetric"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "input_sequence": datasets.Value("string"),
                    "target": datasets.Value("string"),
                    "group_type": datasets.Value("string"),
                    "group_degree": datasets.Value("int32"),
                    "group_order": datasets.Value("int32"),
                    "sequence_length": datasets.Value("int32"),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        # Determine which datasets to load
        if self.config.group_type:
            # Load the superset for this group type
            datasets_to_load = [f"{self.config.group_type}_superset"]
        else:
            # Load all supersets
            datasets_to_load = [
                "symmetric_superset",
                "alternating_superset",
                "cyclic_superset",
                "dihedral_superset",
                "klein_superset",
                "quaternion_superset",
                "elementary_abelian_superset",
                "psl_superset",
                "frobenius_superset",
                "mathieu_superset",
            ]

        # Build file URLs using wildcards
        train_urls = []
        test_urls = []

        for dataset_name in datasets_to_load:
            train_urls.append(f"data/{dataset_name}/train/data-*.arrow")
            test_urls.append(f"data/{dataset_name}/test/data-*.arrow")

        # Download files
        downloaded_files = dl_manager.download({"train": train_urls, "test": test_urls})

        # Flatten the lists of files
        train_files = []
        test_files = []

        for file_list in downloaded_files["train"]:
            if isinstance(file_list, list):
                train_files.extend(file_list)
            else:
                train_files.append(file_list)

        for file_list in downloaded_files["test"]:
            if isinstance(file_list, list):
                test_files.extend(file_list)
            else:
                test_files.append(file_list)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files": train_files,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "files": test_files,
                },
            ),
        ]

    def _generate_examples(self, files):
        """Yield examples with filtering."""
        idx = 0

        for file_path in files:
            # Load the Arrow file
            table = datasets.table.read_table(file_path)

            # Convert to pandas for easier filtering
            df = table.to_pandas()

            # Apply filters
            mask = pd.Series([True] * len(df))

            # Filter by group type (if specified in config)
            if self.config.group_type:
                mask &= df["group_type"] == self.config.group_type

            # Filter by degree
            if self.config.min_degree is not None:
                mask &= df["group_degree"] >= self.config.min_degree
            if self.config.max_degree is not None:
                mask &= df["group_degree"] <= self.config.max_degree

            # Filter by order
            if self.config.min_order is not None:
                mask &= df["group_order"] >= self.config.min_order
            if self.config.max_order is not None:
                mask &= df["group_order"] <= self.config.max_order

            # Filter by sequence length
            if self.config.min_len is not None:
                mask &= df["sequence_length"] >= self.config.min_len
            if self.config.max_len is not None:
                mask &= df["sequence_length"] <= self.config.max_len

            # Apply mask
            filtered_df = df[mask]

            # Yield filtered examples
            for _, row in filtered_df.iterrows():
                yield (
                    idx,
                    {
                        "input_sequence": row["input_sequence"],
                        "target": row["target"],
                        "group_type": row["group_type"],
                        "group_degree": int(row["group_degree"]),
                        "group_order": int(row["group_order"]),
                        "sequence_length": int(row["sequence_length"]),
                    },
                )
                idx += 1
