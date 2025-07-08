import datasets
import json
import os
import pyarrow as pa

_DESCRIPTION = "A collection of permutation composition datasets for various symmetric and alternating groups."
_HOMEPAGE = "https://huggingface.co/datasets/BeeGass/permutation-groups"
_LICENSE = "MIT"

class PermutationGroupsConfig(datasets.BuilderConfig):
    def __init__(self, *args, group_name=None, group_degree=None, group_order=None, data_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_name = group_name
        self.group_degree = group_degree
        self.group_order = group_order
        self.data_dir = data_dir

class PermutationGroups(datasets.ArrowBasedBuilder):
    """Use ArrowBasedBuilder for better performance with Arrow files."""
    
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        PermutationGroupsConfig(
            name="s3_data",
            description="Permutation Composition Dataset for the Symmetric Group S3.",
            group_name="S3",
            group_degree=3,
            group_order=6,
            data_dir="data/s3_data",
        ),
        PermutationGroupsConfig(
            name="s4_data",
            description="Permutation Composition Dataset for the Symmetric Group S4.",
            group_name="S4",
            group_degree=4,
            group_order=24,
            data_dir="data/s4_data",
        ),
        PermutationGroupsConfig(
            name="s5_data",
            description="Permutation Composition Dataset for the Symmetric Group S5.",
            group_name="S5",
            group_degree=5,
            group_order=120,
            data_dir="data/s5_data",
        ),
        PermutationGroupsConfig(
            name="s6_data",
            description="Permutation Composition Dataset for the Symmetric Group S6.",
            group_name="S6",
            group_degree=6,
            group_order=720,
            data_dir="data/s6_data",
        ),
        PermutationGroupsConfig(
            name="s7_data",
            description="Permutation Composition Dataset for the Symmetric Group S7.",
            group_name="S7",
            group_degree=7,
            group_order=5040,
            data_dir="data/s7_data",
        ),
        PermutationGroupsConfig(
            name="a5_data",
            description="Permutation Composition Dataset for the Alternating Group A5.",
            group_name="A5",
            group_degree=5,
            group_order=60,
            data_dir="data/a5_data",
        ),
        PermutationGroupsConfig(
            name="a6_data",
            description="Permutation Composition Dataset for the Alternating Group A6.",
            group_name="A6",
            group_degree=6,
            group_order=360,
            data_dir="data/a6_data",
        ),
        PermutationGroupsConfig(
            name="a7_data",
            description="Permutation Composition Dataset for the Alternating Group A7.",
            group_name="A7",
            group_degree=7,
            group_order=2520,
            data_dir="data/a7_data",
        ),
        PermutationGroupsConfig(
            name="all",
            description="All Permutation Composition Datasets (S3-S7 and A5-A7).",
            group_name="All",
            group_degree=None,
            group_order=None,
            data_dir=None,  # Special handling for 'all'
        ),
    ]

    DEFAULT_CONFIG_NAME = "s5_data"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "input_sequence": datasets.Value("string"),
                "target": datasets.Value("string"),
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        # Handle the "all" configuration specially
        if self.config.name == "all":
            # Get all individual dataset configurations
            all_configs = ["s3_data", "s4_data", "s5_data", "s6_data", "s7_data", 
                          "a5_data", "a6_data", "a7_data"]
            
            # Download all arrow files
            train_files = []
            test_files = []
            
            for config in all_configs:
                data_urls = {
                    "train": f"data/{config}/train/data-00000-of-00001.arrow",
                    "test": f"data/{config}/test/data-00000-of-00001.arrow",
                }
                downloaded = dl_manager.download(data_urls)
                train_files.append(downloaded["train"])
                test_files.append(downloaded["test"])
            
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
        else:
            # Single configuration
            # Download the actual data files
            data_urls = {
                "train": f"{self.config.data_dir}/train/data-00000-of-00001.arrow",
                "test": f"{self.config.data_dir}/test/data-00000-of-00001.arrow",
            }
            
            downloaded_files = dl_manager.download(data_urls)
            
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "files": [downloaded_files["train"]],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "files": [downloaded_files["test"]],
                    },
                ),
            ]

    def _generate_tables(self, files):
        """Yield arrow tables directly for better performance."""
        for file_idx, file in enumerate(files):
            # Load the dataset using the datasets library format
            dataset = datasets.Dataset.from_file(file)
            # Get the underlying Arrow table
            table = dataset.data.table
            yield file_idx, table