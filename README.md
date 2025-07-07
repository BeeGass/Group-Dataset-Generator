# s5-data-gen

This script generates permutation composition datasets for groups like S5 and A5.

## How to Use

1.  **Install dependencies:**

    ```bash
uv pip install -r requirements.txt
    ```

2.  **Run the script:**

    *   **Generate S5 dataset and save locally:**

        ```bash
python generate.py \
          --group-name S5 \
          --num-samples 50000 \
          --max-len 30 \
          --output-dir ./s5_data
        ```

    *   **Generate A5 dataset and push to Hugging Face:**

        ```bash
python generate.py \
          --group-name A5 \
          --num-samples 20000 \
          --output-dir ./a5_data \
          --hf-repo "your-username/a5-permutation-benchmark"
        ```

