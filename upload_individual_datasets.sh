#!/bin/bash
# Upload individual datasets to HuggingFace
# Usage: ./upload_individual_datasets.sh [pattern1] [pattern2] ...
# Examples:
#   ./upload_individual_datasets.sh s3 s4 s5 s6 s7
#   ./upload_individual_datasets.sh s* a* c*
#   ./upload_individual_datasets.sh s[3-7] a[3-7]
#   ./upload_individual_datasets.sh  # uploads all

echo "Uploading individual datasets to HuggingFace..."

# If no arguments provided, upload all
if [ $# -eq 0 ]; then
    echo "No patterns specified, uploading ALL datasets"
    patterns=("*")
else
    echo "Upload patterns: $@"
    patterns=("$@")
fi

echo ""

# Counter for progress
uploaded=0
skipped=0

# Process each dataset directory
for dataset_dir in individual_datasets/*_data; do
    if [ -d "$dataset_dir" ]; then
        # Get the base name and remove the "_data" suffix
        dataset_name=$(basename "$dataset_dir")
        clean_name=${dataset_name%_data}
        
        # Check if this dataset matches any of the patterns
        match_found=false
        for pattern in "${patterns[@]}"; do
            if [[ "$clean_name" == $pattern ]]; then
                match_found=true
                break
            fi
        done
        
        if [ "$match_found" = false ]; then
            skipped=$((skipped + 1))
            continue
        fi
        
        echo "========================================"
        echo "Uploading $clean_name..."
        echo "========================================"
        
        uv run huggingface-cli upload \
            BeeGass/Group-Theory-Collection \
            "$dataset_dir" \
            "data/$clean_name" \
            --repo-type dataset \
            --commit-message "Add individual dataset: $clean_name"
        
        uploaded=$((uploaded + 1))
        echo "✓ $clean_name uploaded"
        echo ""
    fi
done

echo "========================================"
echo "Upload complete!"
echo "Uploaded: $uploaded datasets"
echo "Skipped: $skipped datasets"
echo "========================================"
echo ""
echo "Users can now load datasets with:"
echo '  load_dataset("BeeGass/Group-Theory-Collection", data_dir="data/s5")'
echo '  load_dataset("BeeGass/Group-Theory-Collection", data_dir="data/a4")'
echo '  load_dataset("BeeGass/Group-Theory-Collection", data_dir="data/c10")'