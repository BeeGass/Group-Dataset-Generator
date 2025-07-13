#!/bin/bash
# Upload individual datasets to HuggingFace
# Usage: ./upload_datasets.sh [--overwrite] [pattern1] [pattern2] ...
# Examples:
#   ./upload_datasets.sh s3 s4 s5 s6 s7
#   ./upload_datasets.sh s* a* c*
#   ./upload_datasets.sh s[3-7] a[3-7]
#   ./upload_datasets.sh  # uploads all
#   ./upload_datasets.sh --overwrite s3 s4  # overwrite existing datasets

echo "Uploading individual datasets to HuggingFace..."

# Check for --overwrite flag
overwrite=false
if [[ "$1" == "--overwrite" ]]; then
    overwrite=true
    shift  # Remove --overwrite from arguments
    echo "OVERWRITE mode enabled - will delete existing data before uploading"
fi

# Define TC^0 and NC^1 groups
TC0_GROUPS=(
    # Cyclic groups
    c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20
    c21 c22 c23 c24 c25 c26 c27 c28 c29 c30
    # Dihedral groups
    d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15 d16 d17 d18 d19 d20
    # Klein four-group
    v4
    # Quaternion groups
    q8 q16 q32
    # Elementary abelian groups
    z2_1 z2_2 z2_3 z2_4 z2_5
    z3_1 z3_2 z3_3 z3_4
    z5_1 z5_2 z5_3 z5_4
    # Frobenius groups
    f20 f21
    # Solvable symmetric and alternating
    s3 s4 a3 a4
    # Solvable PSL
    psl2_2 psl2_3
)

NC1_GROUPS=(
    # Non-solvable symmetric
    s5 s6 s7 s8 s9
    # Non-solvable alternating
    a5 a6 a7 a8 a9
    # Non-solvable PSL
    psl2_5 psl2_7 psl2_8 psl2_9 psl2_11
    psl3_2 psl3_3
    # Mathieu groups
    m11 m12
)

# If no arguments provided (after potential --overwrite), upload all
if [ $# -eq 0 ]; then
    echo "No patterns specified, uploading ALL datasets"
    patterns=("*")
else
    echo "Upload patterns: $@"
    patterns=("$@")
fi

# Handle special keywords
expanded_patterns=()
for pattern in "${patterns[@]}"; do
    case "$pattern" in
        "tc0")
            echo "Expanding 'tc0' to all TC0 groups..."
            expanded_patterns+=("${TC0_GROUPS[@]}")
            ;;
        "nc1")
            echo "Expanding 'nc1' to all NC1 groups..."
            expanded_patterns+=("${NC1_GROUPS[@]}")
            ;;
        "all")
            echo "Expanding 'all' to all datasets..."
            expanded_patterns=("*")
            ;;
        *)
            expanded_patterns+=("$pattern")
            ;;
    esac
done

# Use expanded patterns if any special keywords were found
if [ ${#expanded_patterns[@]} -gt 0 ]; then
    patterns=("${expanded_patterns[@]}")
fi

echo ""

# Function to check if a group is in TC0
is_tc0_group() {
    local group=$1
    for tc0_group in "${TC0_GROUPS[@]}"; do
        if [[ "$group" == "$tc0_group" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to check if a group is in NC1
is_nc1_group() {
    local group=$1
    for nc1_group in "${NC1_GROUPS[@]}"; do
        if [[ "$group" == "$nc1_group" ]]; then
            return 0
        fi
    done
    return 1
}

# Counter for progress
uploaded=0
skipped=0

# Process each dataset directory
for dataset_dir in datasets/*_data; do
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
        
        # Clean up any cache files before upload
        echo "Cleaning cache files from $dataset_dir..."
        find "$dataset_dir" -name "cache*.arrow" -type f -delete 2>/dev/null || true
        
        # Upload to data/ directory
        upload_success=true
        if [ "$overwrite" = true ]; then
            if ! uv run huggingface-cli upload \
                BeeGass/Group-Theory-Collection \
                "$dataset_dir" \
                "data/$clean_name" \
                --repo-type dataset \
                --commit-message "Update dataset: $clean_name" \
                --delete "data/$clean_name/**" \
                --exclude "**/cache*.arrow" \
                --exclude "cache*.arrow" \
                --exclude "*/cache*.arrow"; then
                echo "✗ Failed to upload $clean_name to data/"
                upload_success=false
            fi
        else
            if ! uv run huggingface-cli upload \
                BeeGass/Group-Theory-Collection \
                "$dataset_dir" \
                "data/$clean_name" \
                --repo-type dataset \
                --commit-message "Add individual dataset: $clean_name" \
                --exclude "**/cache*.arrow" \
                --exclude "cache*.arrow" \
                --exclude "*/cache*.arrow"; then
                echo "✗ Failed to upload $clean_name to data/"
                upload_success=false
            fi
        fi
        
        # Also upload to TC0/ or NC1/ if applicable
        if [ "$upload_success" = true ]; then
            if is_tc0_group "$clean_name"; then
                echo "Also uploading to TC0/ directory..."
                if [ "$overwrite" = true ]; then
                    if ! uv run huggingface-cli upload \
                        BeeGass/Group-Theory-Collection \
                        "$dataset_dir" \
                        "TC0/$clean_name" \
                        --repo-type dataset \
                        --commit-message "Update TC0 dataset: $clean_name" \
                        --delete "TC0/$clean_name/**" \
                        --exclude "**/cache*.arrow" \
                --exclude "cache*.arrow" \
                --exclude "*/cache*.arrow"; then
                        echo "✗ Failed to upload $clean_name to TC0/"
                        upload_success=false
                    fi
                else
                    if ! uv run huggingface-cli upload \
                        BeeGass/Group-Theory-Collection \
                        "$dataset_dir" \
                        "TC0/$clean_name" \
                        --repo-type dataset \
                        --commit-message "Add TC0 dataset: $clean_name" \
                        --exclude "**/cache*.arrow" \
                --exclude "cache*.arrow" \
                --exclude "*/cache*.arrow"; then
                        echo "✗ Failed to upload $clean_name to TC0/"
                        upload_success=false
                    fi
                fi
            elif is_nc1_group "$clean_name"; then
                echo "Also uploading to NC1/ directory..."
                if [ "$overwrite" = true ]; then
                    if ! uv run huggingface-cli upload \
                        BeeGass/Group-Theory-Collection \
                        "$dataset_dir" \
                        "NC1/$clean_name" \
                        --repo-type dataset \
                        --commit-message "Update NC1 dataset: $clean_name" \
                        --delete "NC1/$clean_name/**" \
                        --exclude "**/cache*.arrow" \
                --exclude "cache*.arrow" \
                --exclude "*/cache*.arrow"; then
                        echo "✗ Failed to upload $clean_name to NC1/"
                        upload_success=false
                    fi
                else
                    if ! uv run huggingface-cli upload \
                        BeeGass/Group-Theory-Collection \
                        "$dataset_dir" \
                        "NC1/$clean_name" \
                        --repo-type dataset \
                        --commit-message "Add NC1 dataset: $clean_name" \
                        --exclude "**/cache*.arrow" \
                --exclude "cache*.arrow" \
                --exclude "*/cache*.arrow"; then
                        echo "✗ Failed to upload $clean_name to NC1/"
                        upload_success=false
                    fi
                fi
            fi
        fi
        
        if [ "$upload_success" = true ]; then
            uploaded=$((uploaded + 1))
            echo "✓ $clean_name uploaded successfully"
        else
            skipped=$((skipped + 1))
            echo "✗ $clean_name upload failed"
            # Add a small delay to avoid rate limiting
            echo "Waiting 5 seconds before next upload to avoid rate limiting..."
            sleep 5
        fi
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