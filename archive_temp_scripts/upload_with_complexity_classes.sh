#!/bin/bash
# Upload datasets with complexity class organization
# Usage: ./upload_with_complexity_classes.sh [all|tc0|nc1]
# Examples:
#   ./upload_with_complexity_classes.sh        # uploads everything (default)
#   ./upload_with_complexity_classes.sh all    # uploads everything
#   ./upload_with_complexity_classes.sh tc0    # uploads only TC0 directory
#   ./upload_with_complexity_classes.sh nc1    # uploads only NC1 directory

MODE=${1:-all}

# Validate mode
if [ "$MODE" != "all" ] && [ "$MODE" != "tc0" ] && [ "$MODE" != "nc1" ]; then
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: $0 [all|tc0|nc1]"
    echo "  all - Upload everything (default)"
    echo "  tc0 - Upload only TC0 directory and its datasets"
    echo "  nc1 - Upload only NC1 directory and its datasets"
    exit 1
fi

echo "Upload mode: $MODE"
echo ""

# Define TC0 (solvable) groups
TC0_GROUPS=(
    # Small symmetric/alternating (solvable)
    "s3" "s4" "a3" "a4"
    
    # All cyclic (solvable abelian)
    "c3" "c4" "c5" "c6" "c7" "c8" "c9" "c10" "c12" "c15" "c20" "c25" "c30"
    
    # All dihedral (solvable)
    "d3" "d4" "d5" "d6" "d7" "d8" "d9" "d10" "d12" "d15" "d20"
    
    # Klein four-group (solvable abelian)
    "v4"
    
    # All quaternion (solvable)
    "q8" "q16" "q32"
    
    # All elementary abelian (solvable)
    "z2_2" "z2_3" "z2_4" "z2_5" "z3_1" "z3_2" "z3_3" "z5_1" "z5_2"
    
    # Frobenius (solvable)
    "f20" "f21"
)

# Define NC1 (non-solvable) groups
NC1_GROUPS=(
    # Symmetric n≥5 (non-solvable) - up to S9
    "s5" "s6" "s7" "s8" "s9"
    
    # Alternating n≥5 (non-solvable) - up to A9
    "a5" "a6" "a7" "a8" "a9"
    
    # PSL groups (non-solvable)
    "psl2_5" "psl2_7"
    
    # Mathieu groups (non-solvable sporadic simple)
    "m11" "m12"
)

# Function to check if a dataset already exists in the HF repo
check_dataset_exists() {
    local target_path=$1
    
    # Create a temporary directory for the check
    local temp_dir="/tmp/hf_check_$$"
    mkdir -p "$temp_dir"
    
    # Try to download the dataset_dict.json file from the target path
    # This file should exist at the root of each dataset
    uv run huggingface-cli download \
        BeeGass/permutation-groups \
        --repo-type dataset \
        --include "${target_path}/dataset_dict.json" \
        --local-dir "$temp_dir" \
        --quiet 2>/dev/null
    
    # Check if the dataset_dict.json file was downloaded
    if [ -f "$temp_dir/${target_path}/dataset_dict.json" ]; then
        # File found - dataset exists
        rm -rf "$temp_dir"
        return 0
    else
        # No file found - dataset doesn't exist
        rm -rf "$temp_dir"
        return 1
    fi
}

# Function to upload a dataset to a specific path
upload_dataset() {
    local dataset_name=$1
    local target_path=$2
    local dataset_dir="individual_datasets/${dataset_name}_data"
    
    if [ -d "$dataset_dir" ]; then
        # Check if dataset already exists
        if check_dataset_exists "$target_path"; then
            echo "✓ $dataset_name already exists at $target_path, skipping..."
            return 2  # Return 2 to indicate skipped
        fi
        
        echo "Uploading $dataset_name to $target_path..."
        uv run huggingface-cli upload \
            BeeGass/permutation-groups \
            "$dataset_dir" \
            "$target_path" \
            --repo-type dataset \
            --commit-message "Add $dataset_name to $target_path" \
            --quiet
        
        if [ $? -eq 0 ]; then
            # Sleep to avoid rate limiting (3-5 seconds between uploads)
            sleep 4
            return 0
        else
            echo "✗ Failed to upload $dataset_name"
            return 1
        fi
    else
        echo "Warning: $dataset_dir not found"
        return 1
    fi
}

# Upload all datasets to data/ first (only if mode is 'all')
if [ "$MODE" = "all" ]; then
    echo "========================================"
    echo "PHASE 1: Uploading all datasets to data/"
    echo "========================================"
    echo ""

    uploaded_data=0
    skipped_data=0
    for dataset_name in "${TC0_GROUPS[@]}" "${NC1_GROUPS[@]}"; do
        upload_dataset "$dataset_name" "data/$dataset_name"
        result=$?
        if [ $result -eq 0 ]; then
            uploaded_data=$((uploaded_data + 1))
        elif [ $result -eq 2 ]; then
            skipped_data=$((skipped_data + 1))
        fi
    done

    echo ""
    echo "✓ Uploaded $uploaded_data new datasets to data/"
    if [ $skipped_data -gt 0 ]; then
        echo "✓ Skipped $skipped_data existing datasets"
    fi
    echo ""
fi

# Upload TC0 groups (if mode is 'all' or 'tc0')
if [ "$MODE" = "all" ] || [ "$MODE" = "tc0" ]; then
    echo "========================================"
    echo "PHASE 2: Uploading TC⁰ (solvable) groups"
    echo "========================================"
    echo ""

    uploaded_tc0=0
    skipped_tc0=0
    for dataset_name in "${TC0_GROUPS[@]}"; do
        upload_dataset "$dataset_name" "TC0/$dataset_name"
        result=$?
        if [ $result -eq 0 ]; then
            uploaded_tc0=$((uploaded_tc0 + 1))
        elif [ $result -eq 2 ]; then
            skipped_tc0=$((skipped_tc0 + 1))
        fi
    done

    # Create TC0 README if it doesn't exist
    if check_dataset_exists "TC0/README.md"; then
        echo "✓ TC0/README.md already exists, skipping..."
    else
        echo "Creating TC0/README.md..."
cat > tc0_readme.md << 'EOF'
# TC⁰ Groups (Solvable)

This directory contains all solvable permutation groups from our dataset. These groups can theoretically be solved by constant-depth threshold circuits (TC⁰), which means models like Transformers and standard SSMs should be able to learn them.

## Groups in this category:

### Small Symmetric/Alternating (Solvable)
- S3 (order 6)
- S4 (order 24)
- A3 (order 3)
- A4 (order 12)

### Cyclic Groups (All solvable abelian)
- C3 through C30

### Dihedral Groups (All solvable)
- D3 through D20

### Other Solvable Groups
- V4: Klein four-group (order 4)
- Q8, Q16, Q32: Quaternion groups
- Elementary abelian groups (Z_p^k)
- F20, F21: Frobenius groups

## Key Property
All these groups have a composition series with abelian quotients, making them solvable and placing them in TC⁰.
EOF

        uv run huggingface-cli upload \
            BeeGass/permutation-groups \
            tc0_readme.md \
            TC0/README.md \
            --repo-type dataset \
            --commit-message "Add TC0 README"

        rm tc0_readme.md
    fi

    echo ""
    echo "✓ Uploaded $uploaded_tc0 new TC⁰ groups"
    if [ $skipped_tc0 -gt 0 ]; then
        echo "✓ Skipped $skipped_tc0 existing TC⁰ groups"
    fi
    echo ""
fi

# Upload NC1 groups (if mode is 'all' or 'nc1')
if [ "$MODE" = "all" ] || [ "$MODE" = "nc1" ]; then
    echo "========================================"
    echo "PHASE 3: Uploading NC¹ (non-solvable) groups"
    echo "========================================"
    echo ""

    uploaded_nc1=0
    skipped_nc1=0
    for dataset_name in "${NC1_GROUPS[@]}"; do
        upload_dataset "$dataset_name" "NC1/$dataset_name"
        result=$?
        if [ $result -eq 0 ]; then
            uploaded_nc1=$((uploaded_nc1 + 1))
        elif [ $result -eq 2 ]; then
            skipped_nc1=$((skipped_nc1 + 1))
        fi
    done

    # Create NC1 README if it doesn't exist
    if check_dataset_exists "NC1/README.md"; then
        echo "✓ NC1/README.md already exists, skipping..."
    else
        echo "Creating NC1/README.md..."
cat > nc1_readme.md << 'EOF'
# NC¹ Groups (Non-Solvable)

This directory contains all non-solvable permutation groups from our dataset. Computing compositions in these groups is NC¹-complete, meaning it requires at least logarithmic-depth circuits. This is beyond the computational power of TC⁰ models like Transformers and SSMs, leading to the "Illusion of State" phenomenon.

## Groups in this category:

### Symmetric Groups (n ≥ 5)
- S5 (order 120) - First non-solvable symmetric group
- S6 (order 720)
- S7 (order 5,040)
- S8 (order 40,320)
- S9 (order 362,880)

### Alternating Groups (n ≥ 5)
- A5 (order 60) - The smallest non-solvable group
- A6 (order 360)
- A7 (order 2,520)
- A8 (order 20,160)
- A9 (order 181,440)

### Projective Special Linear Groups
- PSL(2,5) (order 60) - Isomorphic to A5
- PSL(2,7) (order 168)

### Sporadic Simple Groups
- M11 (order 7,920) - Mathieu group
- M12 (order 95,040) - Mathieu group

## Key Property
These groups are all non-solvable (simple or containing simple subgroups), placing them in NC¹-complete. The boundary at A5/S5 marks the theoretical limit of what TC⁰ models can solve.
EOF

        uv run huggingface-cli upload \
            BeeGass/permutation-groups \
            nc1_readme.md \
            NC1/README.md \
            --repo-type dataset \
            --commit-message "Add NC1 README"

        rm nc1_readme.md
    fi

    echo ""
    echo "✓ Uploaded $uploaded_nc1 new NC¹ groups"
    if [ $skipped_nc1 -gt 0 ]; then
        echo "✓ Skipped $skipped_nc1 existing NC¹ groups"
    fi
    echo ""
fi

# Summary
echo "========================================"
echo "UPLOAD COMPLETE!"
echo "========================================"
echo ""
echo "Upload summary:"
if [ "$MODE" = "all" ]; then
    echo "  data/          : $uploaded_data new, $skipped_data skipped"
    echo "  TC0/           : $uploaded_tc0 new, $skipped_tc0 skipped"
    echo "  NC1/           : $uploaded_nc1 new, $skipped_nc1 skipped"
elif [ "$MODE" = "tc0" ]; then
    echo "  TC0/           : $uploaded_tc0 new, $skipped_tc0 skipped"
elif [ "$MODE" = "nc1" ]; then
    echo "  NC1/           : $uploaded_nc1 new, $skipped_nc1 skipped"
fi
echo ""
echo "Users can now load datasets in multiple ways:"
echo '  load_dataset("BeeGass/permutation-groups", data_dir="data/s5")'
echo '  load_dataset("BeeGass/permutation-groups", data_dir="TC0/c10")'
echo '  load_dataset("BeeGass/permutation-groups", data_dir="NC1/a5")'