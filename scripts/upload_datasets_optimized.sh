#!/bin/bash
# Optimized upload script using upload-large-folder for better performance
# Usage: ./upload_datasets_optimized.sh [--overwrite] [pattern1] [pattern2] ...

echo "Optimized HuggingFace dataset upload script"
echo "=========================================="

# Check for --overwrite flag
overwrite=false
if [[ "$1" == "--overwrite" ]]; then
    overwrite=true
    shift
    echo "OVERWRITE mode enabled"
fi

# Repository ID
REPO_ID="BeeGass/Group-Theory-Collection"

# Number of parallel workers for upload
NUM_WORKERS=4

# Define all groups (same as before)
TC0_GROUPS=(
    c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20
    c21 c22 c23 c24 c25 c26 c27 c28 c29 c30
    d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15 d16 d17 d18 d19 d20
    v4 q8 q16 q32
    z2_1 z2_2 z2_3 z2_4 z2_5 z3_1 z3_2 z3_3 z3_4 z5_1 z5_2 z5_3 z5_4
    f20 f21 s3 s4 a3 a4 psl2_2 psl2_3
)

NC1_GROUPS=(
    s5 s6 s7 s8 s9 a5 a6 a7 a8 a9
    psl2_4 psl2_5 psl2_7 psl2_8 psl2_9 psl2_11
    psl3_2 psl3_3 psl3_4 psl3_5 m11 m12
)

# Parse patterns
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

if [ ${#expanded_patterns[@]} -gt 0 ]; then
    patterns=("${expanded_patterns[@]}")
fi

# Create temporary directory structure for batch upload
TEMP_DIR=$(mktemp -d)
echo "Creating temporary upload structure in: $TEMP_DIR"

# Organize datasets
DATA_DIR="$TEMP_DIR/data"
TC0_DIR="$TEMP_DIR/TC0"
NC1_DIR="$TEMP_DIR/NC1"

mkdir -p "$DATA_DIR" "$TC0_DIR" "$NC1_DIR"

# Counter
copied=0
total=0

# Process each dataset directory
for dataset_dir in datasets/*_data; do
    if [ -d "$dataset_dir" ]; then
        dataset_name=$(basename "$dataset_dir")
        clean_name=${dataset_name%_data}
        
        # Check if this dataset matches any pattern
        match_found=false
        for pattern in "${patterns[@]}"; do
            # Use glob pattern matching
            if [[ "$clean_name" == $pattern ]]; then
                match_found=true
                break
            fi
        done
        
        if [ "$match_found" = false ]; then
            continue
        fi
        
        total=$((total + 1))
        echo "Processing $clean_name..."
        
        # Clean cache files
        find "$dataset_dir" -name "cache*.arrow" -type f -delete 2>/dev/null || true
        
        # Copy to data directory
        cp -r "$dataset_dir" "$DATA_DIR/$clean_name"
        
        # Also copy to TC0 or NC1 if applicable
        is_tc0=false
        is_nc1=false
        
        for tc0_group in "${TC0_GROUPS[@]}"; do
            if [[ "$clean_name" == "$tc0_group" ]]; then
                cp -r "$dataset_dir" "$TC0_DIR/$clean_name"
                is_tc0=true
                break
            fi
        done
        
        if [ "$is_tc0" = false ]; then
            for nc1_group in "${NC1_GROUPS[@]}"; do
                if [[ "$clean_name" == "$nc1_group" ]]; then
                    cp -r "$dataset_dir" "$NC1_DIR/$clean_name"
                    is_nc1=true
                    break
                fi
            done
        fi
        
        copied=$((copied + 1))
    fi
done

echo ""
echo "Prepared $copied datasets for upload"
echo "=========================================="

# Upload using upload-large-folder
if [ $copied -gt 0 ]; then
    echo "Starting batch upload with $NUM_WORKERS workers..."
    
    # Clean up remote if overwrite is enabled
    if [ "$overwrite" = true ]; then
        echo "Cleaning remote repository..."
        # Note: This would require using the Python API or manual deletion
        echo "WARNING: --overwrite with upload-large-folder requires manual cleanup"
    fi
    
    # Upload the entire structure at once
    uv run huggingface-cli upload-large-folder \
        "$REPO_ID" \
        "$TEMP_DIR" \
        --repo-type dataset \
        --num-workers "$NUM_WORKERS" \
        --exclude "**/cache*.arrow" \
        --exclude "cache*.arrow" \
        --exclude "*/cache*.arrow"
    
    upload_status=$?
    
    if [ $upload_status -eq 0 ]; then
        echo "✓ Upload completed successfully!"
    else
        echo "✗ Upload failed with status $upload_status"
    fi
else
    echo "No datasets matched the patterns"
fi

# Cleanup
echo "Cleaning up temporary directory..."
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Upload complete!"
echo "Datasets uploaded: $copied"
echo "=========================================="
echo ""
echo "Users can now load datasets with:"
echo '  load_dataset("BeeGass/Group-Theory-Collection", name="s5")'
echo '  load_dataset("BeeGass/Group-Theory-Collection", name="a4")'
echo '  load_dataset("BeeGass/Group-Theory-Collection", name="c10")'