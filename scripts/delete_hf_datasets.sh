#!/bin/bash
# Delete datasets from HuggingFace repository (REMOTE ONLY - does not touch local files)
# Usage: ./delete_hf_datasets.sh [pattern1] [pattern2] ...
# 
# Pattern examples (deletes from data/ directory):
#   ./delete_hf_datasets.sh s3 s4 s5 s6 s7  # specific datasets
#   ./delete_hf_datasets.sh s* a* c*  # wildcard patterns
#   ./delete_hf_datasets.sh s[3-7] a[3-7]  # range patterns
#   ./delete_hf_datasets.sh  # no args = delete all from data/
#
# Special keywords:
#   ./delete_hf_datasets.sh data  # deletes entire data/ directory
#   ./delete_hf_datasets.sh tc0  # deletes entire TC0/ directory  
#   ./delete_hf_datasets.sh nc1  # deletes entire NC1/ directory
#   ./delete_hf_datasets.sh all  # deletes EVERYTHING (TC0/, NC1/, and data/)

echo "Deleting datasets from HuggingFace repository (remote only)..."
echo "NOTE: This does NOT delete any local files"

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

# Process special keywords and patterns
processed_patterns=()
special_keyword=""
for arg in "$@"; do
    case "$arg" in
        "all"|"data"|"tc0"|"nc1")
            special_keyword="$arg"
            processed_patterns+=("$arg")
            ;;
        *)
            processed_patterns+=("$arg")
            ;;
    esac
done

# If no arguments provided, delete all
if [ $# -eq 0 ]; then
    echo "No patterns specified, will delete ALL datasets from HuggingFace"
    patterns=("*")
else
    echo "Delete patterns: $@"
    patterns=("${processed_patterns[@]}")
fi

echo ""

# Get list of all possible datasets from local directory
# (only used to match patterns - no local files are modified)
all_datasets=()
for dataset_dir in datasets/*_data; do
    if [ -d "$dataset_dir" ]; then
        dataset_name=$(basename "$dataset_dir")
        clean_name=${dataset_name%_data}
        all_datasets+=("$clean_name")
    fi
done

# Filter datasets based on patterns (skip if we're using special keywords)
datasets_to_delete=()
if [[ "$special_keyword" != "all" && "$special_keyword" != "tc0" && "$special_keyword" != "nc1" ]]; then
    for dataset in "${all_datasets[@]}"; do
        match_found=false
        for pattern in "${patterns[@]}"; do
            # Skip special keywords in pattern matching
            if [[ "$pattern" == "all" || "$pattern" == "tc0" || "$pattern" == "nc1" || "$pattern" == "data" ]]; then
                continue
            fi
            if [[ "$dataset" == $pattern ]]; then
                match_found=true
                break
            fi
        done
        
        if [ "$match_found" = true ]; then
            datasets_to_delete+=("$dataset")
        fi
    done
fi

# Show what will be deleted from HuggingFace
if [[ " $@ " =~ " all " ]] && [ ${#@} -eq 1 ]; then
    echo "Will delete from HuggingFace:"
    echo "  - TC0/ (entire directory)"
    echo "  - NC1/ (entire directory)"
    echo "  - data/ (entire directory)"
    echo ""
elif [[ " $@ " =~ " tc0 " ]] && [ ${#@} -eq 1 ]; then
    echo "Will delete from HuggingFace:"
    echo "  - TC0/ (entire directory)"
    echo ""
elif [[ " $@ " =~ " nc1 " ]] && [ ${#@} -eq 1 ]; then
    echo "Will delete from HuggingFace:"
    echo "  - NC1/ (entire directory)"
    echo ""
elif [[ " $@ " =~ " data " ]] && [ ${#@} -eq 1 ]; then
    echo "Will delete from HuggingFace:"
    echo "  - data/ (entire directory)"
    echo ""
else
    echo "Datasets to delete from HuggingFace:"
    for dataset in "${datasets_to_delete[@]}"; do
        echo "  - data/$dataset/"
    done
    echo ""
    echo "Total: ${#datasets_to_delete[@]} datasets"
    echo ""
fi

# Confirm
read -p "Delete these datasets from HuggingFace? (yes/no): " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "========================================"
echo "Deleting from HuggingFace repository..."
echo "========================================"
echo ""

# Create empty temp directory for upload (we're only deleting)
temp_dir=$(mktemp -d)
touch "$temp_dir/.gitkeep"

# Execute the deletion on HuggingFace
if [[ " $@ " =~ " all " ]] && [ ${#@} -eq 1 ]; then
    uv run huggingface-cli upload \
        BeeGass/Group-Theory-Collection \
        "$temp_dir" \
        . \
        --repo-type dataset \
        --commit-message "Delete all directories" \
        --delete "TC0/**" \
        --delete "NC1/**" \
        --delete "data/**"
elif [[ " $@ " =~ " tc0 " ]] && [ ${#@} -eq 1 ]; then
    uv run huggingface-cli upload \
        BeeGass/Group-Theory-Collection \
        "$temp_dir" \
        . \
        --repo-type dataset \
        --commit-message "Delete TC0 directory" \
        --delete "TC0/**"
elif [[ " $@ " =~ " nc1 " ]] && [ ${#@} -eq 1 ]; then
    uv run huggingface-cli upload \
        BeeGass/Group-Theory-Collection \
        "$temp_dir" \
        . \
        --repo-type dataset \
        --commit-message "Delete NC1 directory" \
        --delete "NC1/**"
elif [[ " $@ " =~ " data " ]] && [ ${#@} -eq 1 ]; then
    uv run huggingface-cli upload \
        BeeGass/Group-Theory-Collection \
        "$temp_dir" \
        . \
        --repo-type dataset \
        --commit-message "Delete data directory" \
        --delete "data/**"
else
    # Build the command for individual datasets
    cmd="uv run huggingface-cli upload BeeGass/Group-Theory-Collection \"$temp_dir\" . --repo-type dataset --commit-message \"Delete datasets: ${patterns[*]}\""
    for dataset in "${datasets_to_delete[@]}"; do
        cmd="$cmd --delete \"data/$dataset/**\""
    done
    eval "$cmd"
fi

# Check exit status
exit_status=$?

# Clean up temp directory
rm -rf "$temp_dir"

if [ $exit_status -eq 0 ]; then
    echo ""
    if [[ " $@ " =~ " all " ]] && [ ${#@} -eq 1 ]; then
        echo "✓ Successfully deleted TC0/, NC1/, and data/ directories from HuggingFace"
    elif [[ " $@ " =~ " tc0 " ]] && [ ${#@} -eq 1 ]; then
        echo "✓ Successfully deleted TC0/ directory from HuggingFace"
    elif [[ " $@ " =~ " nc1 " ]] && [ ${#@} -eq 1 ]; then
        echo "✓ Successfully deleted NC1/ directory from HuggingFace"
    elif [[ " $@ " =~ " data " ]] && [ ${#@} -eq 1 ]; then
        echo "✓ Successfully deleted data/ directory from HuggingFace"
    else
        echo "✓ Successfully deleted ${#datasets_to_delete[@]} datasets from HuggingFace"
    fi
    echo "✓ Local files remain untouched"
else
    echo ""
    echo "✗ Error occurred during HuggingFace deletion"
    exit 1
fi

echo ""
echo "========================================"
echo "Done!"
echo "========================================"