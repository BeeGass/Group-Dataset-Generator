#!/bin/bash
# Upload all clean datasets to HuggingFace, overwriting existing files

echo "Uploading clean datasets to HuggingFace..."
echo "This will overwrite existing files and remove cache files"
echo ""

# Upload each dataset
for dataset in alternating cyclic dihedral elementary_abelian frobenius klein mathieu psl quaternion symmetric; do
    echo "========================================"
    echo "Uploading ${dataset}_superset..."
    echo "========================================"
    
    uv run huggingface-cli upload \
        BeeGass/permutation-groups \
        /tmp/clean_datasets_upload/${dataset}_superset \
        data/${dataset}_superset \
        --repo-type dataset
    
    echo "✓ ${dataset}_superset uploaded"
    echo ""
done

echo "All datasets uploaded successfully!"