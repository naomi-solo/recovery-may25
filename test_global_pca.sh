#!/usr/bin/env bash

set -e

echo "===== GLOBAL PCA SANITY TEST ====="

N_TOTAL=15
SEED=0
LAYER=14
EPS=8.0
CC=ambig
TAG=flip
MODE=chosen
SPLIT=train

DATASETS=("bbq" "math_mc")

echo "Datasets: ${DATASETS[*]}"
echo "N_TOTAL: $N_TOTAL"

echo ""
echo "=== STEP 1: Extract directions ==="

for DATASET in "${DATASETS[@]}"; do
  echo "Running extract for $DATASET"
  PYTHONPATH=. \
  DATASET=$DATASET \
  N_TOTAL=$N_TOTAL \
  SEED=$SEED \
  LAYER=$LAYER \
  EPS=$EPS \
  CC=$CC \
  TAG=$TAG \
  python src/core/extract_directions.py
done

echo ""
echo "=== STEP 2: Verify dataset field ==="

PYTHONPATH=. python - <<EOF
import torch

path = "results/outputs/directions/direction_records_bbq_train_seed0_layer14_eps8.0_ambig_flip_n15.pt"
obj = torch.load(path, map_location="cpu")
rec = obj["records"][0]

print("Dataset field:", rec.get("dataset"))
assert "dataset" in rec, "Missing dataset field!"
EOF

echo ""
echo "=== STEP 3: Run GLOBAL PCA ==="

PYTHONPATH=. \
GLOBAL_PCA=1 \
DATASETS=$(IFS=,; echo "${DATASETS[*]}") \
MODE=$MODE \
N=$N_TOTAL \
SEED=$SEED \
LAYER=$LAYER \
EPS=$EPS \
CC=$CC \
TAG=$TAG \
python src/core/pca_directions.py

echo ""
echo "=== STEP 4: Label components ==="

PYTHONPATH=. \
GLOBAL_PCA=1 \
DATASETS=$(IFS=,; echo "${DATASETS[*]}") \
MODE=$MODE \
N=$N_TOTAL \
SEED=$SEED \
LAYER=$LAYER \
EPS=$EPS \
CC=$CC \
TAG=$TAG \
python src/core/label_components.py

echo ""
echo "===== DONE ====="
echo "Check above output for:"
echo "- dataset_counts (should show bbq + math_mc)"
echo "- DATASET: lines in label prompt"
