#!/bin/bash
# =============================================================================
# Korean VQA Experiment Pipeline
# =============================================================================
# This script runs the full experiment:
# 1. Data preparation
# 2. Fine-tuning Qwen2-VL on Korean data
# 3. CMAAM merging with original model
# 4. Baseline comparisons
# =============================================================================

set -e

# Configuration
PROJECT_ROOT="/NetDisk/juyeon/AdaMMS"
BASE_MODEL="${PROJECT_ROOT}/Qwen2-VL-7B-Instruct"
DATA_DIR="${PROJECT_ROOT}/data/korean_vqa"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/korean_experiment"

# Experiment mode: tiny, quick, or full
MODE=${1:-"quick"}

echo "=============================================================="
echo "Korean VQA Experiment Pipeline"
echo "=============================================================="
echo "Mode: ${MODE}"
echo "Project root: ${PROJECT_ROOT}"
echo "Base model: ${BASE_MODEL}"
echo ""

# =============================================================================
# Step 1: Data Preparation
# =============================================================================
echo ""
echo "=============================================================="
echo "Step 1: Data Preparation"
echo "=============================================================="

if [ "$MODE" == "tiny" ]; then
    python ${PROJECT_ROOT}/scripts/prepare_korean_vqa_data.py \
        --output ${DATA_DIR}_tiny \
        --tiny
    DATA_DIR="${DATA_DIR}_tiny"
elif [ "$MODE" == "quick" ]; then
    python ${PROJECT_ROOT}/scripts/prepare_korean_vqa_data.py \
        --output ${DATA_DIR}_quick \
        --quick
    DATA_DIR="${DATA_DIR}_quick"
else
    python ${PROJECT_ROOT}/scripts/prepare_korean_vqa_data.py \
        --output ${DATA_DIR}
fi

echo "Data prepared at: ${DATA_DIR}"

# =============================================================================
# Step 2: Fine-tuning
# =============================================================================
echo ""
echo "=============================================================="
echo "Step 2: Fine-tuning Qwen2-VL on Korean VQA"
echo "=============================================================="

FINETUNE_OUTPUT="${OUTPUT_DIR}/finetuned_${MODE}"

if [ "$MODE" == "tiny" ]; then
    # Tiny mode: 1 epoch, small batch
    python ${PROJECT_ROOT}/scripts/finetune_qwen2vl_korean.py \
        --model ${BASE_MODEL} \
        --data ${DATA_DIR} \
        --output ${FINETUNE_OUTPUT} \
        --epochs 1 \
        --batch-size 1 \
        --grad-accum 4 \
        --save-steps 100 \
        --eval-steps 100 \
        --use-4bit
elif [ "$MODE" == "quick" ]; then
    # Quick mode: 2 epochs
    python ${PROJECT_ROOT}/scripts/finetune_qwen2vl_korean.py \
        --model ${BASE_MODEL} \
        --data ${DATA_DIR} \
        --output ${FINETUNE_OUTPUT} \
        --epochs 2 \
        --batch-size 1 \
        --grad-accum 8 \
        --save-steps 500 \
        --eval-steps 500 \
        --use-4bit
else
    # Full mode: 3 epochs
    python ${PROJECT_ROOT}/scripts/finetune_qwen2vl_korean.py \
        --model ${BASE_MODEL} \
        --data ${DATA_DIR} \
        --output ${FINETUNE_OUTPUT} \
        --epochs 3 \
        --batch-size 2 \
        --grad-accum 8 \
        --save-steps 1000 \
        --eval-steps 1000
fi

KOREAN_EXPERT="${FINETUNE_OUTPUT}/merged"
echo "Fine-tuned model saved at: ${KOREAN_EXPERT}"

# =============================================================================
# Step 3: CMAAM Merging
# =============================================================================
echo ""
echo "=============================================================="
echo "Step 3: CMAAM Merging"
echo "=============================================================="

CMAAM_OUTPUT="${OUTPUT_DIR}/cmaam_merged_${MODE}"

python ${PROJECT_ROOT}/merge/cmaam/cmaam_merge.py \
    --source ${KOREAN_EXPERT} \
    --target ${BASE_MODEL} \
    --output ${CMAAM_OUTPUT} \
    --model-type qwen2vl \
    --strategy full \
    --base-alpha 0.3 \
    --analyze \
    --save-alphas

echo "CMAAM merged model saved at: ${CMAAM_OUTPUT}"

# =============================================================================
# Step 4: Baseline Comparisons
# =============================================================================
echo ""
echo "=============================================================="
echo "Step 4: Baseline Merging Methods"
echo "=============================================================="

# Linear merge with different alphas
for ALPHA in 0.3 0.5 0.7; do
    LINEAR_OUTPUT="${OUTPUT_DIR}/linear_alpha${ALPHA}_${MODE}"
    echo "Creating linear merge with alpha=${ALPHA}..."

    python -c "
import torch
import safetensors.torch
import os
import json

source_path = '${KOREAN_EXPERT}'
target_path = '${BASE_MODEL}'
output_path = '${LINEAR_OUTPUT}'
alpha = ${ALPHA}

print(f'Loading source model...')
# Load source weights
source_files = sorted([f for f in os.listdir(source_path) if f.endswith('.safetensors')])
source_weights = {}
for f in source_files:
    source_weights.update(safetensors.torch.load_file(os.path.join(source_path, f)))

print(f'Loading target model...')
# Load target weights
target_files = sorted([f for f in os.listdir(target_path) if f.endswith('.safetensors')])
target_weights = {}
for f in target_files:
    target_weights.update(safetensors.torch.load_file(os.path.join(target_path, f)))

print(f'Merging with alpha={alpha}...')
# Simple linear merge
merged = {}
for key in target_weights:
    if key in source_weights and source_weights[key].shape == target_weights[key].shape:
        merged[key] = (1 - alpha) * target_weights[key] + alpha * source_weights[key]
    else:
        merged[key] = target_weights[key]

# Add source-only keys
for key in source_weights:
    if key not in merged:
        merged[key] = source_weights[key]

print(f'Saving to {output_path}...')
os.makedirs(output_path, exist_ok=True)

# Load index from target
with open(os.path.join(target_path, 'model.safetensors.index.json'), 'r') as f:
    index = json.load(f)

# Save in same structure as target
weight_map = index['weight_map']
file_weights = {}
for key, filename in weight_map.items():
    if key in merged:
        if filename not in file_weights:
            file_weights[filename] = {}
        file_weights[filename][key] = merged[key]

for filename, weights in file_weights.items():
    safetensors.torch.save_file(weights, os.path.join(output_path, filename))

# Copy index
import shutil
shutil.copy(os.path.join(target_path, 'model.safetensors.index.json'), output_path)

# Symlink configs
for cfg in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json']:
    src = os.path.join(target_path, cfg)
    dst = os.path.join(output_path, cfg)
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(src, dst)

print('Done!')
"
done

echo ""
echo "=============================================================="
echo "Experiment Complete!"
echo "=============================================================="
echo ""
echo "Models created:"
echo "  1. Fine-tuned (Korean Expert): ${KOREAN_EXPERT}"
echo "  2. CMAAM Merged: ${CMAAM_OUTPUT}"
echo "  3. Linear (alpha=0.3): ${OUTPUT_DIR}/linear_alpha0.3_${MODE}"
echo "  4. Linear (alpha=0.5): ${OUTPUT_DIR}/linear_alpha0.5_${MODE}"
echo "  5. Linear (alpha=0.7): ${OUTPUT_DIR}/linear_alpha0.7_${MODE}"
echo ""
echo "Next step: Run evaluation on Korean VQA and English benchmarks"
echo "=============================================================="
