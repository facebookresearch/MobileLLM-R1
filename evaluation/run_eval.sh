# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

DATA_NAME="math,gsm8k,aime24"
SPLIT="test"
PROMPT_TYPE="raw"
MODEL_NAME_OR_PATH="facebook/MobileLLM-R1-950M"
DATA_DIR="./data"
OUTPUT_DIR="/tmp/output/$MODEL_NAME_OR_PATH"

CUDA_VISIBLE_DEVICES=0 python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name "${DATA_NAME}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --prompt_type ${PROMPT_TYPE} \
    --max_tokens_per_call 30000 \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --use_vllm \
    --save_outputs \
    --apply_chat_template
