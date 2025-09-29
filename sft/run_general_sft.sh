# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

torchrun --nproc_per_node=2 --nnodes=1 sft.py \
    --model_name_or_path base_model_with_chat_template \
    --dataset_name allenai/tulu-3-sft-olmo-2-mixture-0225 \
    --max_length 4096 \
    --learning_rate 5.0e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --eos_token '<|eot_id|>' \
    --eval_strategy no \
    --logging_steps 1 \
    --saving_steps 1000 \
    --output_dir /tmp/MobileLLM-R1-950M \
    --use_liger_kernel True
