# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

torchrun --nproc_per_node=8 --nnodes=16 sft.py \
    --model_name_or_path facebook/MobileLLM-R1-950M-base \
    --dataset_name dataset_chatml_format \
    --max_length 32768 \
    --learning_rate 8.0e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --eos_token '<|eot_id|>' \
    --eval_strategy no \
    --logging_steps 1 \
    --saving_steps 1000 \
    --output_dir /tmp/MobileLLM-R1-950M \
    --use_liger_kernel True
