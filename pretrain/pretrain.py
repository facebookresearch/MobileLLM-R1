# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import os
from logging import Logger
from typing import List, Optional

import torch
import transformers
from torch import distributed as dist
from transformers import AutoConfig, default_data_collator
from utils.multi_jsonl import MultiJSONLIterator

from utils.pretrain_trainer import PretrainTrainer
from utils.process_args import process_args


# Define a utility method for setting the logging parameters of a logger
def get_logger(logger_name: Optional[str]) -> logging.Logger:
    # Get the logger with the specified name
    logger = logging.getLogger(logger_name)

    # Set the logging level of the logger to INFO
    logger.setLevel(logging.INFO)

    # Define a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler for outputting log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


log: Logger = get_logger("mobileLLM")


def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK"):
        return int(os.environ["LOCAL_RANK"])
    else:
        logging.warning(
            "LOCAL_RANK from os.environ is None, fall back to get rank from torch distributed"
        )
        return torch.distributed.get_rank()


def get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0


def get_folder_paths(directory: str) -> List[str]:
    folder_paths = [
        os.path.join(directory, item)
        for item in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, item))
    ]
    return folder_paths


def train() -> None:
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", timeout=datetime.timedelta(hours=8)
    )
    model_args, data_args, training_args = process_args()

    global_rank = get_global_rank()
    local_rank = get_local_rank()

    log.info(f"Global Rank: {global_rank}")
    log.info(f"Local Rank: {local_rank}")
    config = AutoConfig.from_pretrained(model_args.input_model_filename)
    model = transformers.AutoModelForCausalLM.from_config(
        config=config,
    )
    log.info(
        "model size is "
        + str(sum(param.numel() for param in model.model.parameters()) / 1024 / 1024)
    )
    log.info("Start to load tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    log.info("Complete tokenizer loading...")

    # go to current node's data rank
    local_data_folder = os.path.join(
        data_args.train_data_local_path, str(global_rank // 8 + 1)
    )

    # Data load locally from shard total data, so world_size is 8 and rank is the current node's local rank
    log.info("world_rank for data loader is " + str(local_rank))
    log.info("world_size for data loader is " + str(8))
    assert os.path.isdir(local_data_folder), local_data_folder
    folder_paths_string = ",".join(get_folder_paths(local_data_folder))
    train_data = MultiJSONLIterator(
        tokenizer=tokenizer,
        data=folder_paths_string,
        instruct_data="",
        seq_len=training_args.model_max_length,
        batch_size=training_args.per_device_train_batch_size,
        buffer_size=2048,
        world_rank=local_rank,
        world_size=8,
        multiprocess=True,
        max_precompute=500,
        ignore_extra_chunks=False,
    )
    trainer = PretrainTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=None,
        data_collator=default_data_collator,
    )
    torch.distributed.barrier(device_ids=[local_rank])

    if training_args.do_train:
        _ = trainer.train()
        trainer.save_state()

    torch.distributed.barrier(device_ids=[local_rank])


if __name__ == "__main__":
    train()
