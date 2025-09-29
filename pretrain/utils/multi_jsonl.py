# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import re
from dataclasses import dataclass
from logging import getLogger, Logger
from pathlib import Path
from queue import Empty, Full
from typing import Iterator, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
import torch

from .base import Batch, DataIterator

from .jsonl import JSONLIterator, sequence_iterator, SequenceWithMask


logger: Logger = getLogger()


Position = List[Optional[int]]


def _combine_seq_iterators(
    seq_iterators: List[Iterator[SequenceWithMask]],
    src_names: List[str],
    weights: npt.NDArray,
    seq_len: int,
    batch_size: int,
    rng: np.random.RandomState,
) -> Iterator[Batch]:
    assert len(seq_iterators) == len(src_names) == len(weights)
    while True:
        tokens: List[List[int]] = []
        masks: List[List[bool]] = []
        srcs: List[str] = []
        for _ in range(batch_size):
            src_id = rng.choice(len(weights), p=weights)
            _tokens, _mask = next(seq_iterators[src_id])
            assert len(_tokens) == len(_mask) == seq_len + 1
            tokens.append(_tokens)
            masks.append(_mask)
            srcs.append(src_names[src_id])
        x_tokens = np.array(tokens)
        assert x_tokens.shape == (batch_size, seq_len + 1)

        # With the following shifting, we don't need extra shifting in the loss
        # calculation.
        yield dict(
            input_ids=torch.tensor(x_tokens[:, :-1]),
            labels=torch.tensor(x_tokens[:, :-1]),
        )


@dataclass
class DataAssignment:
    path: str  # .jsonl path
    rank: int  # rank among workers on this file
    size: int  # number of workers on this file
    weight: float  # weight

    @property
    def name(self) -> str:
        return Path(self.path).parent.name

    def __post_init__(self) -> None:
        assert self.path.endswith(".jsonl"), self.path
        assert os.path.isfile(self.path), self.path
        assert 0 <= self.rank < self.size
        assert self.weight > 0


def _assign_data(
    path: str, world_size: int, ignore_extra: bool
) -> List[Tuple[str, int, int]]:
    """
    Given a directory, list .jsonl files, and assign one to a worker.
    """
    assert os.path.isdir(path), path
    fname_match_re: str = r"\.jsonl$"
    fnames = [x for x in os.listdir(path) if re.search(fname_match_re, x)]
    assert (
        len(fnames) > 0
    ), "found no files matching '{fname_match_re}' in path '{path}'"

    if ignore_extra and len(fnames) > world_size:
        logger.warning(f"Removing {len(fnames) - world_size} extra chunks for {path}")
        fnames = fnames[:world_size]
    fpaths = [os.path.join(path, fname) for fname in sorted(fnames)]
    assert world_size % len(fpaths) == 0, (world_size, len(fpaths), path)
    n = world_size // len(fpaths)  # number of workers on the same file
    res = []
    for path in fpaths:
        for i in range(n):
            res.append((path, i, n))
    assert len(res) == world_size
    return res


def _get_data_assignment(
    data: str,
    world_rank: int,
    world_size: int,
    ignore_extra: bool,
    data_weight: Optional[str] = None,
) -> List[DataAssignment]:
    """
    `data` can be either of the form:
        - wiki
    or
        - wiki:2,ccnet:10,github:5
    In the second case, data weights can be arbitrary float values.
    Each data directory must contain a number of files that divides `world_size`.
    """
    assert len(data) > 0
    assert 0 <= world_rank < world_size

    # same folder for all workers
    if "," not in data:
        fpath, rank, size = _assign_data(data, world_size, ignore_extra)[world_rank]
        return [DataAssignment(fpath, rank, size, 1.0)]
    # Parse data_weight if provided
    data_weight_dict = None
    if data_weight:
        data_weight_dict = {}
        for item in data_weight.split(","):
            name, weight = item.strip().split(":")
            data_weight_dict[name.strip()] = weight.strip()

    # otherwise, one folder per dataset
    assignment: List[DataAssignment] = []
    seen: Set[str] = set()
    for x in data.split(","):
        name, weight = x.split(":")
        name = name.split("/")[-1].split(":")[0]
        if data_weight_dict and name in data_weight_dict:
            weight = data_weight_dict[name]
        path = x
        assert path not in seen
        assert re.fullmatch(r"\d+(\.\d*)?", weight) and float(weight) > 0
        seen.add(path)
        fpath, rank, size = _assign_data(path, world_size, ignore_extra)[world_rank]
        assignment.append(DataAssignment(fpath, rank, size, float(weight)))

    assert len(assignment) == len(data.split(","))
    return assignment


class MultiJSONLIterator(DataIterator):
    def __init__(
        self,
        tokenizer,
        data: str,
        instruct_data: str,
        seq_len: int,
        batch_size: int,
        buffer_size: int,
        world_rank: int,
        world_size: int,
        multiprocess: bool,
        max_precompute: int,
        ignore_extra_chunks: bool,
        instruct: bool = False,
        data_weight: Optional[str] = None,
    ) -> None:
        # tokenizer
        self.tokenizer = tokenizer

        # main parameters
        self.data = data
        self.instruct_data = instruct_data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.world_rank = world_rank
        self.world_size = world_size
        self.multiprocess = multiprocess
        self.max_precompute = max_precompute
        self.ignore_extra_chunks = ignore_extra_chunks
        assert data or instruct_data
        assert 0 <= world_rank < world_size
        # data assigned to worker  / verify paths
        self.data_assignment = _get_data_assignment(
            data=",".join([x for x in [data, instruct_data] if x]),
            world_rank=world_rank,
            world_size=world_size,
            ignore_extra=ignore_extra_chunks,
            data_weight=data_weight,
        )
        logger.info(
            f"Starting iteration on {str(self.data_assignment)} "
            f"({world_rank}/{world_size}) ..."
        )

        # .jsonl iterators
        self.jsonl_iterators = [
            JSONLIterator(
                fpath=x.path,
                world_rank=x.rank,
                world_size=x.size,
                infinite=True,
            )
            for x in self.data_assignment
        ]
        self.src_names = sorted([x.name for x in self.data_assignment])
        assert len(self.src_names) == len(set(self.src_names)), self.src_names

        # check what is pretraining and instruct data
        n_pretrain = len([x for x in data.split(",") if x])
        n_instruct = len([x for x in instruct_data.split(",") if x])
        is_instruct = [False] * n_pretrain + [True] * n_instruct
        assert len(self.jsonl_iterators) == n_pretrain + n_instruct

        # sequence iterators
        self.seq_iterators: List[Iterator[SequenceWithMask]] = [
            sequence_iterator(
                jsonl_iterator=jsonl_iterator,
                tokenizer=self.tokenizer,
                slen=self.seq_len + 1,  # +1 for input/output 1-shift
                buffer_size=self.buffer_size,
                rng=np.random.RandomState((world_rank, world_size)),
            )
            for _, jsonl_iterator in zip(is_instruct, self.jsonl_iterators)
        ]

        # data source weights
        self.weights = np.array(
            [x.weight for x in self.data_assignment], dtype=np.float64
        )
        self.weights = self.weights / self.weights.sum()
        assert (
            abs(self.weights.sum() - 1) < 1e-6 and min(self.weights) > 0
        ), self.weights
        logger.info(f"Data source weights: {self.weights}")

        # multiprocessing
        self.batch_queue: Optional[mp.Queue] = None
        self.mp_position: Optional[List[int]] = None
        self.process: Optional[mp.process.BaseProcess] = None
        self.stop: Optional[mp.synchronize.Event] = None

    def _init_multi_process(self) -> None:
        logger.info("Initializing multi process ...")
        assert self.multiprocess
        assert self.process is None
        ctx = mp.get_context("fork")
        self.stop = ctx.Event()
        self.batch_queue = ctx.Queue(maxsize=self.max_precompute)
        self.process = ctx.Process(
            name="iterator_multi",
            target=self._multiprocess_iterator,
        )
        assert self.process is not None
        self.process.start()

    def build_iterator(self) -> Iterator[Batch]:
        return _combine_seq_iterators(
            seq_iterators=self.seq_iterators,
            src_names=self.src_names,
            weights=self.weights,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            rng=np.random.RandomState((self.world_rank, self.world_size)),
        )

    def _multiprocess_iterator(self) -> None:
        iterator = self.build_iterator()
        assert self.batch_queue is not None and self.stop is not None
        try:
            batch: Optional[Batch] = None
            while not self.stop.is_set():
                batch = next(iterator) if batch is None else batch
                try:
                    self.batch_queue.put((batch, self._get_position()))
                    batch = None
                except Full:
                    pass
        finally:
            self.stop.set()

    def multiprocess_iterator_loop(self) -> Iterator[Batch]:
        assert self.batch_queue is not None and self.stop is not None
        try:
            while not self.stop.is_set():
                try:
                    batch, self.mp_position = self.batch_queue.get()
                    yield batch
                except Empty:
                    pass
        finally:
            self.stop.set()

    def __iter__(self) -> Iterator[Batch]:
        if not self.multiprocess:
            return self.build_iterator()
        else:
            self._init_multi_process()
            return self.multiprocess_iterator_loop()

    #  `GenericDataIterator` inconsistently.
    def set_position(
        self, position: Optional[List[int]], percentage: Optional[float]
    ) -> None:
        assert self.process is None  # if multiprocessing, position must be set before
        if percentage is not None:
            position = [percentage] * len(self.jsonl_iterators)
        if position is None:
            return
        logger.warning(
            f"Setting JSONL position on {self.data_assignment} "
            f"({self.world_rank}/{self.world_size}): {position}"
        )
        assert type(position) is list
        assert len(position) == len(self.jsonl_iterators)
        for x, pos in zip(self.jsonl_iterators, position):
            x.set_position(pos, percentage=percentage)  # `pos` can be int or None

    def _get_position(self) -> Optional[List[int]]:
        pos = [x.get_position() for x in self.jsonl_iterators]
        if all(p is None for p in pos):
            return None
        else:
            assert all(p is None or type(p) is int for p in pos), pos
            return pos  # type: ignore

    def get_position(self) -> Optional[List[int]]:
        return self.mp_position if self.multiprocess else self._get_position()

    def close(self) -> None:
        if self.process is not None:
            print(f"Attempting to close process nicely (I'm process {os.getpid()})")
            if self.stop is not None:
                self.stop.set()
            p = self.process
            p.join(timeout=5)
            if p.exitcode is None:
                print(f"Killing data process {p.pid} ...")
                p.kill()
            else:
                print(f"Data process {p.pid} exited with code {p.exitcode}")
