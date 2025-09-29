# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

SequenceWithMask = Tuple[List[int], List[bool]]


def get_content_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().rstrip()
        try:
            x = json.loads(line)
        except UnicodeDecodeError as e:
            print(f"Error when trying to decode '{line}': {str(e)}")
            raise
        for k in ["text", "content", "raw_content"]:
            if k in x:
                return k
        raise RuntimeError(f"Unable to determine key for {path}")


class JSONLIterator:
    def __init__(
        self,
        fpath: str,
        world_size: int,
        world_rank: int,
        infinite: bool,
    ) -> None:
        assert 0 <= world_rank < world_size, (world_rank, world_size)
        self.f = open(fpath, "r", encoding="utf-8", errors="ignore")
        self.fpath = fpath
        self.world_size = world_size
        self.world_rank = world_rank
        self.line_num = 0
        self.iter = iter(self.gen(infinite))
        self.iter_id = 0

    def __iter__(self) -> "JSONLIterator":
        return self

    def __next__(self):
        return next(self.iter)

    def gen(self, infinite: bool) -> Iterator[Dict]:
        while True:
            self.iter_id += 1
            while True:
                try:
                    line, self.line_num = self.f.readline(), self.line_num + 1
                    if not line:
                        break
                    if (self.line_num - 1) % self.world_size == self.world_rank:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            print("Failed to parse JSON:", e)
                        except Exception as e:
                            print(f"Unexpected Jsonl error: {e}")
                        continue  # Skip to the next iteration
                except Exception as e:
                    print(f"Unexpected error while reading line: {e}")
                continue
            if not infinite:
                break
            self.set_position(None, percentage=None)
        self.f.close()

    def set_position(
        self, position: Optional[int], percentage: Optional[float]
    ) -> None:
        if percentage is not None:
            position = int(np.floor(os.path.getsize(self.fpath) * percentage))
        if position is None:
            self.f.seek(0)
            self.line_num = 0
        else:
            assert type(position) is int
            self.f.seek(position)
            self.f.readline()
            self.line_num = 0  # change to 0 after adding custom resuming
            #     self.world_rank + 1
            # restore value of line_num (modulo world_size)

    def get_position(self) -> Optional[int]:
        file_pos = self.f.tell()
        if (
            file_pos == 0 or self.line_num == 0
        ):  # change "and" clause to "or" to after adding custom resuming
            return None
        assert (self.line_num - 1) % self.world_size == self.world_rank
        return file_pos


def sequence_iterator(
    jsonl_iterator: JSONLIterator,
    tokenizer,
    slen: int,
    buffer_size: int,
    rng: np.random.RandomState,
) -> Iterator[SequenceWithMask]:
    """
    Take as input a JSONLIterator and return an iterator of sequences.
    """
    content_key = get_content_key(jsonl_iterator.fpath)
    n_buffer_toks = buffer_size * slen

    tokens: List[int] = []
    mask: List[bool] = []

    for sample in jsonl_iterator:
        assert len(tokens) < n_buffer_toks

        _tokens = tokenizer(sample[content_key])["input_ids"]
        _mask = [True] * len(_tokens)

        assert len(_tokens) == len(_mask)
        tokens.extend(_tokens)
        mask.extend(_mask)

        while len(tokens) >= n_buffer_toks:
            x_tokens = np.array(tokens[:n_buffer_toks]).reshape(buffer_size, slen)
            x_mask = np.array(mask[:n_buffer_toks]).reshape(buffer_size, slen)

            tokens = tokens[n_buffer_toks:]
            mask = mask[n_buffer_toks:]

            seq_tokens: List[List[int]] = x_tokens.tolist()
            seq_mask: List[List[bool]] = x_mask.tolist()
            assert len(seq_tokens) == len(seq_mask) == buffer_size

            for idx in rng.permutation(len(seq_tokens)):
                assert len(seq_tokens[idx]) == slen
                yield seq_tokens[idx], seq_mask[idx]
