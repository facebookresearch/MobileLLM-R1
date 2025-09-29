# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterator, List, Optional, TypeVar

import numpy as np
import numpy.typing as npt


@dataclass
class Batch:
    x: npt.NDArray
    y: npt.NDArray
    mask: Optional[npt.NDArray] = None
    logits: Optional[npt.NDArray] = None
    src_names: Optional[List[str]] = None
    weight: Optional[npt.NDArray] = None

    def __post_init__(self):
        assert self.x.ndim == 2
        assert self.x.shape == self.y.shape
        assert self.x.dtype == np.int64
        assert self.y.dtype in [np.int64, np.float32]
        assert self.mask is None or self.mask.shape == self.x.shape
        assert self.src_names is None or len(self.src_names) == len(self.x)
        weight = self.weight
        assert weight is None or (
            weight.dtype == np.float32
        )  # and (weights.shape == self.y.shape)

    def concat(self, other: "Batch", dim: int = 0) -> "Batch":
        assert (self.mask is None) == (other.mask is None)
        assert (self.logits is None) == (other.logits is None)
        assert (self.weight is None) == (other.weight is None)
        return Batch(
            x=np.concatenate([self.x, other.x], axis=dim),
            y=np.concatenate([self.y, other.y], axis=dim),
            mask=(
                None
                if self.mask is None
                else np.concatenate([self.mask, other.mask], axis=dim)
            ),
            logits=(
                None
                if self.logits is None
                else np.concatenate([self.logits, other.logits], axis=dim)
            ),
            weight=(
                None
                if self.weight is None
                else np.concatenate([self.weight, other.weight], axis=dim)
            ),
            src_names=(
                (self.src_names or ([""] * self.x.shape[0]))
                + (other.src_names or ([""] * other.x.shape[0]))
            ),
        )


T = TypeVar("T")


# Note: technically this is an iterable not an iterator, since it doesn't implement __next__.
# To get the iterator you can call __iter__
class GenericDataIterator(Generic[T]):
    @abstractmethod
    def __iter__(self) -> Iterator[T]: ...

    @abstractmethod
    def get_position(self) -> Optional[List[int]]: ...

    @abstractmethod
    def set_position(self, position: Optional[List[int]]) -> None: ...

    def close(self) -> None:
        pass


DataIterator = GenericDataIterator[Batch]
