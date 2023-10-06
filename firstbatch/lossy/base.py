from __future__ import annotations
from typing import List, Optional
from abc import ABC, abstractmethod
from dataclasses_json import DataClassJsonMixin
from dataclasses import dataclass
from firstbatch.vector_store.schema import Vector


@dataclass
class CompressedVector(DataClassJsonMixin):
    vector: List[int]
    residual: Optional[List[int]]
    id: str


class BaseLossy(ABC):
    """Base class for lossy compression algorithms."""

    @abstractmethod
    def train(self, data: List[Vector]) -> None:
        """Train the algorithm.

        Args:
            data (Vector): Data to train the algorithm.
        """
        ...

    @abstractmethod
    def compress(self, data: Vector) -> CompressedVector:
        """Compress data.

        Args:
            data (Any): Data to be compressed.

        Returns:
            Any: Compressed data.
        """
        ...

    @abstractmethod
    def decompress(self, data: CompressedVector) -> Vector:
        """Decompress data.

        Args:
            data (Any): Data to be decompressed.

        Returns:
            Any: Decompressed data.
        """
        ...