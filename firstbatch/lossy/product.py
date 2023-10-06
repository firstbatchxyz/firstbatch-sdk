from __future__ import annotations
from typing import List, TYPE_CHECKING

from firstbatch.lossy.base import BaseLossy, CompressedVector
from firstbatch.vector_store.schema import Vector

if TYPE_CHECKING:
    import nanopq


class ProductQuantizer(BaseLossy):
    """Product quantizer algorithm."""

    def __init__(self, cluster_size: int = 256, subquantizer_size: int = 32, verbose=False):
        try:
            import numpy as np
            import nanopq
        except ImportError:
            raise ImportError("Please install numpy && nanopq to use ProductQuantizer")

        self.data = None
        self.m = subquantizer_size
        self.ks = cluster_size
        self.__trained = False
        self.quantizer = nanopq.PQ(M=subquantizer_size, Ks=cluster_size, verbose=verbose)  # 1024
        self.quantizer_residual = nanopq.PQ(M=subquantizer_size, Ks=cluster_size, verbose=verbose)

    def train(self, data: List[Vector]) -> None:
        # Encode data to PQ-codes
        if data[0].dim % self.m != 0:
            raise ValueError("input dimension must be dividable by M")

        if self.__trained:
            return None

        train_x = np.array([v.vector for v in data], dtype=np.float32)
        self.quantizer.fit(train_x)
        x_code = self.quantizer.encode(train_x)
        x = self.quantizer.decode(x_code)

        residuals = train_x - x
        self.quantizer_residual.fit(residuals)

        self.__trained = True

    # can only be used if train() has been called
    def compress(self, data: Vector) -> CompressedVector:
        if not self.__trained:
            raise ValueError("train() must be called before compress()")
        x = self.quantizer.encode(np.array(data.vector, dtype=np.float32).reshape(1, -1))
        residual = self.quantizer_residual.encode(np.array(data.vector, dtype=np.float32) - self.quantizer.decode(x))
        return CompressedVector(x.tolist()[0], residual.tolist()[0], data.id)

    def decompress(self, data: CompressedVector) -> Vector:
        if not self.__trained:
            raise ValueError("train() must be called before compress()")
        x = self.quantizer.decode(np.array(data.vector, dtype=np.uint16).reshape(1, -1))
        residual = self.quantizer_residual.decode(np.array(data.residual, dtype=np.uint16).reshape(1, -1))
        return Vector((x + residual).tolist(), len(data.vector), data.id)

