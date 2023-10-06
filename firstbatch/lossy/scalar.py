from __future__ import annotations
from typing import List
from firstbatch.lossy.base import BaseLossy, CompressedVector
from firstbatch.vector_store.schema import Vector


class ScalarQuantizer(BaseLossy):
    """Scalar quantizer algorithm."""
    try:
        from tdigest import TDigest
    except ImportError:
        raise ImportError("Please install tdigest to use ScalarQuantizer")

    def __init__(self, levels=256):
        self.quantizer = self.TDigest()
        self.quantiles: List[float] = []
        self.levels: int = levels

    def train(self, data: List[Vector]) -> None:
        scalars = Vector(vector=[], dim=0, id="")
        for vector in data:
            scalars = scalars.concat(vector)
        for scalar in scalars.vector:
            self.quantizer.update(scalar)
        self.quantiles = [self.quantizer.percentile(i * 100.0 / self.levels) for i in range(self.levels)]

    def __dequantize(self, qv):
        return [self.quantiles[val] for val in qv]

    def __quantize(self, v):
        return [self.__quantize_scalar(val) for val in v]

    def __quantize_scalar(self, scalar):
        return next((i for i, q in enumerate(self.quantiles) if scalar < q), self.levels - 1)

    def compress(self, data: Vector) -> CompressedVector:
        return CompressedVector(self.__quantize(data.vector), None, data.id)

    def decompress(self, data: CompressedVector) -> Vector:
        return Vector(self.__dequantize(data.vector), len(data.vector), data.id)
