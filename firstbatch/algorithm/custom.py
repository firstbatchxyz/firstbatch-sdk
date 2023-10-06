from __future__ import annotations
from firstbatch.algorithm.base import BaseAlgorithm
from firstbatch.algorithm.registry import AlgorithmLabel
from firstbatch.algorithm.blueprint import DFAParser
from firstbatch.constants import DEFAULT_EMBEDDING_SIZE


class CustomAlgorithm(BaseAlgorithm, label=AlgorithmLabel.CUSTOM):
    is_custom: bool = True
    name: str = "CUSTOM"

    def __init__(self, bp, batch_size: int, **kwargs):
        parser = DFAParser(bp)
        blueprint = parser.parse()
        self._blueprint = blueprint
        self.embedding_size = DEFAULT_EMBEDDING_SIZE
        self.batch_size = batch_size
        self._include_values = True
        if "embedding_size" in kwargs:
            self.embedding_size = kwargs["embedding_size"]
        if "include_values" in kwargs:
            self._include_values = kwargs["include_values"]
