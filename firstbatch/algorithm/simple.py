from __future__ import annotations
from firstbatch.algorithm.base import BaseAlgorithm
from firstbatch.algorithm.registry import AlgorithmLabel
from firstbatch.algorithm.blueprint import DFAParser
from firstbatch.algorithm.blueprint.library import User_Centric_Promoted_Content_Curations as Simple
from firstbatch.constants import DEFAULT_BATCH_SIZE, DEFAULT_EMBEDDING_SIZE


class SimpleAlgorithm(BaseAlgorithm, label=AlgorithmLabel.SIMPLE):
    batch_size: int = DEFAULT_BATCH_SIZE
    is_custom: bool = False
    name: str = "SIMPLE"

    def __init__(self, batch_size: int, **kwargs):
        parser = DFAParser(Simple)
        blueprint = parser.parse()
        self._blueprint = blueprint
        self.embedding_size = DEFAULT_EMBEDDING_SIZE
        self.batch_size = batch_size
        self._include_values = True
        if "embedding_size" in kwargs:
            self.embedding_size = kwargs["embedding_size"]
        if "include_values" in kwargs:
            self._include_values = kwargs["include_values"]
