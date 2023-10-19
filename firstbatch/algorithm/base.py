"""BaseAlgorithm"""
from __future__ import annotations
from abc import ABC
from typing import Optional, Any, List
from firstbatch.algorithm.blueprint import Blueprint, UserAction
from firstbatch.vector_store.utils import maximal_marginal_relevance
from firstbatch.vector_store.schema import BatchQueryResult, BatchQuery, QueryMetadata
from firstbatch.algorithm.registry import AlgorithmLabel, AlgorithmRegistry


class BaseAlgorithm(ABC):

    batch_size: int
    is_custom: bool
    name: str

    def __init_subclass__(cls, label: Optional[AlgorithmLabel] = None, **kwargs):
        """Automatically registers subclasses with the AlgorithmRegistry."""
        super().__init_subclass__(**kwargs)
        if label:
            AlgorithmRegistry.register_algorithm(label, cls)

    @property
    def _blueprint(self) -> Blueprint:
        return self.__blueprint

    @_blueprint.setter
    def _blueprint(self, value):
        self.__blueprint = value

    def blueprint_step(self, state: str, action: UserAction) -> Any:
        """Call the step method of the _blueprint."""
        return self._blueprint.step(state, action)

    def random_batch(self, batch: BatchQueryResult, query: BatchQuery, **kwargs) -> Any:
        # We can't use apply threshold with random batches
        if "apply_threshold" in kwargs:
            del kwargs["apply_threshold"]
        if "apply_mmr" in kwargs:
            del kwargs["apply_mmr"]
        kwargs["shuffle"] = True
        ids, metadata = self._apply_params(batch, query, **kwargs)
        return ids[:batch.batch_size], metadata[:batch.batch_size]

    def biased_batch(self, batch: BatchQueryResult, query: BatchQuery, **kwargs) \
            -> Any:
        kwargs["shuffle"] = True
        ids, metadata = self._apply_params(batch, query, **kwargs)
        return ids[:batch.batch_size], metadata[:batch.batch_size]

    def sampled_batch(self, batch: BatchQueryResult, query: BatchQuery, **kwargs: Any) -> Any:
        kwargs["shuffle"] = True
        ids, metadata = self._apply_params(batch, query, **kwargs)
        return ids[:batch.batch_size], metadata[:batch.batch_size]

    @staticmethod
    def _apply_params(batch: BatchQueryResult, query: BatchQuery, **kwargs: Any):

        if len(batch.results) != len(query.queries):
            raise ValueError("Number of results is not equal to number of queries!")

        if "apply_threshold" in kwargs:
            if isinstance(kwargs["apply_threshold"], list):
                if kwargs["apply_threshold"][0]:
                    for i in range(len(batch.results)):
                        batch.results[i] = batch.results[i].apply_threshold(kwargs["apply_threshold"][1])
            else:
                if kwargs["apply_threshold"] > 0:
                    for i in range(len(batch.results)):
                        batch.results[i] = batch.results[i].apply_threshold(kwargs["apply_threshold"])

        if "apply_mmr" in kwargs:
            if kwargs["apply_mmr"]:
                i = 0
                for q, embeddings in zip(query.queries, batch.results):
                    if q.embedding is None:
                        raise ValueError("Embedding cannot be None")
                    batch.results[i] = maximal_marginal_relevance(q.embedding, embeddings, 0.5, q.top_k_mmr)
                    i += 1

        if "remove_duplicates" in kwargs:
            if kwargs["remove_duplicates"]:
                batch.remove_duplicates()

        batch.sort()

        ids: List[str] = []
        metadata: List[QueryMetadata] = []

        for i, result in enumerate(batch.results):
            k = query.queries[i].top_k
            if result.ids is not None and result.metadata is not None:
                ids += result.ids[:k]
                metadata += result.metadata[:k]
            else:
                raise ValueError("Result ids or metadata is None")

        if "shuffle" in kwargs:
            if kwargs["shuffle"]:
                import random
                c = list(zip(ids, metadata))
                random.shuffle(c)
                ids, metadata = (list(x) for x in zip(*c))

        return ids, metadata

    def _reset(self, *args, **kwargs):
        pass
