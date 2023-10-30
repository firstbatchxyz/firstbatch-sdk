from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    List,
    Dict,
    Optional,
    Union
)
from firstbatch.lossy.base import CompressedVector
from firstbatch.vector_store.schema import FetchQuery, Query, BatchQuery, BatchQueryResult,\
    QueryResult, BatchFetchQuery, FetchResult, BatchFetchResult, Vector, MetadataFilter


class VectorStore(ABC):
    """Interface for vector store."""

    @property
    @abstractmethod
    def quantizer(self):
        ...

    @quantizer.setter
    @abstractmethod
    def quantizer(self, value):
        ...

    @property
    @abstractmethod
    def embedding_size(self):
        ...

    @embedding_size.setter
    @abstractmethod
    def embedding_size(self, value):
        ...

    @property
    @abstractmethod
    def history_field(self):
        ...

    @abstractmethod
    def train_quantizer(self, vectors: List[Vector]):
        ...

    @abstractmethod
    def quantize_vector(self, vector: Vector) -> CompressedVector:
        ...

    @abstractmethod
    def dequantize_vector(self, vector: CompressedVector) -> Vector:
        ...

    @abstractmethod
    def search(self, query: Query, **kwargs: Any) -> QueryResult:
        """Return docs most similar to query using specified search type."""
        ...

    @abstractmethod
    async def asearch(
        self, query: Query, **kwargs: Any
    ) -> QueryResult:
        """Return docs most similar to query using specified search type."""
        ...

    @abstractmethod
    def fetch(
        self, query: FetchQuery, **kwargs: Any
    ) -> FetchResult:
        """Return docs most similar to query using specified search type."""
        ...

    @abstractmethod
    async def afetch(
        self, query: FetchQuery, **kwargs: Any
    ) -> FetchResult:
        """Return docs most similar to query using specified search type."""
        ...

    @abstractmethod
    def multi_search(
        self, batch_query: BatchQuery, **kwargs: Any
    ) -> BatchQueryResult:
        """Return docs most similar to query using specified search type."""
        ...

    @abstractmethod
    def multi_fetch(
        self, batch_query: BatchFetchQuery, **kwargs: Any
    ) -> BatchFetchResult:
        """Return docs most similar to query using specified search type."""
        ...

    @abstractmethod
    async def a_multi_search(
        self, batch_query: BatchQuery, **kwargs: Any
    ) -> BatchQueryResult:
        """Return docs most similar to query using specified search type."""
        ...

    @abstractmethod
    def history_filter(self, ids: List[str], prev_filter: Optional[Union[Dict, str]] = None) -> MetadataFilter:
        """Return docs most similar to query using specified search type."""
        ...


