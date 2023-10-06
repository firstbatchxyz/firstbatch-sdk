from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Union
import concurrent.futures
from functools import partial
import asyncio
import logging
from firstbatch.vector_store.base import VectorStore
from firstbatch.vector_store.schema import FetchQuery, Query, BatchQuery, BatchQueryResult, \
    QueryResult, SearchType, QueryMetadata, BatchFetchQuery, BatchFetchResult, FetchResult, Vector, MetadataFilter, \
    DistanceMetric
from firstbatch.lossy.base import BaseLossy, CompressedVector
from firstbatch.constants import DEFAULT_EMBEDDING_SIZE

if TYPE_CHECKING:
    from pinecone import Index

logger = logging.getLogger("FirstBatchLogger")


class Pinecone(VectorStore):
    """`Pinecone` vector store."""

    def __init__(
            self,
            index: Index,
            namespace: Optional[str] = None,
            distance_metric: Optional[DistanceMetric] = None
    ):
        """Initialize with Pinecone client."""
        try:
            import pinecone
        except ImportError:
            raise ImportError(
                "Could not import pinecone python package. "
                "Please install it with `pip install pinecone-client`."
            )
        if not isinstance(index, pinecone.Index):
            raise ValueError(
                f"client should be an instance of pinecone.index.Index, "
                f"got {type(index)}"
            )
        self._index = index
        self._namespace = namespace
        self._embedding_size = DEFAULT_EMBEDDING_SIZE
        self._distance_metric = DistanceMetric.COSINE_SIM if distance_metric is None else distance_metric
        logger.debug("Pinecone vector store initialized with namespace: {}".format(namespace))

    @property
    def quantizer(self):
        return self._quantizer

    @quantizer.setter
    def quantizer(self, value):
        self._quantizer = value

    @property
    def embedding_size(self):
        return self._embedding_size

    @embedding_size.setter
    def embedding_size(self, value):
        self._embedding_size = value

    def train_quantizer(self, vectors: List[Vector]):
        if isinstance(self._quantizer, BaseLossy):
            self._quantizer.train(vectors)
        else:
            raise ValueError("Quantizer is not initialized or of the wrong type")

    def quantize_vector(self, vector: Vector) -> CompressedVector:
        return self._quantizer.compress(vector)

    def dequantize_vector(self, vector: CompressedVector) -> Vector:
        return self._quantizer.decompress(vector)

    def search(self, query: Query, **kwargs: Any) -> QueryResult:
        """Return docs most similar to query using specified search type."""
        if query.search_type == SearchType.FETCH:
            raise ValueError("search_type must be 'default' or 'sparse' to use search method")
        elif query.search_type == SearchType.SPARSE:
            raise NotImplementedError("Sparse search is not implemented yet.")
        else:
            result = self._index.query(query.embedding.vector,
                                       top_k=query.top_k,
                                       filter=query.filter.filter,
                                       include_metadata=query.include_metadata,
                                       include_values=query.include_values,
                                       )
            ids, scores, vectors, metadata = [], [], [], []
            for r in result["matches"]:
                ids.append(r["id"])
                scores.append(r["score"])
                vectors.append(Vector(vector=r["values"], dim=len(r["values"]), id=r["id"]))
                metadata.append(QueryMetadata(id=r["id"], data=r["metadata"]))

            return QueryResult(ids=ids, scores=scores, vectors=vectors, metadata=metadata,
                               distance_metric=self._distance_metric)

    async def asearch(
            self, query: Query, **kwargs: Any
    ) -> QueryResult:
        func = partial(
            self.search,
            query=query,
            **kwargs,
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def fetch(
            self, query: FetchQuery, **kwargs: Any
    ) -> FetchResult:
        """Return docs most similar to query using specified search type."""
        assert query.id is not None, "id must be provided for fetch query"
        result = self._index.fetch([query.id])
        fetches = []
        for k, v in result["vectors"].items():
            fetches.append(FetchResult(id=k, vector=Vector(vector=v["values"],
                                                           dim=len(v["values"]), id=k),
                                       metadata=QueryMetadata(id=k, data=v["metadata"])))

        return fetches[0]

    async def afetch(
            self, query: FetchQuery, **kwargs: Any
    ) -> FetchResult:
        func = partial(
            self.fetch,
            query=query,
            **kwargs,
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def multi_search_c(
            self, batch_query: BatchQuery, **kwargs: Any
    ) -> BatchQueryResult:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.search, batch_query.queries))
            return BatchQueryResult(results, batch_query.batch_size)

    def multi_search(self, batch_query: BatchQuery, **kwargs: Any) -> BatchQueryResult:
        async def _async_multi_search():
            coroutines = [self.asearch(query, **kwargs) for query in batch_query.queries]
            results = await asyncio.gather(*coroutines)
            return BatchQueryResult(results, batch_query.batch_size)

        return asyncio.run(_async_multi_search())

    async def a_multi_search(self, batch_query: BatchQuery, **kwargs: Any):
        coroutines = [self.asearch(query, **kwargs) for query in batch_query.queries]
        results = await asyncio.gather(*coroutines)
        return BatchQueryResult(results, batch_query.batch_size)

    def multi_fetch(
            self, batch_query: BatchFetchQuery, **kwargs: Any
    ) -> BatchFetchResult:

        ids = [q.id for q in batch_query.fetches]
        result = self._index.fetch(ids)
        fetches = [FetchResult(id=k, vector=Vector(vector=v["values"], dim=len(v["values"])),
                               metadata=QueryMetadata(id="", data=v["metadata"])
                               )
                   for k, v in result["vectors"].items()]
        return BatchFetchResult(batch_size=batch_query.batch_size, results=fetches)

    def history_filter(self, ids: List[str], prev_filter: Optional[Union[Dict, str]] = None,
                       id_field: str = "id") -> MetadataFilter:

        filter_ = {
            id_field: {"$nin": ids}
        }
        if prev_filter is not None:
            if isinstance(prev_filter, str):
                raise ValueError("prev_filter must be a dict for Pinecone")

            merged = prev_filter.copy()

            if id_field in prev_filter:
                merged[id_field]["$nin"] = list(set(prev_filter[id_field]["$nin"] + filter_[id_field]["$nin"]))
            else:
                merged[id_field] = filter_[id_field]

            for key, value in filter_.items():
                if key != id_field and key not in merged:
                    merged[key] = value

            return MetadataFilter(name="history", filter=merged)

        else:
            return MetadataFilter(name="History", filter=filter_)
