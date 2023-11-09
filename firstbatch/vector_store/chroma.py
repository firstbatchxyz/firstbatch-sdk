from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Union, cast
from functools import partial
import asyncio
import logging
from firstbatch.constants import DEFAULT_COLLECTION, DEFAULT_EMBEDDING_SIZE, DEFAULT_HISTORY_FIELD
from firstbatch.vector_store.schema import MetadataFilter
from firstbatch.vector_store.base import VectorStore
from firstbatch.vector_store.schema import FetchQuery, Query, BatchQuery, BatchQueryResult,\
    QueryResult, QueryMetadata, BatchFetchQuery, BatchFetchResult, FetchResult, Vector, DistanceMetric
from firstbatch.lossy.base import BaseLossy, CompressedVector

if TYPE_CHECKING:
    import chromadb
    import chromadb.config
    from chromadb.api.types import Where

logger = logging.getLogger("FirstBatchLogger")


class Chroma(VectorStore):

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[Any] = None,
        distance_metric: Optional[DistanceMetric] = None,
        history_field: Optional[str] = None,
        embedding_size: Optional[int] = None
    ) -> None:
        try:
            import chromadb
            import chromadb.config

        except ImportError:
            raise ImportError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )

        if client is not None:
            self._client_settings = client_settings
            self._client = client
            self._persist_directory = persist_directory
        else:
            raise ValueError("client should be an instance of chromadb.Client, got {type(client)}")

        self._collection: chromadb.Collection = self._client.get_collection(collection_name)
        self._embedding_size = DEFAULT_EMBEDDING_SIZE if embedding_size is None else embedding_size
        self._history_field = DEFAULT_HISTORY_FIELD if history_field is None else history_field
        self._distance_metric: DistanceMetric = DistanceMetric.COSINE_SIM if distance_metric is None else distance_metric
        logger.debug("Chrome vector store initialized with collection: {}".format(collection_name))

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

    @property
    def history_field(self):
        return self._history_field

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
        from chromadb.api.types import Where

        if query.include_values:
            include = ["metadatas", "documents", "distances", "embeddings"]
        else:
            include = ["metadatas", "documents", "distances"]

        if query.embedding is None:
            raise ValueError("Query must have an embedding.")

        result = self._collection.query(
            query_embeddings=query.embedding.vector,
            n_results=query.top_k,
            where=cast(Optional[Where], query.filter.filter),
            include=include,  # type: ignore
            **kwargs,
        )

        metadatas = result.get("metadatas")
        ids = result.get("ids")
        distances = result.get("distances")
        if None in [metadatas, ids, distances]:
            raise ValueError("Query result does not contain metadatas, ids or distances.")

        if query.include_values:
            if result.get("embeddings") is None:
                raise ValueError("Query result does not contain embeddings.")
            vectors = [Vector(vec, len(query.embedding.vector), "") for vec in result["embeddings"][0]]  # type: ignore
        else:
            vectors = []

        metadata = [QueryMetadata(id=result["ids"][0][i], data=doc)
                    for i, doc in enumerate(result["metadatas"][0])]  # type: ignore

        return QueryResult(ids=result["ids"][0], scores=result["distances"][0], vectors=vectors, metadata=metadata,  # type: ignore
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

    def fetch(self, query: FetchQuery, **kwargs: Any) -> FetchResult:
        if query.id is None:
            raise ValueError("id must be provided for fetch query")

        result = self._collection.get(query.id, include=["metadatas", "documents", "embeddings"])

        metadatas = result.get("metadatas")
        distances = result.get("embeddings")
        if None in [metadatas, distances]:
            raise ValueError("Query result does not contain metadatas, ids or distances.")

        m = QueryMetadata(id=query.id, data=result["metadatas"][0])  # type: ignore
        v = Vector(vector=result["embeddings"][0], id=query.id, dim=len(result["embeddings"][0]))  # type: ignore
        return FetchResult(id=query.id, vector=v, metadata=m)

    async def afetch(
        self, query: FetchQuery, **kwargs: Any
    ) -> FetchResult:
        func = partial(
            self.fetch,
            query=query,
            **kwargs,
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

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
        result = self._collection.get(ids, include=["metadatas", "documents", "embeddings"])
        fetches = [FetchResult(id=idx, vector=Vector(vector=result["embeddings"][i], dim=len(result["embeddings"][i])), # type: ignore
                               metadata=QueryMetadata(id=idx, data=result["metadatas"][i])) for i, idx in enumerate(result["ids"])] # type: ignore
        return BatchFetchResult(batch_size=batch_query.batch_size, results=fetches)

    def history_filter(self, ids: List[str], prev_filter: Optional[Union[Dict, str]] = None) -> MetadataFilter:

        if isinstance(prev_filter, str):
            raise ValueError("prev_filter must be a dict for Chroma")

        if prev_filter is not None:
            if "$and" not in prev_filter:
                prev_filter["$and"] = []
            for id in ids:
                prev_filter["$and"].append({self._history_field: {"$ne": id}})

            return MetadataFilter(name="", filter=prev_filter)
        else:
            filter_: Dict = {
                "$and": [
                ]
            }
            for id in ids:
                filter_["$and"].append({self._history_field: {"$ne": id}})

            return MetadataFilter(name="", filter=filter_)