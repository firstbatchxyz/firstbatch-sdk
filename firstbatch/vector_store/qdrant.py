from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Union
import concurrent.futures
from functools import partial
import asyncio
import logging
from firstbatch.vector_store.base import VectorStore
from firstbatch.vector_store.schema import FetchQuery, Query, BatchQuery, BatchQueryResult, \
    QueryResult, QueryMetadata, BatchFetchQuery, BatchFetchResult, FetchResult, Vector, MetadataFilter, \
    DistanceMetric
from firstbatch.lossy.base import BaseLossy, CompressedVector
from firstbatch.constants import DEFAULT_EMBEDDING_SIZE, DEFAULT_COLLECTION, DEFAULT_HISTORY_FIELD

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import FieldCondition, MatchExcept, Filter

logger = logging.getLogger("FirstBatchLogger")


class Qdrant(VectorStore):
    """`Qdrant` vector store."""

    def __init__(
            self,
            client: QdrantClient,
            collection_name: Optional[str] = None,
            distance_metric: Optional[DistanceMetric] = None,
            history_field: Optional[str] = None,
            embedding_size: Optional[int] = None
    ):
        """Initialize with Qdrant client."""
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "Could not import qdrant_client python package. "
                "Please install it with `pip install qdrant-client`."
            )
        if not isinstance(client, QdrantClient):
            raise ValueError(
                f"client should be an instance of QdrantClient, "
                f"got {type(client)}"
            )
        self._client = client
        self._collection_name = collection_name if collection_name is not None else DEFAULT_COLLECTION
        self._embedding_size = DEFAULT_EMBEDDING_SIZE if embedding_size is None else embedding_size
        self._history_field = DEFAULT_HISTORY_FIELD if history_field is None else history_field
        self._distance_metric = DistanceMetric.COSINE_SIM if distance_metric is None else distance_metric
        logger.debug("Qdrant vector store initialized for collection {}".format(collection_name))

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
        """Return docs most similar to query using specified search type."""
        if query.filter.filter is None:
            query.filter.filter = {}

        result = self._client.search(
            collection_name=self._collection_name,
            query_vector=query.embedding.vector,
            limit=query.top_k,
            append_payload=query.include_metadata,
            with_vectors=query.include_values,
            query_filter=query.filter.filter
        )

        ids, scores, vectors, metadata = [], [], [], []
        for r in result:
            ids.append(str(r.id))
            scores.append(r.score)
            if r.vector is not None:
                vectors.append(Vector(vector=r.vector, dim=len(r.vector), id=str(r.id)))
            if r.payload is not None:
                metadata.append(QueryMetadata(id=str(r.id), data=r.payload))

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
        id_: Union[str, int]
        try:
            id_ = int(query.id)
        except:
            id_ = query.id

        result = self._client.retrieve(collection_name=self._collection_name, ids=[id_], with_vectors=True)
        fetches = []
        for r in result:
            fetches.append(FetchResult(id=str(r.id), vector=Vector(vector=r.vector,
                                                           dim=len(r.vector), id=str(r.id)),
                                       metadata=QueryMetadata(id=str(r.id), data=r.payload)))

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

        ids:List[Union[str, int]] = []
        for q in batch_query.fetches:
            try:
                ids.append(int(q.id))
            except:
                ids.append(q.id)

        result = self._client.retrieve(collection_name=self._collection_name, ids=ids, with_vectors=True)
        fetches = []
        for r in result:
            fetches.append(FetchResult(id=str(r.id), vector=Vector(vector=r.vector,
                                                                   dim=len(r.vector), id=str(r.id)),
                                       metadata=QueryMetadata(id=str(r.id), data=r.payload)))
        return BatchFetchResult(batch_size=batch_query.batch_size, results=fetches)

    def history_filter(self, ids: List[str], prev_filter: Optional[Union[Dict, str]] = None) -> MetadataFilter:

        from qdrant_client.http.models import FieldCondition, MatchExcept, Filter
        import json

        if prev_filter is not None:
            raise ValueError("Qdrant implementation currently does not support history filter with previous filter")

        filter_ = Filter(
            must=
            [
                FieldCondition(
                key=self._history_field,
                match=MatchExcept(**{"except": ids}))
            ]
        ).model_dump_json()

        filter_ = json.loads(filter_)
        if "except_" in filter_["must"][0]["match"]:
            filter_["must"][0]["match"]["except"] = filter_["must"][0]["match"]["except_"]
            del filter_["must"][0]["match"]["except_"]

        return MetadataFilter(name="History", filter=filter_)
