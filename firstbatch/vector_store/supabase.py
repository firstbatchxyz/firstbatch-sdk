from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Union
from functools import partial
import asyncio
import logging
from firstbatch.vector_store.schema import MetadataFilter
from firstbatch.vector_store.base import VectorStore
from firstbatch.vector_store.schema import FetchQuery, Query, BatchQuery, BatchQueryResult,\
    QueryResult, QueryMetadata, BatchFetchQuery, BatchFetchResult, FetchResult, Vector, DistanceMetric
from firstbatch.lossy.base import BaseLossy, CompressedVector
from firstbatch.constants import DEFAULT_COLLECTION, DEFAULT_EMBEDDING_SIZE, DEFAULT_HISTORY_FIELD

if TYPE_CHECKING:
    from vecs import Client

logger = logging.getLogger("FirstBatchLogger")


class Supabase(VectorStore):

    def __init__(
        self,
        client: Client,
        collection_name: Optional[str] = None,
        query_name: Optional[str] = None,
        distance_metric: Optional[DistanceMetric] = None,
        history_field: Optional[str] = None,
        embedding_size: Optional[int] = None
    ) -> None:
        try:
            import vecs # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import supabase python package. "
                "Please install it with `pip install supabase`."
            )

        self._client = client
        self.collection_name = collection_name or DEFAULT_COLLECTION
        self.query_name = query_name or "match_documents"
        self._embedding_size = DEFAULT_EMBEDDING_SIZE if embedding_size is None else embedding_size
        self._collection = client.get_or_create_collection(name=self.collection_name, dimension=self._embedding_size)
        self._history_field = DEFAULT_HISTORY_FIELD if history_field is None else history_field
        self._distance_metric = DistanceMetric.COSINE_SIM if distance_metric is None else distance_metric
        logger.debug("Supabase/PGVector initialized with collection: {}".format(collection_name))

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

        if query.include_values:
            result = self._collection.query(query.embedding.vector, query.top_k, query.filter.filter,
                                            include_value=True, include_metadata=False)

            ids, scores, vectors, metadata = [], [], [], []
            ids_score = {r[0]: r[1] for r in result}
            fetches = self._collection.fetch(list(ids_score.keys()))
            for r in fetches:
                ids.append(r[0])
                scores.append(ids_score[r[0]])
                if query.include_values and query.include_metadata:
                    vectors.append(Vector(vector=r[1], dim=len(r[1]), id=r[0]))
                    metadata.append(QueryMetadata(id=r[0], data=r[2]))
        else:
            result = self._collection.query(query.embedding.vector, query.top_k, query.filter.filter,
                                            include_value=True, include_metadata=query.include_metadata)

            ids, scores, vectors, metadata = [], [], [], []
            for r in result:
                ids.append(r[0])
                scores.append(r[1])
                if query.include_metadata:
                    metadata.append(QueryMetadata(id=r[0], data=r[2]))
                vectors.append(Vector([], 0, r[0]))

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

    def fetch(self, query: FetchQuery, **kwargs: Any) -> FetchResult:
        assert query.id is not None, "id must be provided for fetch query"
        result = self._collection.fetch([query.id])
        m = QueryMetadata(id=query.id, data=result[0][2])
        v = Vector(vector=result[0][1], id=query.id, dim=len(result[0][1]))
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
        result = self._collection.fetch(ids)
        fetches = [FetchResult(id=idx[0], vector=Vector(vector=idx[1].tolist(), dim=len(idx[1].tolist())),
                               metadata=QueryMetadata(id=idx[0], data=idx[2])) for i, idx in enumerate(result)]
        return BatchFetchResult(batch_size=batch_query.batch_size, results=fetches)

    def history_filter(self, ids: List[str], prev_filter: Optional[Union[Dict, str]] = None) -> MetadataFilter:

        if prev_filter is not None and not isinstance(prev_filter, str):
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

            return MetadataFilter(name="History", filter=filter_)

