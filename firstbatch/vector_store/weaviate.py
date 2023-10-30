from __future__ import annotations

from typing import Any, Optional, List, Dict, Union
from functools import partial
import asyncio
import logging
from firstbatch.vector_store.base import VectorStore
from firstbatch.vector_store.schema import FetchQuery, Query, BatchQuery, BatchQueryResult, \
    QueryResult, SearchType, QueryMetadata, BatchFetchQuery, Vector, FetchResult, BatchFetchResult, MetadataFilter, \
    DistanceMetric
from firstbatch.lossy.base import BaseLossy, CompressedVector
from firstbatch.constants import DEFAULT_EMBEDDING_SIZE, DEFAULT_COLLECTION, DEFAULT_HISTORY_FIELD


logger = logging.getLogger("FirstBatchLogger")


class Weaviate(VectorStore):
    """`Weaviate` vector store."""

    def __init__(
            self,
            client: Any,
            index_name: Optional[str] = None,
            output_fields: Optional[List[str]] = None,
            distance_metric: Optional[DistanceMetric] = None,
            history_field: Optional[str] = None,
            embedding_size: Optional[int] = None
    ):
        """Initialize with Weaviate client."""

        if output_fields is None:
            output_fields = ["text"]
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "Could not import weaviate python package. "
                "Please install it with `pip install weaviate-client`."
            )
        if not isinstance(client, weaviate.Client):
            raise ValueError(
                f"client should be an instance of weaviate.Client, got {type(client)}"
            )
        self._client = client
        self._index_name = DEFAULT_COLLECTION if index_name is None else index_name
        self._output_fields = output_fields
        self._embedding_size = DEFAULT_EMBEDDING_SIZE if embedding_size is None else embedding_size
        self._history_field = DEFAULT_HISTORY_FIELD if history_field is None else history_field
        self._distance_metric = DistanceMetric.COSINE_SIM if distance_metric is None else distance_metric
        logger.debug("Weaviate initialized with index: {}".format(index_name))

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
        if query.search_type == SearchType.FETCH:
            raise ValueError("search_type must be 'default' or 'sparse' to use search method")
        elif query.search_type == SearchType.SPARSE:
            raise NotImplementedError("Sparse search not implemented for Weaviate")
        else:
            vector = {"vector": query.embedding.vector}
            query_obj = self._client.query.get(self._index_name, self._output_fields)
            if query.filter.filter != {}:
                query_obj = query_obj.with_where(query.filter.filter)
            if kwargs.get("additional"):
                query_obj = query_obj.with_additional(kwargs.get("additional"))
            if query.include_values:
                query_obj = query_obj.with_additional(["vector", "distance", "id"])
            else:
                query_obj = query_obj.with_additional(["distance", "id"])

            result = query_obj.with_near_vector(vector).with_limit(query.top_k).do()
            if "errors" in result:
                raise ValueError(f"Error during query: {result['errors']}")
            ids, scores, vectors, metadata = [], [], [], []
            for res in result["data"]["Get"][self._index_name.capitalize()]:
                _id = res["_additional"]["id"]
                ids.append(_id)
                m = {k: res.pop(k) for k in self._output_fields}
                metadata.append(QueryMetadata(id=_id, data=m))
                scores.append(res["_additional"]["distance"])
                if query.include_values:
                    vectors.append(Vector(vector=res["_additional"]["vector"],
                                          id=_id, dim=len(res["_additional"]["vector"])))
                else:
                    vectors.append(Vector(vector=[], id=_id, dim=0))

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
        # query_obj = self._client.query.get(self._index_name, self._search_properties)
        assert query.id is not None, "id must be provided for fetch query"
        data_object = self._client.data_object.get_by_id(
            query.id,
            class_name=self._index_name,
            with_vector=True
        )
        m = QueryMetadata(id=query.id, data=data_object["properties"])
        v = Vector(vector=data_object["vector"], id=query.id, dim=len(data_object["vector"]))
        return FetchResult(vector=v, metadata=m, id=query.id)

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
        async def _async_multi_fetch():
            coroutines = [self.afetch(fetch, **kwargs) for fetch in batch_query.fetches]
            results = await asyncio.gather(*coroutines)
            return BatchFetchResult(results, batch_query.batch_size)

        return asyncio.run(_async_multi_fetch())

    def history_filter(self, ids: List[str], prev_filter: Optional[Union[Dict, str]] = None) -> MetadataFilter:

        filter: Dict[str, Any]

        if prev_filter is not None:
            if isinstance(prev_filter, dict):
                filter = prev_filter
            else:
                raise TypeError("prev_filter must be a dictionary.")
        else:
            filter = {
                "operator": "And",
                "operands": []
            }

        for id in ids:
            f = {
                "path": [self._history_field],
                "operator": "NotEqual",
                "valueText": id
            }
            filter["operands"].append(f)

        return MetadataFilter(name="History", filter=filter)
