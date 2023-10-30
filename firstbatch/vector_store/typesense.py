from typing import TYPE_CHECKING, Any, Optional, cast, List, Dict, Union
from functools import partial
import asyncio
import logging
from firstbatch.vector_store.schema import MetadataFilter
from firstbatch.vector_store.base import VectorStore
from firstbatch.vector_store.schema import FetchQuery, Query, BatchQuery, BatchQueryResult, \
    QueryResult, QueryMetadata, BatchFetchQuery, BatchFetchResult, FetchResult, Vector, DistanceMetric
from firstbatch.constants import DEFAULT_COLLECTION, DEFAULT_EMBEDDING_SIZE
from firstbatch.lossy.base import BaseLossy, CompressedVector


if TYPE_CHECKING:
    from typesense.client import Client


logger = logging.getLogger("FirstBatchLogger")


class TypeSense(VectorStore):

    def __init__(self,
                 client: "Client",
                 collection_name: str = DEFAULT_COLLECTION,
                 distance_metric: Optional[DistanceMetric] = None,
                 history_field: Optional[str] = None,
                 embedding_size: Optional[int] = None
                 ) -> None:
        """Initialize params."""
        import_err_msg = (
            "`typesense` package not found, please run `pip install typesense`"
        )
        try:
            import typesense  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        if client is not None:
            if not isinstance(client, typesense.Client):
                raise ValueError(
                    f"client should be an instance of typesense.Client, "
                    f"got {type(client)}"
                )
            self._client = cast(typesense.Client, client)
        self._collection_name = collection_name
        self._collection = self._client.collections[self._collection_name]
        self._metadata_key = "metadata"
        self._embedding_size = DEFAULT_EMBEDDING_SIZE if embedding_size is None else embedding_size
        self._history_field = "_id" if history_field is None else history_field
        self._distance_metric = DistanceMetric.COSINE_SIM if distance_metric is None else distance_metric
        logger.debug("TypeSense initialized with collection: {}".format(collection_name))

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
        if query.filter.filter == {}:
            query_obj = {
                "q": "*",
                "vector_query": f'vec:({query.embedding.vector}, k:{query.top_k})',
                "collection": self._collection_name,
            }
        else:
            query_obj = {
                "q": "*",
                "vector_query": f'vec:({query.embedding.vector}, k:{query.top_k})',
                "collection": self._collection_name,
                "filter_by": str(query.filter.filter)
            }

        response = self._client.multi_search.perform(
            {"searches": [query_obj]}, {}
        )
        q = QueryResult([], [], [], [], distance_metric=self._distance_metric)
        for hit in response["results"][0]["hits"]:
            document = hit["document"]
            metadata = {k: v for k, v in document.items() if k != 'vec'}

            q.metadata.append(QueryMetadata(id="", data=metadata))  # type: ignore
            q.vectors.append(Vector(vector=document["vec"], dim=len(document["vec"]), id=document["id"]))  # type: ignore
            q.scores.append(hit["vector_distance"])  # type: ignore
            q.ids.append(document["id"])
        return q

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
        res = self._client.collections[self._collection_name].documents[query.id].retrieve()
        metadata = {k: v for k, v in res.items() if k != 'vec'}
        vec = Vector(vector=res["vec"], id=res["id"], dim=len(res["vec"]))
        return FetchResult(vector=vec, metadata=QueryMetadata(id=res["id"], data=metadata), id=res["id"])

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

        if self._history_field == "id":
            logger.debug("TypeSense doesn't allow filtering on id field. Try duplicating id in another field like _id.")
            raise ValueError("ID field error")

        filter_ = "{}:!=".format(self._history_field) + "[" + ",".join(ids) + "]"
        if prev_filter is not None and isinstance(prev_filter, str):
            filter_ += " && " + prev_filter
        return MetadataFilter(name="History", filter=filter_)
