import pytest
import weaviate
import warnings
from firstbatch.vector_store import Weaviate
from firstbatch.vector_store.utils import generate_query, generate_batch
from firstbatch.vector_store.schema import BatchQueryResult, QueryResult, FetchQuery, BatchFetchQuery, BatchFetchResult, FetchResult
import os

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", ResourceWarning)


@pytest.fixture
def setup_weaviate_client():
    auth_config = weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"])
    client = weaviate.Client(
        url=os.environ["WEAVIATE_URL"],
        auth_client_secret=auth_config,
    )
    index_name = "default"
    dim = 1536
    return Weaviate(client=client, index_name=index_name), dim


def test_search(setup_weaviate_client):
    weaviate_client, dim = setup_weaviate_client
    query = next(generate_query(1, dim, 10, True))
    res = weaviate_client.search(query)
    assert isinstance(res, QueryResult)


def test_fetch(setup_weaviate_client):
    weaviate_client, dim = setup_weaviate_client
    query = next(generate_query(1, dim, 10, True))
    res = weaviate_client.search(query)
    assert isinstance(res, QueryResult)
    fetch = FetchQuery(id=res.ids[0])
    res = weaviate_client.fetch(fetch)
    assert isinstance(res, FetchResult)


def test_multi_search(setup_weaviate_client):
    weaviate_client, dim = setup_weaviate_client
    batch = generate_batch(10, dim, 10, True)
    res = weaviate_client.multi_search(batch)
    assert isinstance(res, BatchQueryResult)


def test_multi_fetch(setup_weaviate_client):
    weaviate_client, dim = setup_weaviate_client
    query = next(generate_query(1, dim, 10, True))
    res = weaviate_client.search(query)
    assert isinstance(res, QueryResult)
    ids = [id for id in res.ids]
    bfq = BatchFetchQuery(batch_size=10, fetches=[FetchQuery(id=id) for id in ids])
    res = weaviate_client.multi_fetch(bfq)
    assert isinstance(res, BatchFetchResult)


def test_history(setup_weaviate_client):
    weaviate_client, dim = setup_weaviate_client
    query = next(generate_query(1, dim, 10, False))
    res = weaviate_client.search(query)
    filt = weaviate_client.history_filter(res.ids)
    query.filter = filt
    res_ = weaviate_client.search(query)
    assert len(set(res.ids).intersection(set(res_.ids))) == 0
