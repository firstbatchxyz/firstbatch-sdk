import pytest
import pinecone
import warnings
from firstbatch.vector_store import Pinecone
from firstbatch.vector_store.utils import generate_query, generate_batch
from firstbatch.vector_store.schema import BatchQueryResult, QueryResult, FetchQuery, FetchResult, BatchFetchQuery, BatchFetchResult
import os

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", ResourceWarning)


@pytest.fixture
def setup_pinecone_client():
    api_key = os.environ["PINECONE_API_KEY"]
    env =  os.environ["PINECONE_ENV"]
    index_name =  os.environ["INDEX_NAME"]
    dim = 384
    pinecone.init(api_key=api_key, environment=env)
    pinecone.describe_index(index_name)
    index = pinecone.Index(index_name)
    return Pinecone(index=index, namespace=None), dim


def test_search(setup_pinecone_client):
    pinecone_client, dim = setup_pinecone_client
    query = next(generate_query(1, dim, 10, True))
    res = pinecone_client.search(query)
    assert isinstance(res, QueryResult)


def test_fetch(setup_pinecone_client):
    pinecone_client, dim = setup_pinecone_client
    query = next(generate_query(1, dim, 10, False))
    res = pinecone_client.search(query)
    assert isinstance(res, QueryResult)
    fetch = FetchQuery(id=res.ids[0])
    res = pinecone_client.fetch(fetch)
    assert isinstance(res, FetchResult)


def test_multi_search(setup_pinecone_client):
    pinecone_client, dim = setup_pinecone_client
    batch = generate_batch(10, dim, 10, True)
    res = pinecone_client.multi_search(batch)
    assert isinstance(res, BatchQueryResult)


def test_multi_fetch(setup_pinecone_client):
    pinecone_client, dim = setup_pinecone_client
    query = next(generate_query(1, dim, 10, False))
    res = pinecone_client.search(query)
    assert isinstance(res, QueryResult)
    ids = [id for id in res.ids]
    bfq = BatchFetchQuery(batch_size=10, fetches=[FetchQuery(id=id) for id in ids])
    res = pinecone_client.multi_fetch(bfq)
    assert isinstance(res, BatchFetchResult)


def test_history(setup_pinecone_client):
    pinecone_client, dim = setup_pinecone_client
    query = next(generate_query(1, dim, 10, False))
    res = pinecone_client.search(query)
    filt = pinecone_client.history_filter([d.data[pinecone_client.history_field] for d in res.metadata])
    query.filter = filt
    res_ = pinecone_client.search(query)
    assert len(set(res.ids).intersection(set(res_.ids))) == 0
