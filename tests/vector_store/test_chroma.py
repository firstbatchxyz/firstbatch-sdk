import pytest
import chromadb
import warnings
from firstbatch.vector_store import Chroma
from firstbatch.vector_store.utils import generate_query, generate_batch
from firstbatch.vector_store.schema import BatchQueryResult, QueryResult, FetchQuery, FetchResult, BatchFetchQuery, BatchFetchResult
import os

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", ResourceWarning)


@pytest.fixture
def setup_chroma_client():
    path = os.environ["CHROMA_CLIENT_PATH"]
    collection = "default"
    client = chromadb.PersistentClient(path=path)
    dim = 1536
    return Chroma(client=client, collection_name=collection), dim


def test_search(setup_chroma_client):
    chroma_client, dim = setup_chroma_client
    query = next(generate_query(1, dim, 10, True))
    res = chroma_client.search(query)
    assert isinstance(res, QueryResult)


def test_fetch(setup_chroma_client):
    chroma_client, dim = setup_chroma_client
    query = next(generate_query(1, dim, 10, False))
    res = chroma_client.search(query)
    assert isinstance(res, QueryResult)
    fetch = FetchQuery(id=res.ids[0])
    res = chroma_client.fetch(fetch)
    assert isinstance(res, FetchResult)


def test_multi_search(setup_chroma_client):
    chroma_client, dim = setup_chroma_client
    batch = generate_batch(10, dim, 10, True)
    res = chroma_client.multi_search(batch)
    assert isinstance(res, BatchQueryResult)


def test_multi_fetch(setup_chroma_client):
    chroma_client, dim = setup_chroma_client
    query = next(generate_query(1, dim, 10, False))
    res = chroma_client.search(query)
    assert isinstance(res, QueryResult)
    ids = [id for id in res.ids]
    bfq = BatchFetchQuery(batch_size=10, fetches=[FetchQuery(id=id) for id in ids])
    res = chroma_client.multi_fetch(bfq)
    assert isinstance(res, BatchFetchResult)


def test_history(setup_chroma_client):
    """
    Not implemented for Chroma
    """
    assert True
