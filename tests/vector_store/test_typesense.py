import pytest
import typesense
from firstbatch.vector_store import TypeSense
from firstbatch.vector_store.utils import generate_query, generate_batch
from firstbatch.vector_store.schema import (
    BatchQueryResult, QueryResult, FetchQuery, FetchResult, BatchFetchQuery, BatchFetchResult)
import os

@pytest.fixture
def setup_typesense_client():
    client = typesense.Client({
        'api_key': os.environ["TYPESENSE_API_KEY"],
        'nodes': [{
            'host': os.environ["TYPESENSE_URL"],
            'port': '443',
            'protocol': 'https'  # or http if working locally
        }],
        'connection_timeout_seconds': 2
    })
    return TypeSense(client=client, collection_name="tiktok")


@pytest.fixture
def dim():
    return 384


def test_search(setup_typesense_client, dim):
    query = next(generate_query(1, dim, 10, True))
    res = setup_typesense_client.search(query)
    assert isinstance(res, QueryResult)


def test_fetch(setup_typesense_client, dim):
    query = next(generate_query(1, dim, 10, False))
    res = setup_typesense_client.search(query)
    assert isinstance(res, QueryResult)
    fetch = FetchQuery(id=res.ids[0])
    res = setup_typesense_client.fetch(fetch)
    assert isinstance(res, FetchResult)


def test_multi_search(setup_typesense_client, dim):
    batch = generate_batch(10, dim, 10, True)
    res = setup_typesense_client.multi_search(batch)
    assert isinstance(res, BatchQueryResult)


def test_multi_fetch(setup_typesense_client, dim):
    query = next(generate_query(1, dim, 10, False))
    res = setup_typesense_client.search(query)
    assert isinstance(res, QueryResult)
    ids = [id for id in res.ids]
    bfq = BatchFetchQuery(batch_size=10, fetches=[FetchQuery(id=id) for id in ids])
    res = setup_typesense_client.multi_fetch(bfq)
    assert isinstance(res, BatchFetchResult)


def test_history(setup_typesense_client, dim):
    query = next(generate_query(1, dim, 10, False))
    res = setup_typesense_client.search(query)
    filt = setup_typesense_client.history_filter([d.data[setup_typesense_client.history_field] for d in res.metadata])
    query.filter = filt
    res_ = setup_typesense_client.search(query)
    assert len(set(res.ids).intersection(set(res_.ids))) == 0
