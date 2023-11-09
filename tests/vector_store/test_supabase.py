import pytest
import vecs
import warnings
from firstbatch.vector_store import Supabase
from firstbatch.vector_store.utils import generate_query, generate_batch
from firstbatch.vector_store.schema import BatchQueryResult, QueryResult, FetchQuery, FetchResult, BatchFetchQuery, BatchFetchResult
import os

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", ResourceWarning)


@pytest.fixture
def setup_supabase_client():
    supabase_uri = os.environ["SUPABASE_URL"]
    client = vecs.create_client(supabase_uri)
    dim = 1536
    return Supabase(client=client, collection_name="new", query_name="match_documents"), dim


def test_search(setup_supabase_client):
    supabase_client, dim = setup_supabase_client
    query = next(generate_query(1, dim, 10, True))
    res = supabase_client.search(query)
    assert isinstance(res, QueryResult)


def test_fetch(setup_supabase_client):
    supabase_client, dim = setup_supabase_client
    query = next(generate_query(1, dim, 10, False))
    res = supabase_client.search(query)
    assert isinstance(res, QueryResult)
    fetch = FetchQuery(id=res.ids[0])
    res = supabase_client.fetch(fetch)
    assert isinstance(res, FetchResult)


def test_multi_search(setup_supabase_client):
    supabase_client, dim = setup_supabase_client
    batch = generate_batch(10, dim, 10, True)
    res = supabase_client.multi_search(batch)
    assert isinstance(res, BatchQueryResult)


def test_multi_fetch(setup_supabase_client):
    supabase_client, dim = setup_supabase_client
    query = next(generate_query(1, dim, 10, False))
    res = supabase_client.search(query)
    assert isinstance(res, QueryResult)
    ids = [id for id in res.ids]
    bfq = BatchFetchQuery(batch_size=10, fetches=[FetchQuery(id=id) for id in ids])
    res = supabase_client.multi_fetch(bfq)
    assert isinstance(res, BatchFetchResult)


def test_history(setup_supabase_client):
    supabase_client, dim = setup_supabase_client
    query = next(generate_query(1, dim, 10, False))
    res = supabase_client.search(query)
    filt = supabase_client.history_filter([d.data[setup_supabase_client.history_field] for d in res.metadata])
    query.filter = filt
    res_ = supabase_client.search(query)
    assert len(set(res.ids).intersection(set(res_.ids))) == 0
