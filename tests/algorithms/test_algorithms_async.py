import pytest
from firstbatch import AsyncFirstBatch, Pinecone, Config, UserAction, Signal, AlgorithmLabel
import pinecone
import queue
import os


@pytest.fixture
def setup():
    api_key = os.environ["PINECONE_API_KEY"]
    env = os.environ["PINECONE_ENV"]
    index_name = "rss"
    pinecone.init(api_key=api_key, environment=env)
    pinecone.describe_index(index_name)
    index = pinecone.Index(index_name)

    cfg = Config(embedding_size=1536, batch_size=20, quantizer_train_size=100, quantizer_type="scalar",
                 enable_history=True, verbose=True)
    personalized = AsyncFirstBatch(api_key=os.environ["FIRSTBATCH_API_KEY"], **cfg.to_dict())
    return personalized, index


@pytest.mark.asyncio
async def test_async_simple(setup):
    actions = [("batch", 0), ("signal", 2), ("batch", 0)]
    action_queue = queue.Queue()
    for h in actions:
        action_queue.put(h)

    personalized, index = setup
    await personalized.add_vdb("pinecone_db_rss", Pinecone(index))
    session = await personalized.session(algorithm=AlgorithmLabel.SIMPLE, vdbid="pinecone_db_rss")
    ids, batch = [], []

    while not action_queue.empty():
        a = action_queue.get()
        if a[0] == "batch":
            ids, batch = await personalized.batch(session.data)
        elif a[0] == "signal":
            cid = a[1]
            await personalized.add_signal(session.data, UserAction(Signal.LIKE), ids[cid])

