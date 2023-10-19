import pytest
from firstbatch import AsyncFirstBatch, Pinecone, Config, UserAction, Signal, AlgorithmLabel
import pinecone
import queue
import os


@pytest.fixture
def setup():
    api_key = os.environ["PINECONE_API_KEY"]
    env = os.environ["PINECONE_ENV"]
    vdb_name = os.environ["VDB_NAME"]
    index_name = os.environ["INDEX_NAME"]
    embedding_size = int(os.environ["EMBEDDING_SIZE"])

    pinecone.init(api_key=api_key, environment=env)
    pinecone.describe_index(index_name)
    index = pinecone.Index(index_name)

    config = Config(embedding_size=embedding_size, batch_size=20, quantizer_train_size=100, quantizer_type="scalar",
                 enable_history=True, verbose=True)
    personalized = AsyncFirstBatch(api_key=os.environ["FIRSTBATCH_API_KEY"], config=config)
    return personalized, index, vdb_name


@pytest.mark.asyncio
async def test_async_simple(setup):
    actions = [("batch", 0), ("signal", 2), ("batch", 0)]
    action_queue = queue.Queue()
    for h in actions:
        action_queue.put(h)

    personalized, index, vdb = setup
    await personalized.add_vdb(vdb, Pinecone(index))
    session = await personalized.session(algorithm=AlgorithmLabel.SIMPLE, vdbid=vdb)
    ids, batch = [], []

    while not action_queue.empty():
        a = action_queue.get()
        if a[0] == "batch":
            ids, batch = await personalized.batch(session)
        elif a[0] == "signal":
            cid = a[1]
            await personalized.add_signal(session, UserAction(Signal.LIKE), ids[cid])

