import pytest
from firstbatch import FirstBatch, Qdrant, Config, UserAction, Signal, AlgorithmLabel
from qdrant_client import QdrantClient
import queue
from firstbatch.vector_store.utils import generate_vectors
import os


@pytest.fixture
def setup():
    embedding_size = int(os.environ["EMBEDDING_SIZE"])
    vdb_name = os.environ["VDB_NAME"]
    client = QdrantClient(os.environ["QDRANT_URL"])
    cl = Qdrant(client=client, collection_name="farcaster", embedding_size=embedding_size)

    cfg = Config(batch_size=20, quantizer_train_size=100, quantizer_type="scalar",
                 enable_history=True, verbose=True)
    personalized = FirstBatch(api_key=os.environ["FIRSTBATCH_API_KEY"], config=cfg)
    personalized.add_vdb(vdb_name, cl)

    return personalized, vdb_name


def test_simple(setup):
    actions = [("batch", 0), ("signal", 2), ("batch", 0)]
    action_queue = queue.Queue()
    for h in actions:
        action_queue.put(h)

    personalized, vdbid = setup
    session = personalized.session(algorithm=AlgorithmLabel.SIMPLE, vdbid=vdbid)
    ids, batch = [], []

    while not action_queue.empty():
        a = action_queue.get()
        if a[0] == "batch":
            ids, batch = personalized.batch(session)
        elif a[0] == "signal":
            cid = a[1]
            personalized.add_signal(session, UserAction(Signal.LIKE), ids[cid if cid < len(ids) else len(ids)-1])


def test_w_bias_vectors(setup):

    actions = [("batch", 0), ("signal", 2), ("batch", 0), ("signal", 4), ("batch", 0),
               ("batch", 0), ("signal", 1), ("signal", 2), ("signal", 3), ("batch", 0)]
    action_queue = queue.Queue()
    for h in actions:
        action_queue.put(h)

    starting_vectors = [vec.vector for vec in generate_vectors(int(os.environ["EMBEDDING_SIZE"]), 5)]
    starting_weights = [1.0] * 5
    data = {"bias_vectors": starting_vectors, "bias_weights": starting_weights}

    personalized, vdbid = setup
    session = personalized.session(algorithm=AlgorithmLabel.SIMPLE, vdbid=vdbid)
    ids, batch = [], []

    while not action_queue.empty():
        a = action_queue.get()
        if a[0] == "batch":
            ids, batch = personalized.batch(session, **data)
        elif a[0] == "signal":
            cid = a[1]
            personalized.add_signal(session, UserAction(Signal.LIKE), ids[cid if cid < len(ids) else len(ids)-1])

def test_custom(setup):

    actions = [("batch", 0), ("signal", 2), ("batch", 0)]
    action_queue = queue.Queue()
    for h in actions:
        action_queue.put(h)

    personalized, vdbid = setup
    session = personalized.session(algorithm=AlgorithmLabel.CUSTOM, vdbid=vdbid, custom_id="f23a2cfe-5a38-4671-927d-0897c01a2d25")
    ids, batch = [], []

    while not action_queue.empty():
        a = action_queue.get()
        if a[0] == "batch":
            ids, batch = personalized.batch(session)
        elif a[0] == "signal":
            cid = a[1]
            personalized.add_signal(session, UserAction(Signal.ADD_TO_CART), ids[cid if cid < len(ids) else len(ids)-1])



