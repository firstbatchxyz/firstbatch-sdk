import pytest
from firstbatch.lossy import ScalarQuantizer, CompressedVector
from firstbatch.vector_store.schema import Vector
import numpy as np


def random_vec(dim):
    vec = np.random.random(dim)
    vec /= (np.linalg.norm(vec) + np.finfo(np.float64).eps)
    return vec


@pytest.fixture
def setup_data():
    data = [Vector(random_vec(1536).tolist(), 1536, str(i)) for i in range(1000)]
    pq = ScalarQuantizer(256)
    return data, pq


def test_comp_decomp(setup_data):
    data, pq = setup_data
    pq.train(data)
    comp = pq.compress(data[0])
    assert isinstance(comp, CompressedVector)

    decomp = pq.decompress(comp)
    assert isinstance(decomp, Vector)

    error = np.sum(np.abs(np.array(decomp.vector) - np.array(data[0].vector)))
    print("Error", error)
