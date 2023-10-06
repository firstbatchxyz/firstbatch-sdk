import pytest
import numpy as np
import nanopq
from firstbatch.lossy import ProductQuantizer, CompressedVector
from firstbatch.vector_store.schema import Vector


def random_vec(dim):
    vec = np.random.random(dim)
    vec /= (np.linalg.norm(vec) + np.finfo(np.float64).eps)
    return vec


@pytest.fixture
def setup_data():
    data = [Vector(random_vec(1536).tolist(), 1536, str(i)) for i in range(10000)]
    pq = ProductQuantizer(512, 32)
    return data, pq


def test_comp_decomp(setup_data):
    data, pq = setup_data
    pq.train(data)
    comp = pq.compress(data[0])
    assert isinstance(comp, CompressedVector)

    decomp = pq.decompress(comp)
    assert isinstance(decomp, Vector)
    print("Error", np.sum(np.abs(np.array(decomp.vector) - np.array(data[0].vector))))


def test_reproduce(setup_data):
    data, pq = setup_data
    pq.train(data)

    new_pq = nanopq.PQ(32, 512)
    new_pq_res = nanopq.PQ(32, 512)

    new_pq.codewords = pq.quantizer.codewords
    new_pq_res.codewords = pq.quantizer_residual.codewords

    new_pq.Ds = pq.quantizer.Ds
    new_pq_res.Ds = pq.quantizer_residual.Ds

    comp = pq.quantizer.encode(np.array(data[0].vector, dtype=np.float32).reshape(1, -1))
    comp2 = new_pq.encode(np.array(data[0].vector, dtype=np.float32).reshape(1, -1))
    assert comp.all() == comp2.all()

    comp = pq.quantizer_residual.encode(np.array(data[0].vector, dtype=np.float32).reshape(1, -1))
    comp2 = new_pq_res.encode(np.array(data[0].vector, dtype=np.float32).reshape(1, -1))
    assert comp.all() == comp2.all()
