"""Utility functions for working with vectors and vectorstores."""
from math import ceil
from typing import List, Union
import numpy as np
from firstbatch.vector_store.schema import Vector, Query, BatchQuery, QueryResult

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def random_vec(dim):
    vec = np.random.random(dim)
    vec /= (np.linalg.norm(vec) + np.finfo(np.float64).eps)
    return vec


def generate_vectors(dim, num_vectors):
    return [Vector(random_vec(dim).tolist(), dim, str(i)) for i in range(num_vectors)]


def generate_query(num_vecs: int, dim: int, top_k: int, include_values: bool):
    for vec in generate_vectors(dim, num_vecs):
        yield Query(embedding=vec, top_k=top_k, top_k_mmr=int(top_k/2), include_values=include_values)


def generate_batch(num_vecs: int, dim: int, top_k: int, include_values: bool):
    return BatchQuery(queries=list(generate_query(num_vecs, dim, top_k, include_values)), batch_size=num_vecs)


def adjust_weights(weights: List[float], batch_size: float, c: float) -> List[int]:
    # Ensure the minimum weight is at least 1
    min_weight = min(weights)
    if min_weight < 1:
        diff = 1 - min_weight
        weights = [w + diff for w in weights]

    # Scale the weights so their sum is approximately batch_size
    current_sum = sum(weights)
    target_sum = batch_size
    if not (batch_size - c <= current_sum <= batch_size + c):
        scale_factor = target_sum / current_sum
        weights = [ceil(w * scale_factor) for w in weights]

    return [int(w) for w in weights]


def maximal_marginal_relevance(
    query_embedding: Vector,
    batch: QueryResult,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> QueryResult:

    embeddings = batch.to_ndarray()
    query = np.array(query_embedding.vector)

    if min(k, len(embeddings)) <= 0:
        return batch

    embeddings_norm = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query)
    dists = (embeddings @ query) / (embeddings_norm * query_norm)
    minval = np.argsort(dists)
    indices = [minval[0]]
    selected :np.ndarray = np.array([embeddings[minval[0]]])
    while len(indices) < min(k, len(embeddings)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embeddings, selected)

        for i, query_score in enumerate(dists):
            if i in indices:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        indices.append(idx_to_add)
        selected = np.append(selected, [embeddings[idx_to_add]], axis=0)
    return QueryResult(
        ids=[batch.ids[i] for i in indices] if batch.ids is not None else [],
        metadata=[batch.metadata[i] for i in indices] if batch.metadata is not None else [],
        scores=[batch.scores[i] for i in indices] if batch.scores is not None else [],
        vectors=[batch.vectors[i] for i in indices] if batch.vectors is not None else []
        )


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])
    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    # Ignore divide by zero errors run time warnings as those are handled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


