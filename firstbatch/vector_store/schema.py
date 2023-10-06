from __future__ import annotations
from firstbatch.imports import Field, BaseModel
from dataclasses_json import DataClassJsonMixin
from dataclasses import dataclass
from typing import List, Optional, Union, Any, Dict
from enum import Enum
import numpy as np


@dataclass
class Vector(DataClassJsonMixin):
    vector: List[float]
    dim: int
    id: str = ""

    def concat(self, other: Vector) -> Vector:
        """Concatenates the current vector with another vector."""
        if not isinstance(other, Vector):
            raise ValueError("The 'other' parameter must be an instance of Vector.")
        new_vector = self.vector + other.vector
        new_dim = self.dim + other.dim
        new_id = self.id + "_" + other.id  # You can choose a different scheme for the new id if you want
        return Vector(vector=new_vector, dim=new_dim, id=new_id)


@dataclass
class Container:
    volume: Dict[str, Any]


class DistanceMetric(Enum):
    COSINE_SIM = "cosine_sim"
    EUCLIDEAN_DIST = "euclidean_dist"
    DOT_PRODUCT = "dot_product"


class SearchType(str, Enum):
    """Search types."""
    DEFAULT = "default"
    SPARSE = "sparse"
    FETCH = "fetch"


class MetadataFilter(BaseModel):
    """Interface for interacting with a document."""

    name: str
    filter: Union[dict, str] = {}


class QueryMetadata(BaseModel):
    """Interface for interacting with a document."""
    id: str
    data: dict = Field(default_factory=dict)


@dataclass
class Query:
    """Vector store query."""

    embedding: Vector
    top_k: int = 1
    top_k_mmr: int = top_k
    return_fields: Optional[List[str]] = None

    search_type: SearchType = SearchType.DEFAULT

    # metadata filters
    filter: MetadataFilter = MetadataFilter(name="")

    # include options
    include_metadata: bool = True
    include_values: bool = True

    # NOTE: currently only used by postgres hybrid search
    sparse_top_k: Optional[int] = None


@dataclass
class FetchQuery:
    """Vector store query."""

    id: str
    search_type: SearchType = SearchType.FETCH
    return_fields: Optional[List[str]] = None


@dataclass
class BatchFetchQuery:
    """Vector store query."""

    fetches: List[FetchQuery]
    batch_size: int = 1


@dataclass
class BatchQuery:
    """Vector store batch query."""

    queries: List[Query]
    batch_size: int = 1

    def concat(self, other: BatchQuery) -> BatchQuery:
        """Concatenate two BatchQuery objects."""
        if not isinstance(other, BatchQuery):
            raise ValueError("The 'other' parameter must be an instance of BatchQuery.")

        if not self.batch_size == other.batch_size:
            raise ValueError("The batch sizes must be equal.")

        # Extend the queries list
        new_queries = self.queries + other.queries

        return BatchQuery(batch_size=self.batch_size, queries=new_queries)


@dataclass
class FetchResult:
    """Vector store query result."""

    metadata: QueryMetadata
    vector: Vector
    id: str


@dataclass
class QueryResult:
    """Vector store query result."""

    ids: List[str]
    vectors: Optional[List[Vector]] = None
    metadata: Optional[List[QueryMetadata]] = None
    scores: Optional[List[float]] = None
    distance_metric: DistanceMetric = DistanceMetric.COSINE_SIM

    def to_ndarray(self) -> np.ndarray:

        if not self.scores:
            return np.array([])

        if self.vectors is None:
            raise ValueError("Vectors must be provided to convert to ndarray.")

        matrix = [vec.vector for vec in self.vectors]
        return np.array(matrix)

    def apply_threshold(self, threshold: float) -> QueryResult:
        if not self.scores:
            return self

        avg: float = float(np.mean(np.array(self.scores).ravel()).item())
        # TODO: Maybe remove this safety measure?
        if self.distance_metric == DistanceMetric.EUCLIDEAN_DIST:
            threshold = avg if threshold < avg else threshold
            indices_to_keep = [index for index, score in enumerate(self.scores) if score <= threshold]
        else:
            threshold = avg if threshold > avg else threshold
            indices_to_keep = [index for index, score in enumerate(self.scores) if score >= threshold]

        new_vectors = [self.vectors[i] for i in indices_to_keep] if self.vectors else None
        new_metadata = [self.metadata[i] for i in indices_to_keep] if self.metadata else None
        new_scores = [self.scores[i] for i in indices_to_keep] if self.scores else None
        new_ids = [self.ids[i] for i in indices_to_keep] if self.ids else None

        return QueryResult(
            vectors=new_vectors,
            metadata=new_metadata,
            scores=new_scores,
            ids=new_ids if new_ids else []
        )

    def remove_ids(self, ids: List[str]) -> QueryResult:
        if not self.ids:
            return self

        indices_to_keep = [index for index, id in enumerate(self.ids) if id not in ids]

        new_vectors = [self.vectors[i] for i in indices_to_keep] if self.vectors else None
        new_metadata = [self.metadata[i] for i in indices_to_keep] if self.metadata else None
        new_scores = [self.scores[i] for i in indices_to_keep] if self.scores else None
        new_ids = [self.ids[i] for i in indices_to_keep] if self.ids else None

        return QueryResult(
            vectors=new_vectors,
            metadata=new_metadata,
            scores=new_scores,
            ids= new_ids if new_ids else []
        )

    def concat(self, other: QueryResult) -> QueryResult:
        """Concatenate the current QueryResult with another QueryResult."""
        if not isinstance(other, QueryResult):
            raise ValueError("The 'other' parameter must be an instance of QueryResult.")

        new_vectors = (self.vectors or []) + (other.vectors or [])
        new_metadata = (self.metadata or []) + (other.metadata or [])
        new_scores = (self.scores or []) + (other.scores or [])
        new_ids = (self.ids or []) + (other.ids or [])

        return QueryResult(
            vectors=new_vectors,
            metadata=new_metadata,
            scores=new_scores,
            ids=new_ids
        )

    def non_unique_ids(self):
        from collections import Counter
        return [k for k, v in Counter(self.ids).items() if v > 1]


@dataclass
class BatchQueryResult:
    """Vector store query result."""

    results: List[QueryResult]
    batch_size: int = 1

    def vectors(self) -> List[Vector]:
        return [v for r in self.results if r.vectors is not None for v in r.vectors]

    def remove_duplicates(self) -> None:
        """ Remove duplicate ids from the batch result """
        if not self.results:
            return None

        flat = self.flatten()
        unique_ids = dict(zip(flat.ids, [0] * len(flat.ids)))
        # Initialize with the first result
        for result in self.results:
            selected_indices = []
            for i, _id in enumerate(result.ids):
                if unique_ids[_id] == 0:
                    unique_ids[_id] += 1
                    selected_indices.append(i)
            result.ids = [result.ids[i] for i in selected_indices]
            result.metadata = [result.metadata[i] for i in selected_indices if result.metadata is not None]
            result.scores = [result.scores[i] for i in selected_indices if result.scores is not None]
            result.vectors = [result.vectors[i] for i in selected_indices if result.vectors is not None]

    def sort(self) -> None:
        if not self.results:
            return None

        for result in self.results:
            if result.scores is None:
                raise ValueError("Scores must be provided to sort.")
            sorted_indices = sorted(range(len(result.scores)), key=lambda k: result.scores[k], reverse=True)  # type: ignore

            result.ids = [result.ids[i] for i in sorted_indices]
            result.metadata = [result.metadata[i] for i in sorted_indices if result.metadata is not None]
            result.scores = [result.scores[i] for i in sorted_indices]
            result.vectors = [result.vectors[i] for i in sorted_indices if result.vectors is not None]

    def flatten(self) -> QueryResult:
        """Flatten the batch results into a single QueryResult."""
        if not self.results:
            return QueryResult(ids=[])

        # Initialize with the first result
        flattened_result = self.results[0]

        # Iterate over the rest of the results and concatenate
        for result in self.results[1:]:
            flattened_result = flattened_result.concat(result)

        return flattened_result


@dataclass
class BatchFetchResult:
    """Vector store query result."""

    results: List[FetchResult]
    batch_size: int = 1

