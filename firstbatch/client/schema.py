from pydantic import BaseModel
from typing import List, Optional, Dict, Union, Any
from firstbatch.algorithm.registry import AlgorithmLabel
from firstbatch.algorithm.blueprint.base import SessionObject
from dataclasses_json import DataClassJsonMixin
from dataclasses import dataclass


@dataclass
class Session(DataClassJsonMixin):
    id: str
    algorithm: AlgorithmLabel
    state: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InitRequest:
    vdbid: str
    vecs: List[List[int]]
    quantiles: List[float]
    key: str


@dataclass
class AddHistoryRequest:
    session: SessionObject
    ids: List[str]


@dataclass
class CreateSessionRequest:
    algorithm: str
    vdbid: str
    has_embeddings: bool = False
    custom_id : Optional[str] = None
    factory_id: Optional[str] = None
    id: Optional[str] = None


@dataclass
class UpdateStateRequest:
    session: SessionObject
    state: str
    batch_type: Optional[str] = None


@dataclass
class SignalRequest:
    session: SessionObject
    vector: List[float]
    signal: float
    signal_label: str
    state: str


@dataclass
class BiasedBatchRequest:
    session: SessionObject
    vdbid: str
    state: str
    bias_vectors: Optional[List[List[float]]] = None
    bias_weights: Optional[List[float]] = None
    params: Optional[Dict[str, float]] = None


@dataclass
class SampledBatchRequest:
    session: SessionObject
    n: int
    vdbid: str
    state: str
    params: Optional[Dict[str, float]] = None


class APIResponse(BaseModel):
    success: bool
    code: int
    data: Optional[Union[str, Dict[str, Union[str, int, List[str] ,List[float], List[List[float]]]]]]
    message: Optional[str] = None


class GetHistoryResponse(BaseModel):
    ids: List[str]


class GetSessionResponse(BaseModel):
    state: str
    algorithm: str
    vdbid: str
    has_embeddings: bool
    factory_id: Optional[str] = None
    custom_id: Optional[str] = None


class SignalResponse(BaseModel):
    ...


class BatchResponse(BaseModel):
    vectors: List[List[float]]
    weights: List[float]


class FirstBatchAPIError(Exception):
    ...
