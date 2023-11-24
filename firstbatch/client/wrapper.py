from typing import Any, Dict, List
from firstbatch.client import BiasedBatchRequest, SignalRequest, SampledBatchRequest, CreateSessionRequest, \
    AddHistoryRequest, UpdateStateRequest
from firstbatch.vector_store.utils import generate_batch
from firstbatch.vector_store.schema import BatchQuery
from firstbatch.algorithm.blueprint import SignalObject, SessionObject
from firstbatch.constants import MINIMUM_TOPK


def history_request(session: SessionObject, ids: List[str]) -> AddHistoryRequest:
    return AddHistoryRequest(session, ids)


def session_request(**kwargs) -> CreateSessionRequest:
    return CreateSessionRequest(**kwargs)


def update_state_request(session: SessionObject, state: str, batch_type: str) -> UpdateStateRequest:
    return UpdateStateRequest(session=session, state=state, batch_type=batch_type)


def signal_request(session: SessionObject, state: str, signal: SignalObject) -> SignalRequest:
    return SignalRequest(session=session, vector=signal.vector.vector,
                         signal=signal.action.weight, state=state, signal_label=signal.action.label)


def sampled_batch_request(session: SessionObject, vdbid: str, state: str, n_topics: int) -> SampledBatchRequest:
    return SampledBatchRequest(session=session, n=n_topics, vdbid=vdbid, state=state)


def biased_batch_request(session: SessionObject, vdb_id: str, state: str, params: Dict[str, float], **kwargs) \
        -> BiasedBatchRequest:
    if "bias_vectors" in kwargs and "bias_weights" in kwargs:
        return BiasedBatchRequest(session, vdb_id, state,
                                  bias_vectors=kwargs["bias_vectors"], bias_weights=kwargs["bias_weights"],
                                  params=params)
    else:
        return BiasedBatchRequest(session, vdb_id, state, params=params)


def random_batch_request(batch_size: int, embedding_size: int, **kwargs) -> BatchQuery:
    return generate_batch(batch_size, embedding_size, top_k=MINIMUM_TOPK * 2,
                          include_values=("apply_mmr" in kwargs or "apply_threshold" in kwargs))
