from typing import Any, Dict, List
from firstbatch.client import BiasedBatchRequest, SignalRequest, SampledBatchRequest, CreateSessionRequest, \
    AddHistoryRequest, UpdateStateRequest
from firstbatch.vector_store.utils import generate_batch
from firstbatch.vector_store.schema import BatchQuery
from firstbatch.algorithm.blueprint import SignalObject
from firstbatch.constants import MINIMUM_TOPK


def history_request(session_id: str, ids: List[str]) -> AddHistoryRequest:
    return AddHistoryRequest(session_id, ids)


def session_request(**kwargs) -> CreateSessionRequest:
    return CreateSessionRequest(**kwargs)


def update_state_request(session_id: str, state: str) -> UpdateStateRequest:
    return UpdateStateRequest(id=session_id, state=state)


def signal_request(session_id: str, state: str, signal: SignalObject) -> SignalRequest:
    """Prepare a signal for the API. Add extra calculations if necessary
    @type signal: SignalObject
    """
    return SignalRequest(id=session_id, vector=signal.vector.vector,
                         signal=signal.action.weight, state=state)


def sampled_batch_request(session_id: str, vdbid: str, state:str, n_topics: int, **kwargs: Any) -> SampledBatchRequest:
    return SampledBatchRequest(id=session_id, n=n_topics, vdbid=vdbid, state=state)


def biased_batch_request(session_id: str, vdb_id: str, state: str, params: Dict[str, float], **kwargs) \
        -> BiasedBatchRequest:
    if "bias_vectors" in kwargs and "bias_weights" in kwargs:
        return BiasedBatchRequest(session_id, vdb_id, state,
                                  bias_vectors=kwargs["bias_vectors"], bias_weights=kwargs["bias_weights"], params=params)
    else:
        return BiasedBatchRequest(session_id, vdb_id, state, params=params)


def random_batch_request(batch_size: int, embedding_size: int, **kwargs) -> BatchQuery:
    return generate_batch(batch_size, embedding_size, top_k= MINIMUM_TOPK * 2,include_values=("apply_mmr" in kwargs or "apply_threshold" in kwargs))
