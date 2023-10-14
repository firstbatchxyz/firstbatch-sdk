from abc import ABC, abstractmethod
from typing import List, Any
from firstbatch.client.schema import AddHistoryRequest, CreateSessionRequest, SignalRequest, BiasedBatchRequest, \
    SampledBatchRequest, UpdateStateRequest
from firstbatch.algorithm.blueprint.base import SessionObject


class BaseClient(ABC):

    @abstractmethod
    def _init_vectordb_scalar(self, key: str, vdbid:str, vecs:List[List[int]], quantiles: List[float]) -> Any:
        ...

    @abstractmethod
    def _add_history(self, req: AddHistoryRequest) -> Any:
        ...

    @abstractmethod
    def _create_session(self, req: CreateSessionRequest) -> Any:
        ...

    @abstractmethod
    def _update_state(self, req: UpdateStateRequest) -> Any:
        ...

    @abstractmethod
    def _signal(self, req: SignalRequest) -> Any:
        ...

    @abstractmethod
    def _biased_batch(self, req: BiasedBatchRequest) -> Any:
        ...

    @abstractmethod
    def _sampled_batch(self, req: SampledBatchRequest) -> Any:
        ...

    @abstractmethod
    def _get_session(self, session: SessionObject) -> Any:
        ...

    @abstractmethod
    def _get_history(self, session: SessionObject) -> Any:
        ...

    @abstractmethod
    def _get_user_embeddings(self, session: SessionObject) -> Any:
        ...

    @abstractmethod
    def _vdb_exists(self, vdbid: str) -> Any:
        ...

    @abstractmethod
    def _get_blueprint(self, custom_id: str) -> Any:
        ...

    @staticmethod
    @abstractmethod
    def _mock_history() -> Any:
        ...

    @abstractmethod
    def _set_info(self) -> Any:
        ...


