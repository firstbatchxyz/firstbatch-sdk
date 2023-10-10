from abc import ABC, abstractmethod
from typing import List, Any
from firstbatch.client.schema import AddHistoryRequest, CreateSessionRequest, SignalRequest, BiasedBatchRequest, \
    SampledBatchRequest, UpdateStateRequest


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
    def _get_session(self, id: str) -> Any:
        ...

    @abstractmethod
    def _get_history(self, id: str) -> Any:
        ...

    @abstractmethod
    def _get_user_embeddings(self, id: str) -> Any:
        ...

    @abstractmethod
    def _vdb_exists(self, vdbid: str) -> Any:
        ...

    @abstractmethod
    def _get_blueprint(self, id: str) -> Any:
        ...

    @staticmethod
    @abstractmethod
    def _mock_history() -> Any:
        ...



