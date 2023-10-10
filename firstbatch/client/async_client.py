import httpx
from typing import Dict, List, Any, Union, Optional, cast
from pydantic import ValidationError
import hashlib
from firstbatch.constants import BASE_URL
from firstbatch.client.schema import (APIResponse, BatchResponse, GetSessionResponse, GetHistoryResponse,
                                      AddHistoryRequest, CreateSessionRequest, SignalRequest, BiasedBatchRequest,
                                      SampledBatchRequest, UpdateStateRequest, FirstBatchAPIError)
from firstbatch.client.base import BaseClient


class AsyncFirstBatchClient(BaseClient):
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

    async def _post_request(self, url: str, data: Dict) -> Dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=self.headers, json=data)
            return response.json()

    @staticmethod
    def __error_handling(response: Dict, func_name: str) -> None:
        if response["code"] != 200:
            raise FirstBatchAPIError(f"FirstBatch API error with code {response['status_code']} "
                                     f"in {func_name} with reason: {response['content']}")

    # For example:
    async def _init_vectordb_scalar(self, key: str, vdbid: str, vecs: List[List[int]], quantiles: List[float]) -> Any:
        m = hashlib.md5()
        m.update(key.encode())
        hash_value = m.hexdigest()
        data = {
            "key": hash_value,
            "vdbid": vdbid,
            "mode": "scalar",
            "quantized_vecs": vecs,
            "quantiles": quantiles
        }
        response = await self._post_request(BASE_URL + "embeddings/init_vectordb", data)
        self.__error_handling(response, "init_vectordb_scalar")
        try:
            return APIResponse(**response)
        except ValidationError as e:
            raise e

    async def _add_history(self, req: AddHistoryRequest) -> Any:
        data = {
            "id": req.id,
            "ids": req.ids
        }
        response = await self._post_request(BASE_URL + "embeddings/update_history", data)
        self.__error_handling(response, "add_history")
        try:
            return APIResponse(**response)
        except ValidationError as e:
            raise e

    async def _create_session(self, req: CreateSessionRequest) -> APIResponse:
        data = {"id": req.id, "algorithm": req.algorithm, "vdbid": req.vdbid,
                "custom_id": req.custom_id, "factory_id": req.factory_id, "has_embeddings": req.has_embeddings}
        response = await self._post_request(BASE_URL + "embeddings/create_session", data)
        self.__error_handling(response, "create_session")
        try:
            return APIResponse(**response)
        except ValidationError as e:
            raise e

    async def _update_state(self, req: UpdateStateRequest) -> APIResponse:
        data = {"id": req.id, "state": req.state}
        response = await self._post_request(BASE_URL + "embeddings/update_state", data)
        self.__error_handling(response, "update_state")
        try:
            return APIResponse(**response)
        except ValidationError as e:
            raise e

    async def _signal(self, req: SignalRequest) -> APIResponse:
        data = {
            "id": req.id,
            "vector": req.vector,
            "signal": req.signal,
            "state": req.state
        }
        response = await self._post_request(BASE_URL + "embeddings/signal", data)
        self.__error_handling(response, "signal")
        try:
            return APIResponse(**response)
        except ValidationError as e:
            raise e

    async def _biased_batch(self, req: BiasedBatchRequest) -> BatchResponse:
        data = {
            "id": req.id,
            "vdbid": req.vdbid,
            "bias_vectors": req.bias_vectors,
            "bias_weights": req.bias_weights,
            "params": req.params,
            "state": req.state
        }
        response = await self._post_request(BASE_URL + "embeddings/biased_batch", data)
        self.__error_handling(response, "biased_batch")
        try:
            api_response = APIResponse(**response).data
            if isinstance(api_response, dict):
                vectors = api_response.get('vectors')
                weights = api_response.get('weights')

                if vectors is not None and weights is not None:
                    vectors_casted = cast(List[List[float]], vectors)
                    weights_casted = cast(List[float], weights)
                    return BatchResponse(vectors=vectors_casted, weights=weights_casted)
                else:
                    raise ValueError("Missing 'vectors' or 'weights' in API response.")
            else:
                raise TypeError("Expected a dictionary in APIResponse.data.")
        except ValidationError as e:
            raise e

    async def _sampled_batch(self, req: SampledBatchRequest) -> BatchResponse:
        data = {
            "id": req.id,
            "n": req.n,
            "vdbid": req.vdbid,
            "params": req.params,
            "state": req.state
        }
        response = await self._post_request(BASE_URL + "embeddings/sampled_batch", data)
        self.__error_handling(response, "sampled_batch")
        try:
            api_response = APIResponse(**response).data
            if isinstance(api_response, dict):
                vectors = api_response.get('vectors')
                weights = api_response.get('weights')

                if vectors is not None and weights is not None:
                    vectors_casted = cast(List[List[float]], vectors)
                    weights_casted = cast(List[float], weights)
                    return BatchResponse(vectors=vectors_casted, weights=weights_casted)
                else:
                    raise ValueError("Missing 'vectors' or 'weights' in API response.")
            else:
                raise TypeError("Expected a dictionary in APIResponse.data.")
        except ValidationError as e:
            raise e

    async def _get_session(self, id: str) -> GetSessionResponse:
        data = {"id": id}
        response = await self._post_request(BASE_URL + "embeddings/get_session", data)
        self.__error_handling(response, "get_session")
        try:
            api_response = APIResponse(**response).data
            if isinstance(api_response, dict):
                state = api_response.get("state", "")
                algorithm = api_response.get("algorithm", "")
                vdbid = api_response.get("vdbid", "")
                has_embeddings = api_response.get("has_embeddings", "")
                factory_id = api_response.get("factory_id")
                custom_id = api_response.get("custom_id")

                if state and algorithm and vdbid:
                    return GetSessionResponse(
                        state=cast(str, state),
                        algorithm=cast(str, algorithm),
                        vdbid=cast(str, vdbid),
                        has_embeddings=cast(bool, has_embeddings),
                        factory_id=cast(Optional[str], factory_id),
                        custom_id=cast(Optional[str], custom_id)
                    )
                else:
                    raise ValueError("Missing mandatory keys in API response.")
            else:
                raise TypeError("Expected a dictionary in APIResponse.data.")
        except ValidationError as e:
            raise e

    async def _get_history(self, id: str) -> GetHistoryResponse:
        data = {"id": id}
        response = await self._post_request(BASE_URL + "embeddings/get_history", data)
        self.__error_handling(response, "get_history")
        try:
            api_response = APIResponse(**response).data
            if isinstance(api_response, dict):
                ids = api_response.get('ids')
                if ids is not None:
                    ids_casted = cast(List[str], ids)
                    return GetHistoryResponse(ids=ids_casted)
                else:
                    raise ValueError("Missing 'ids' in API response.")
            else:
                raise TypeError("Expected a dictionary in APIResponse.data.")
        except ValidationError as e:
            raise e

    async def _get_user_embeddings(self, id: str, last_n: Optional[int] = None) -> BatchResponse:

        data = {"id": id, "last_n": 50}
        if last_n is None:
            data["last_n"] = last_n

        response = await self._post_request(BASE_URL + "embeddings/get_embeddings", data)
        self.__error_handling(response, "get_user_embeddings")
        try:
            api_response = APIResponse(**response).data
            if isinstance(api_response, dict):
                vectors = api_response.get('vectors')
                weights = api_response.get('weights')

                if vectors is not None and weights is not None:
                    vectors_casted = cast(List[List[float]], vectors)
                    weights_casted = cast(List[float], weights)
                    return BatchResponse(vectors=vectors_casted, weights=weights_casted)
                else:
                    raise ValueError("Missing 'vectors' or 'weights' in API response.")
            else:
                raise TypeError("Expected a dictionary in APIResponse.data.")
        except ValidationError as e:
            raise e

    async def _vdb_exists(self, vdbid: str) -> bool:
        data = {"vdbid": vdbid}
        response = await self._post_request(BASE_URL + "embeddings/vdb_exists", data)
        self.__error_handling(response, "vdb_exists")
        try:
            return response["data"]
        except ValidationError as e:
            raise e

    async def _get_blueprint(self, id: str) -> Any:
        data = {"id": id}
        response = await self._post_request(BASE_URL + "embeddings/get_blueprint", data)
        self.__error_handling(response, "get_blueprint")
        try:
            return response["data"]
        except ValidationError as e:
            raise e

    @staticmethod
    def _mock_history() -> GetHistoryResponse:
        return GetHistoryResponse(ids=[])
