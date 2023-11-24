"""MasterClass for composition"""
from __future__ import annotations
from typing import Optional
import time
from collections import defaultdict
import logging
from firstbatch.algorithm import AlgorithmLabel, UserAction, BatchType, BatchEnum, \
    BaseAlgorithm, AlgorithmRegistry, SignalType, SessionObject
from firstbatch.client.schema import GetHistoryResponse
from firstbatch.vector_store import (
    VectorStore,
    FetchQuery,
    Query, Vector,
    MetadataFilter,
    adjust_weights,
    generate_batch
)
from firstbatch.lossy import ScalarQuantizer
from firstbatch.client import (
    BatchQuery,
    BatchResponse,
    SignalObject,
    session_request,
    signal_request,
    history_request,
    random_batch_request,
    biased_batch_request,
    sampled_batch_request,
    update_state_request
)

from firstbatch.constants import (
    DEFAULT_VERBOSE,
    DEFAULT_HISTORY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_QUANTIZER_TRAIN_SIZE,
    DEFAULT_QUANTIZER_TYPE,
    DEFAULT_TOPK_QUANT,
    DEFAULT_CONFIDENCE_INTERVAL_RATIO,
    MINIMUM_TOPK
)
from firstbatch.utils import Config
from firstbatch.client.async_client import AsyncFirstBatchClient
from firstbatch.logger_conf import setup_logger


class AsyncFirstBatch(AsyncFirstBatchClient):

    def __init__(self, api_key: str, config: Config):
        """
        Initialize the FirstBatch class
        :param api_key:
        :param config:
        """
        super().__init__(api_key)
        self.store: defaultdict[str, VectorStore] = defaultdict(VectorStore)
        self._batch_size = DEFAULT_BATCH_SIZE
        self._quantizer_train_size = DEFAULT_QUANTIZER_TRAIN_SIZE
        self._quantizer_type = DEFAULT_QUANTIZER_TYPE
        self._enable_history = DEFAULT_HISTORY
        self._verbose = DEFAULT_VERBOSE
        self.logger = setup_logger()
        self.logger.setLevel(logging.WARN)
        self._set_info()

        if config.verbose is not None:
            if config.verbose:
                self._verbose = config.verbose
                self.logger.setLevel(logging.INFO)
            else:
                self.logger.setLevel(logging.WARN)
        if config.batch_size is not None:
            self._batch_size = config.batch_size
        if config.quantizer_train_size is not None:
            self._quantizer_train_size = config.quantizer_train_size
        if config.quantizer_type is not None:
            self.logger.info("Product type quantizer is not supported yet.")
            # self._quantizer_type = kwargs["quantizer_type"]
            self._quantizer_type = "scalar"
        if config.enable_history is not None:
            self._enable_history = config.enable_history

        self.logger.info("Set mode to verbose")
        self.logger.info("Using: {}".format(self.url))

    async def add_vdb(self, vdbid: str, vs: VectorStore):

        exists = await self._vdb_exists(vdbid)

        if not exists:
            if self._quantizer_type == "scalar":
                self.logger.info("VectorDB with id {} not found. Sketching new VectorDB".format(vdbid))
                vs.quantizer = ScalarQuantizer(256)
                ts = min(int(self._quantizer_train_size/DEFAULT_TOPK_QUANT), 500)
                batch = generate_batch(ts, vs.embedding_size, top_k=DEFAULT_TOPK_QUANT, include_values=True)

                results = await vs.a_multi_search(batch)
                vs.train_quantizer(results.vectors())

                quantized_vectors = [vs.quantize_vector(vector).vector for vector in results.vectors()]
                # This might 1-2 minutes with scalar quantizer
                await self._init_vectordb_scalar(self.api_key, vdbid, quantized_vectors, vs.quantizer.quantiles)

                self.store[vdbid] = vs
            elif self._quantizer_type == "product":
                raise NotImplementedError("Product quantizer not supported yet")

            else:
                raise ValueError(f"Invalid quantizer type: {self._quantizer_type}")
        else:
            self.store[vdbid] = vs

    async def user_embeddings(self, session: SessionObject):
        return await self._get_user_embeddings(session)

    async def session(self, algorithm: AlgorithmLabel, vdbid: str, session_id: Optional[str] = None,
                custom_id: Optional[str] = None):

        if session_id is None:
            if algorithm == AlgorithmLabel.SIMPLE:
                req = session_request(**{"algorithm": algorithm.value, "vdbid": vdbid})

            elif algorithm == AlgorithmLabel.CUSTOM:
                req = session_request(**{"algorithm": algorithm.value, "vdbid": vdbid, "custom_id": custom_id})

            else:
                req = session_request(**{"algorithm": "FACTORY", "vdbid": vdbid, "factory_id": algorithm.value})
        else:
            if algorithm == AlgorithmLabel.SIMPLE:
                req = session_request(**{"id": session_id, "algorithm": algorithm.value, "vdbid": vdbid})

            elif algorithm == AlgorithmLabel.CUSTOM:
                req = session_request(**{"id": session_id, "algorithm": algorithm.value, "vdbid": vdbid,
                                         "custom_id": custom_id})

            else:
                req = session_request(**{"id": session_id, "algorithm": "FACTORY", "vdbid": vdbid,
                                         "factory_id": algorithm.value})

        session = await self._create_session(req)
        return SessionObject(id=session.data, is_persistent=(session_id is not None))

    async def add_signal(self, session: SessionObject, user_action: UserAction, cid: str):
        response = await self._get_session(session)
        vs = self.store[response.vdbid]

        if not isinstance(user_action.action_type, SignalType):
            raise ValueError(f"Invalid action type: {user_action.action_type}")

        fetch = FetchQuery(id=cid)
        result = vs.fetch(fetch)

        algo_instance = self.__get_algorithm(vs.embedding_size, self._batch_size, response.algorithm,
                                             response.factory_id, response.custom_id)

        # Create a signal object based on user action and content id
        signal_obj = SignalObject(vector=result.vector, action=user_action.action_type, cid=cid, timestamp=int(time.time()))

        # Call blueprint_step to calculate the next state
        (next_state, batch_type, params) = algo_instance.blueprint_step(response.state, user_action)
        # Send signal
        resp = await self._signal(signal_request(session, next_state.name, signal_obj))

        if self._enable_history:
            await self._add_history(history_request(session, [result.metadata.data[vs.history_field]]))

        return resp.success

    async def batch(self, session: SessionObject, batch_size: Optional[int] = None, **kwargs):
        response = await self._get_session(session)
        vs = self.store[response.vdbid]

        self.logger.info("Session: {} {} {}".format(response.algorithm, response.factory_id, response.state))
        if batch_size is None:
            batch_size = self._batch_size

        algo_instance = self.__get_algorithm(vs.embedding_size, batch_size, response.algorithm, response.factory_id, response.custom_id)
        user_action = UserAction(BatchEnum.BATCH)

        (next_state, batch_type, params) = algo_instance.blueprint_step(response.state, user_action)

        self.logger.info("{} {}".format(batch_type, params.to_dict()))

        history = self._mock_history()
        if self._enable_history:
            history = await self._get_history(session)

        if batch_type == BatchType.RANDOM:
            query = random_batch_request(algo_instance.batch_size, vs.embedding_size, **params.to_dict())
            await self._update_state(update_state_request(session, next_state.name, "RANDOM"))
            batch_response = await vs.a_multi_search(query)
            ids, batch = algo_instance.random_batch(batch_response, query, **params.to_dict())

        elif batch_type == BatchType.PERSONALIZED or batch_type == BatchType.BIASED:

            if batch_type == BatchType.BIASED and not ("bias_vectors" in kwargs and "bias_weights" in kwargs):
                self.logger.info("Bias vectors and weights must be provided for biased batch.")
                raise ValueError("no bias vectors provided")

            if batch_type == BatchType.PERSONALIZED and ("bias_vectors" in kwargs and "bias_weights" in kwargs):
                del kwargs["bias_vectors"]
                del kwargs["bias_weights"]

            if not response.has_embeddings and batch_type == BatchType.PERSONALIZED:
                self.logger.info("No embeddings found for personalized batch. Switching to random batch.")
                query = random_batch_request(algo_instance.batch_size, vs.embedding_size, **{"apply_mmr": True})
                await self._update_state(update_state_request(session, next_state.name, batch_type="PERSONALIZED"))
                batch_response = await vs.a_multi_search(query)
                ids, batch = algo_instance.random_batch(batch_response, query, **params.to_dict())

            else:
                batch_response_ = await self._biased_batch(biased_batch_request(session, response.vdbid, next_state.name,
                                                                         params.to_dict(), **kwargs))
                query = self.__query_wrapper(response.vdbid, algo_instance.batch_size, batch_response_, history, **params.to_dict())
                batch = await vs.a_multi_search(query)
                ids, batch = algo_instance.biased_batch(batch, query, **params.to_dict())

        elif batch_type == BatchType.SAMPLED:
            batch_response_ = await self._sampled_batch(sampled_batch_request(session=session, vdbid=response.vdbid,
                                                                       state=next_state.name, n_topics=params.n_topics))
            query = self.__query_wrapper(response.vdbid, algo_instance.batch_size, batch_response_, history, **params.to_dict())
            batch = await vs.a_multi_search(query)
            ids, batch = algo_instance.sampled_batch(batch, query, **params.to_dict())

        else:
            raise ValueError(f"Invalid batch type: {next_state.batch_type}")

        if self._enable_history:
            content = [b.data[vs.history_field] for b in batch[:algo_instance.batch_size]]
            await self._add_history(history_request(session, content))

        return ids, batch

    def __query_wrapper(self, vdbid: str, batch_size: int, response: BatchResponse,
                        history: Optional[GetHistoryResponse], **kwargs):
        """
        Wrapper for the query method. It applies the parameters from the blueprint
        :param vdbid: VectorDB ID, str
        :param batch_size: Batch size, int
        :param response: response from the API, BatchResponse
        :param history: list of content ids, List[str]
        :param kwargs:
        :return:
        """

        topks = adjust_weights(response.weights, batch_size, max((batch_size * DEFAULT_CONFIDENCE_INTERVAL_RATIO), 1))
        m_filter = MetadataFilter(name="", filter={})
        # We need the vector values to apply MMR or threshold
        include_values = ("apply_mmr" in kwargs or "apply_threshold" in kwargs)
        apply_mmr = "apply_mmr" in kwargs and (kwargs["apply_mmr"] is True or kwargs["apply_mmr"] == 1)

        if self._enable_history:
            if history is None:
                self.logger.info("History is None, No filter will be applied.")
                history = GetHistoryResponse(ids=[])

            if "filter" in kwargs:
                m_filter = self.store[vdbid].history_filter(history.ids, kwargs["filter"])
            else:
                m_filter = self.store[vdbid].history_filter(history.ids)

        if apply_mmr:
            # increase top_k for MMR to work better
            qs = [Query(Vector(vec, len(vec), ""),  max(topks[i], MINIMUM_TOPK) * 2, filter=m_filter, include_values=include_values) for i, vec
                  in
                  enumerate(response.vectors)]
        else:
            qs = [Query(Vector(vec, len(vec), ""),  max(topks[i], MINIMUM_TOPK), filter=m_filter, include_values=include_values) for i, vec in
                  enumerate(response.vectors)]
        return BatchQuery(qs, batch_size)

    def __get_algorithm(self, embedding_size: int, batch_size: int, algorithm: str, factory_id: Optional[str] = None,
                        custom_id: Optional[str] = None) -> BaseAlgorithm:

        if algorithm == "SIMPLE":
            algo_type = AlgorithmRegistry.get_algorithm_by_label(algorithm)
            algo_instance: BaseAlgorithm = algo_type(batch_size, **{"embedding_size": embedding_size})
        elif algorithm == "CUSTOM":
            if custom_id is None:
                raise ValueError("Custom algorithm id is None")
            bp = self._get_blueprint(custom_id)
            algo_type = AlgorithmRegistry.get_algorithm_by_label(algorithm)
            algo_instance: BaseAlgorithm = algo_type(bp, batch_size, **{"embedding_size": embedding_size})  # type: ignore
        elif algorithm == "FACTORY":
            algo_type = AlgorithmRegistry.get_algorithm_by_label(algorithm)
            algo_instance: BaseAlgorithm = algo_type(factory_id, batch_size, **{"embedding_size": embedding_size})  # type: ignore
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")
        return algo_instance


