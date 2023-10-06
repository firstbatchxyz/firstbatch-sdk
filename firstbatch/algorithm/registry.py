from __future__ import annotations
from enum import Enum
from typing import Type, Dict, Union
# from firstbatch.algorithm.base import BaseAlgorithm


class AlgorithmLabel(str, Enum):
    SIMPLE = "SIMPLE"
    CUSTOM = "CUSTOM"
    UNIQUE_JOURNEYS = "Unique_Journeys".upper()
    CONTENT_CURATION = "User_Centric_Promoted_Content_Curations".upper()
    AI_AGENTS = "User_Intent_AI_Agents".upper()
    RECOMMENDATIONS = "Individually_Crafted_Recommendations".upper()
    NAVIGATION = "Navigable_UX".upper()


class AlgorithmRegistry:
    _registry: Dict[str, Type["BaseAlgorithm"]] = {}  # type: ignore

    @classmethod
    def register_algorithm(cls, label: Union[AlgorithmLabel, str], algo_class: Type["BaseAlgorithm"]) -> None:
        if isinstance(label, str):
            if label not in [AlgorithmLabel.SIMPLE, AlgorithmLabel.CUSTOM]:
                cls._registry["FACTORY"] = algo_class
                return
            label = AlgorithmLabel(label)
        cls._registry[label.name] = algo_class

    @classmethod
    def get_algorithm_by_label(cls, label: Union[AlgorithmLabel, str]) -> Type["BaseAlgorithm"]:
        """Retrieve a registered algorithm class by its label."""
        if isinstance(label, AlgorithmLabel):
            label = label.name
        return cls._registry[label]
