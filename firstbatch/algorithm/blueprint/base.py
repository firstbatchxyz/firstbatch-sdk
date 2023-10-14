from __future__ import annotations
import json
from typing import List, Dict, Union, Optional, Any, Tuple
from firstbatch.vector_store.schema import Vector
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from enum import Enum


@dataclass
class Params(DataClassJsonMixin):
    mu: float = 0.0
    alpha: float = 0.0
    r: float = 0.0
    last_n: int = 0
    n_topics: int = 0
    remove_duplicates: bool = True
    apply_threshold: Tuple[bool, float] = (False, 0.0)
    apply_mmr: bool = False


@dataclass
class SignalType:
    label: str
    weight: float

    def to_action(self):
        return self.label.upper()


class Signal:
    """Signal class"""
    DEFAULT = SignalType(label="default", weight=1.0)
    ADD_TO_CART = SignalType("ADD_TO_CART", 16)
    ITEM_VIEW = SignalType("ITEM_VIEW", 10)
    APPLY = SignalType("APPLY", 18)
    PURCHASE = SignalType("PURCHASE", 20)
    HIGHLIGHT = SignalType("HIGHLIGHT", 8)
    GLANCE_VIEW = SignalType("GLANCE_VIEW", 14)
    CAMPAIGN_CLICK = SignalType("CAMPAIGN_CLICK", 6)
    CATEGORY_VISIT = SignalType("CATEGORY_VISIT", 10)
    SHARE = SignalType("SHARE", 10)
    MERCHANT_VIEW = SignalType("MERCHANT_VIEW", 10)
    REIMBURSED = SignalType("REIMBURSED", 20)
    APPROVED = SignalType("APPROVED", 18)
    REJECTED = SignalType("REJECTED", 18)
    SHARE_ARTICLE = SignalType("SHARE_ARTICLE", 10)
    COMMENT = SignalType("COMMENT", 12)
    PERSPECTIVES_SWITCH = SignalType("PERSPECTIVES_SWITCH", 8)
    REPOST = SignalType("REPOST", 20)
    SUBSCRIBE = SignalType("SUBSCRIBE", 18)
    SHARE_PROFILE = SignalType("SHARE_PROFILE", 10)
    PAID_SUBSCRIBE = SignalType("PAID_SUBSCRIBE", 20)
    SAVE = SignalType("SAVE", 8)
    FOLLOW_TOPIC = SignalType("FOLLOW_TOPIC", 10)
    WATCH = SignalType("WATCH", 20)
    CLICK_LINK = SignalType("CLICK_LINK", 6)
    RECOMMEND = SignalType("RECOMMEND", 12)
    FOLLOW = SignalType("FOLLOW", 10)
    VISIT_PROFILE = SignalType("VISIT_PROFILE", 12)
    AUTO_PLAY = SignalType("AUTO_PLAY", 4)
    SAVE_ARTICLE = SignalType("SAVE_ARTICLE", 8)
    REPLAY = SignalType("REPLAY", 20)
    READ = SignalType("READ", 14)
    LIKE = SignalType("LIKE", 8)
    CLICK_EMAIL_LINK = SignalType("CLICK_EMAIL_LINK", 6)
    ADD_TO_LIST = SignalType("ADD_TO_LIST", 12)
    FOLLOW_AUTHOR = SignalType("FOLLOW_AUTHOR", 10)
    SEARCH = SignalType("SEARCH", 15)
    CLICK_AD = SignalType("CLICK_AD", 6.0)

    @staticmethod
    def name(value: SignalType) -> str:
        return value.label.upper()

    @classmethod
    def add_signals(cls, signals: List[Dict[str, Any]]) -> None:
        for signal in signals:
            signal_type = SignalType(**signal)
            setattr(cls, signal['label'].upper(), signal_type)

    @classmethod
    def add_new_signals_from_json(cls, json_path: str) -> None:
        with open(json_path, "r") as f:
            signals = json.load(f)
            cls.add_signals(signals)

    @classmethod
    def add_new_signals_from_json_string(cls, json_string: str) -> None:
        signals = json.loads(json_string)
        cls.add_signals(signals)

    @classmethod
    def length(cls):
        return len([attr for attr in dir(cls) if not callable(getattr(cls, attr))])


@dataclass
class SessionObject(DataClassJsonMixin):
    id: str
    is_persistent: bool


@dataclass
class SignalObject:
    action: SignalType
    vector: Vector
    cid: Optional[str]
    timestamp: Optional[int]


class BatchEnum(str, Enum):
    BATCH = "batch"


class BatchType(Enum):
    PERSONALIZED = "personalized"
    BIASED = "biased"
    SAMPLED = "sampled"
    RANDOM = "random"


class UserAction:

    def __init__(self, action: Union[str, SignalType, BatchEnum]):

        self.action_type: Union[SignalType, BatchEnum] = BatchEnum.BATCH

        if isinstance(action, BatchEnum):
            self.action_type = action
        elif isinstance(action, SignalType):
            self.action_type = action
        else:
            self.action_type = self.parse_action(action)

    @staticmethod
    def parse_action(action: str) -> Union[SignalType, BatchEnum]:

        if action != "BATCH":
            return getattr(Signal, action)
        elif action == "BATCH":
            return BatchEnum.BATCH
        else:
            raise ValueError(f"Invalid action: {action}")


@dataclass
class Vertex:
    name: str
    batch_type: BatchType
    params: Params


@dataclass
class Edge:
    name: str
    edge_type: UserAction
    start: Vertex
    end: Vertex


@dataclass
class Blueprint:
    vertices: List[Vertex] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    map: Dict[str, Vertex] = field(default_factory=dict)

    def add_vertex(self, vertex: Vertex):
        self.vertices.append(vertex)
        self.map[vertex.name] = vertex

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def get_operation(self, state: str) -> Vertex:
        return self.map[state]

    def step(self, state: str, action: UserAction) -> Tuple[Vertex, BatchType, Params]:
        """Step function that takes a node name and a UserAction to determine the next vertex."""
        try:
            if state == "0":
                vertex = self.vertices[0]
            else:
                vertex = self.map[state]
        except KeyError:
            raise ValueError(f"No vertex found for state: {state}")

        if not vertex:
            print(f"No vertex found with name: {state}")
            raise ValueError(f"No vertex found with name: {state}")

        edge = next((e for e in self.edges if e.start == vertex and e.edge_type.action_type == action.action_type),
                    None)
        if edge:
            return edge.end, vertex.batch_type, vertex.params
        else:
            edge = next((e for e in self.edges if e.start == vertex and e.edge_type.action_type == Signal.DEFAULT),
                        None)
            if edge:
                return edge.end, vertex.batch_type, vertex.params
            else:
                raise ValueError("No edge found for given conditions")


class DFAParser:
    def __init__(self, data: Union[str, dict]):
        if isinstance(data, str):
            data = json.loads(data)
        self.data = data
        self.blueprint = Blueprint()

    def __validate_edges(self):
        """Validates that each vertex has at least one BatchEnum typed edge and one Signal typed edge.
           Also checks that the signals are covered.
        """
        for node in self.blueprint.vertices:
            related_edges = [e for e in self.blueprint.edges if e.start == node]

            has_batch = any(isinstance(edge.edge_type.action_type, BatchEnum) for edge in related_edges)
            if not has_batch:
                raise ValueError(f"Node {node.name} is missing a BatchEnum typed edge")

            action_types = [edge.edge_type.action_type for edge in related_edges if
                            not isinstance(edge.edge_type.action_type, BatchEnum)]
            # Check if the Signal is covered
            if Signal.DEFAULT not in action_types and len(action_types) != Signal.length():
                raise ValueError(f"Node {node.name} does not have covered signals")

    def parse(self):

        # Parse Signals
        if "signals" in self.data:
            Signal.add_signals(self.data["signals"])

        # Parse vertices
        for node_data in self.data["nodes"]:
            vertex = Vertex(
                name=node_data["name"],
                batch_type=BatchType(node_data["batch_type"]),
                params=Params(**node_data["params"])
            )
            self.blueprint.add_vertex(vertex)

        # Parse edges
        for edge_data in self.data["edges"]:
            edge = Edge(
                name=edge_data["name"],
                edge_type=UserAction(edge_data["edge_type"]),
                start=self.blueprint.map[edge_data["start"]],
                end=self.blueprint.map[edge_data["end"]]
            )
            self.blueprint.add_edge(edge)

        self.__validate_edges()

        return self.blueprint
