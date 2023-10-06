import pytest
from firstbatch.algorithm import DFAParser, UserAction, AlgorithmRegistry, AlgorithmLabel
from firstbatch.algorithm.blueprint import Signal
from firstbatch.algorithm.blueprint.library import lookup


@pytest.fixture
def setup_data():
    json_str = '''
    {
        "signals": [
            {"label": "NEW_SIGNAL", "weight": 1.5}
        ],
        "nodes": [
            {"name": "1", "batch_type": "biased", "params": {"mu":0.1}},
            {"name": "2", "batch_type": "random", "params": {"r":0.5}},
            {"name": "3", "batch_type": "sampled", "params": {}}
        ],
        "edges": [
            {"name": "edge5", "edge_type": "NEW_SIGNAL", "start": "3", "end": "3"},
            {"name": "edge1", "edge_type": "LIKE", "start": "1", "end": "2"},
            {"name": "edge2", "edge_type": "BATCH", "start": "1", "end": "3"},
            {"name": "edge3", "edge_type": "DEFAULT", "start": "1", "end": "3"},
            {"name": "edge4", "edge_type": "BATCH", "start": "2", "end": "3"},
            {"name": "edge5", "edge_type": "DEFAULT", "start": "2", "end": "2"},
            {"name": "edge5", "edge_type": "BATCH", "start": "3", "end": "2"},
            {"name": "edge5", "edge_type": "DEFAULT", "start": "3", "end": "2"}
        ]
    }'''

    json_str2 = '''{
        "nodes": [
            {"name": "0", "batch_type": "random", "params": {}},
            {"name": "1", "batch_type": "biased", "params": {"mu":0.0}},
            {"name": "2", "batch_type": "biased", "params": {"mu":0.5}},
            {"name": "3", "batch_type": "biased", "params": {"mu":1.0}}
        ],
        "edges": [
            {"name": "edge1", "edge_type": "BATCH", "start": "0", "end": "0"},
            {"name": "edge2", "edge_type": "DEFAULT", "start": "0", "end": "1"},
            {"name": "edge3", "edge_type": "DEFAULT", "start": "1", "end": "1"},
            {"name": "edge4", "edge_type": "BATCH", "start": "1", "end": "2"},
            {"name": "edge5", "edge_type": "DEFAULT", "start": "2", "end": "1"},
            {"name": "edge5", "edge_type": "BATCH", "start": "2", "end": "3"},
            {"name": "edge5", "edge_type": "BATCH", "start": "3", "end": "0"},
            {"name": "edge5", "edge_type": "DEFAULT", "start": "3", "end": "1"}
        ]
    }'''

    d = '''{
                "nodes": [
                    {"name": "Initial_State", "batch_type": "random", "params": {}},
                    {"name": "Personalized_Recommendation", "batch_type": "biased", "params": {"mu": 0.8, "alpha": 0.7, "apply_mmr": 1, "last_n": 5}},
                    {"name": "Personalized_Exploratory", "batch_type": "biased", "params": {"mu": 0.6, "alpha": 0.5, "apply_mmr": 1, "last_n": 5, "r": 0.1}}
                ],
                "edges": [
                    {"name": "edge1", "edge_type": "DEFAULT", "start": "Initial_State", "end": "Personalized_Recommendation"},
                    {"name": "edge2", "edge_type": "DEFAULT", "start": "Personalized_Recommendation", "end": "Personalized_Exploratory"},
                    {"name": "edge3", "edge_type": "DEFAULT", "start": "Personalized_Exploratory", "end": "Initial_State"},
                    {"name": "edge4", "edge_type": "BATCH", "end": "Initial_State", "start": "Initial_State"},
                    {"name": "edge5", "edge_type": "BATCH", "end": "Personalized_Recommendation", "start": "Initial_State"},
                    {"name": "edge6", "edge_type": "BATCH", "end": "Personalized_Exploratory", "start": "Personalized_Recommendation"}
                ]
            }'''
    return json_str, json_str2, d


def test_factory():
    for k, v in lookup.items():
        parser = DFAParser(v)
        try:
            blueprint = parser.parse()
        except Exception as e:
            pytest.fail(f"{e} error with {k}")


def test_signal(setup_data):
    algo_type = AlgorithmRegistry.get_algorithm_by_label(AlgorithmLabel.UNIQUE_JOURNEYS)
    algo_instance = algo_type(factory_id, batch_size, **{"embedding_size": embedding_size})  # type: ignore

    blueprint = algo_instance._blueprint

    assert len(blueprint.vertices) == 7
    assert len(blueprint.edges) == 46

    current_vertex = 'Exploration'
    action = UserAction(Signal.REPOST)
    next_vertex, _, _ = blueprint.step(current_vertex, action)
    assert next_vertex.name == 'Dedicated_2'
