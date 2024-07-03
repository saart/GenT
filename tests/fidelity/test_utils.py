import json

from fidelity.compare_forests import ComparableTree, load_data
from fidelity.utils import tree_to_structure_id, SubGraph
from tests.fidelity.utils import build_tx


def test_load_data(tmp_path):
    tx = build_tx(edges=[("1", "2")])
    raw_tx = {
        "graph": {
            "edges": [{"source": "1", "target": "2"}]
        },
        "nodesData": {
            "1": {"id": "1", "resource": {"name": "1"}},
            "2": {"id": "2", "resource": {"name": "2"}},
        }
    }
    (tmp_path / "file1.txt").write_text(json.dumps(raw_tx) + "\n" + json.dumps(raw_tx))

    forest, _ = load_data(str(tmp_path), str(tmp_path))

    assert forest.forest == {
        tree_to_structure_id(tx): ComparableTree.build(2, build_tx(edges=[("1", "2")]))
    }


def test_load_data_split_by_height(tmp_path):
    raw_tx = {
        "graph": {
            "edges": [
                {"source": "1", "target": "2"},
                {"source": "2", "target": "3"},
                {"source": "3", "target": "4"},
            ]
        },
        "nodesData": {
            "1": {"id": "1", "resource": {"name": "1"}},
            "2": {"id": "2", "resource": {"name": "2"}},
            "3": {"id": "3", "resource": {"name": "3"}},
            "4": {"id": "4", "resource": {"name": "4"}},
        }
    }
    (tmp_path / "file1.txt").write_text(json.dumps(raw_tx) + "\n" + json.dumps(raw_tx))

    forest, _ = load_data(str(tmp_path), str(tmp_path), subtree_height=2)

    subtree_1 = build_tx(edges=[("1", "2"), ("2", "3")])
    subtree_2 = build_tx(edges=[("2", "3"), ("3", "4")])
    subtree_3 = build_tx(edges=[("3", "4")])
    assert forest.forest == {
        tree_to_structure_id(subtree_1): ComparableTree.build(2, subtree_1),
        tree_to_structure_id(subtree_2): ComparableTree.build(2, subtree_2),
        tree_to_structure_id(subtree_3): ComparableTree.build(2, subtree_3),
    }

    forest, _ = load_data(str(tmp_path), str(tmp_path), subtree_height=3)

    subtree_1 = build_tx(edges=[("1", "2"), ("2", "3"), ("3", "4")])
    assert forest.forest == {
        tree_to_structure_id(subtree_1): ComparableTree.build(2, subtree_1),
        tree_to_structure_id(subtree_2): ComparableTree.build(2, subtree_2),
        tree_to_structure_id(subtree_3): ComparableTree.build(2, subtree_3),
    }
