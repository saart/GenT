import json
import os
import random
from collections import defaultdict
from functools import cached_property
from typing import Dict, Generator, NamedTuple, Tuple, List, Optional

import dataclasses
import scipy

import paper
from ml.app_normalizer import get_name

GRACE_TRIGGER_TIME = 4000

TRANSACTION_RAW_DIR = os.path.join(os.path.dirname(paper.__path__[0]), "..", "traces")

RAW_LIKE_GENERATED = True
FIX_MALFORMED_TX = True

StructureId = str
NodeId = str

duration, times = 0, 0


@dataclasses.dataclass
class SubGraph:
    nodes: Dict[str, dict]
    edges: List[Tuple[str, str]]

    @cached_property
    def has_issues(self):
        return any(node.get("issues") for node in self.nodes.values())

    @cached_property
    def timestamp(self):
        return min(n["startTime"] for n in self.nodes.values())


@dataclasses.dataclass
class ComparableTree:
    appearances: int
    graph: str
    appearances_with_issues: int
    graphs: List[SubGraph]

    @staticmethod
    def build(count: int, graph: SubGraph) -> "ComparableTree":
        issues = int(graph.has_issues)

        return ComparableTree(count, tree_to_structure_id(edges=graph.edges), issues, [graph])

    def update(self, graph: SubGraph) -> "ComparableTree":
        self.appearances += 1
        self.appearances_with_issues += int(graph.has_issues)
        self.graphs.append(graph)
        return self


def get_subgraphs(transaction: dict, subtree_height: int) -> List[SubGraph]:
    """
    To the case where the DAG structure could be different than the raw data
    """
    nodes: Dict[str, dict] = transaction["nodesData"]
    parent_to_children: Dict[str, List[dict]] = defaultdict(list)
    for edge in transaction["graph"]["edges"]:
        parent_to_children[edge["source"]].append(nodes[edge["target"]])

    subgraphs: List[SubGraph] = []
    for subtree_root_id, subtree_root in nodes.items():
        subtree_nodes = {subtree_root_id: subtree_root}
        subtree_edges = []
        current_level = [subtree_root]
        for _ in range(subtree_height):
            next_level = []
            for node in current_level:
                for child in parent_to_children[node["id"]]:
                    subtree_nodes[child["id"]] = child
                    next_level.append(child)
                    subtree_edges.append((get_name(nodes, node["id"]), get_name(nodes, child["id"])))
            current_level = next_level
        if subtree_edges:
            # We care only from subgraphs with edges (not from leafs)
            subgraphs.append(SubGraph(subtree_nodes, subtree_edges))
    return subgraphs


class ComparableForest(NamedTuple):
    forest: Dict[StructureId, ComparableTree]

    def add_tree(self, transaction: dict, max_trees_to_add: int, subtree_height: int = 100) -> int:
        """
        Only to the case that the DAG structure is different than the raw data
        """
        number_of_subgraphs = 0
        for subgraph, _ in zip(get_subgraphs(transaction, subtree_height), range(max_trees_to_add)):
            structure_id = tree_to_structure_id(edges=subgraph.edges)
            if structure_id in self.forest:
                self.forest[structure_id] = self.forest[structure_id].update(subgraph)
            else:
                self.forest[structure_id] = ComparableTree.build(1, subgraph)
            number_of_subgraphs += 1
        return number_of_subgraphs

    def size(self) -> int:
        return sum(t.appearances for t in self.forest.values())


def tree_to_structure_id(graph: Optional[SubGraph] = None, edges: Optional[List[Tuple[str, str]]] = None) -> StructureId:
    edges = edges or graph.edges
    edges = {f"{e[0]}-{e[1]}" for e in edges}
    return json.dumps(sorted(edges))


def build_comparable_forest_of_subgraphs(
    trees_generator: Generator[dict, None, None], max_trees: int = 2 ** 32, subtree_height: int = 100
):
    total_trees = 0
    forest = ComparableForest({})
    for graph in trees_generator:
        if max_trees and total_trees >= max_trees:
            return forest
        total_trees += forest.add_tree(graph, subtree_height=subtree_height, max_trees_to_add=max_trees - total_trees)
    return forest


def build_comparable_forest(
    trees_generator: Generator[dict, None, None], max_trees: int = 2 ** 32,
    other_forest: Optional[ComparableForest] = None,
) -> ComparableForest:
    """
    This function also provide guarantee that the output forest will look exactly like the given other_forest.
    I.e. if other_forest contains only subset of the trees that trees_generator contains,
        then we will drop the extra trees (and put them in dropped_txs).
    """
    total_trees = 0
    forest = ComparableForest({})
    dropped_txs = defaultdict(int)
    for transaction in trees_generator:
        name = lambda node_id: get_name(transaction["nodesData"], node_id)
        graph = SubGraph(
            {name(k): v for k, v in transaction["nodesData"].items()},
            [(name(e["source"]), name(e["target"])) for e in transaction["graph"]["edges"]]
        )
        structure_id = tree_to_structure_id(graph)
        if other_forest and structure_id not in other_forest.forest:
            if set(other_forest.forest) == set(forest.forest):
                if all(
                    other_forest.forest[structure_id].appearances == forest.forest[structure_id].appearances
                    for structure_id in other_forest.forest
                ):
                    break
            # raise Exception("We could not generate raw data that looks exactly like the given generated data")
            dropped_txs[structure_id] += 1
            continue
        if structure_id in forest.forest:
            if other_forest and other_forest.forest[structure_id].appearances == forest.forest[structure_id].appearances:
                # We don't generate all the occurrences for every structure
                continue

        if structure_id in forest.forest:
            forest.forest[structure_id] = forest.forest[structure_id].update(graph)
        else:
            forest.forest[structure_id] = ComparableTree.build(1, graph)

        total_trees += 1
        if (not other_forest) and max_trees and total_trees >= max_trees:
            break
    if other_forest:
        if set(other_forest.forest) != set(forest.forest) or not all(
            other_forest.forest[structure_id].appearances == forest.forest[structure_id].appearances
            for structure_id in other_forest.forest
        ):
            diff = list(set(other_forest.forest).difference(set(forest.forest)))
            if len(diff) == 1 and other_forest.forest[diff[0]].appearances == 1:
                print("A tree didn't appear in the generated forest but did in the raw")
                other_forest.forest.pop(diff[0])
            else:
                raise Exception("We could not find raw data that looks exactly like the given generated data")
    if other_forest and dropped_txs:
        print(f"Graphs that didn't appear in the generated forest but did in the raw forest: {dict(dropped_txs)}")
    return forest


def validate_transaction(transaction: dict) -> bool:
    """
    This function validates the given graph with the conditions:
    1. duration is a positive number
    2. child should start after the parent
    returns True if the graph is valid, False otherwise
    """
    global duration, times
    for node in transaction["nodesData"].values():
        if node["duration"] < 0:
            duration += 1
            if FIX_MALFORMED_TX:
                node["duration"] = 0
            else:
                return False
    node_to_childs = defaultdict(set)
    for edge in transaction["graph"]["edges"]:
        if transaction["nodesData"][edge["source"]]["startTime"] > transaction["nodesData"][edge["target"]]["startTime"] + GRACE_TRIGGER_TIME:
            times += 1
            if FIX_MALFORMED_TX:
                transaction["nodesData"][edge["target"]]["startTime"] = transaction["nodesData"][edge["source"]]["startTime"] + GRACE_TRIGGER_TIME
            else:
                return False
        node_to_childs[edge["source"]].add(edge["target"])

    not_roots = set()
    [not_roots.update(children) for children in node_to_childs.values()]

    return True


def get_transactions_graphs_generator(files_dir: str, sample_rate: float = 1.) -> Generator[dict, None, None]:
    global duration, times
    duration, times = 0, 0
    valid, invalid = 0, 0
    graphs = []
    for file in os.listdir(files_dir):
        with open(os.path.join(files_dir, file), "r") as f:
            for graph_line in f.readlines():
                graph_line = graph_line.strip("\n,")
                if graph_line:
                    graph = json.loads(graph_line)
                    if validate_transaction(graph):
                        valid += 1
                        graphs.append(graph)
                    else:
                        invalid += 1
    graphs = random.sample(graphs, int(len(graphs) * sample_rate))
    graphs = sorted(graphs, key=lambda graph: graph["details"]["startTime"])
    for graph in graphs:
        yield graph
    if invalid:
        print(f"Found {valid} valid and {invalid} invalid graphs: {duration} invalid duration, {times} invalid times.")


def load_data(
    generated_dir: str, raw_dir: str = TRANSACTION_RAW_DIR,
        raw_like_generated: bool = RAW_LIKE_GENERATED, max_trees: int = 2 ** 32
) -> Tuple[ComparableForest, ComparableForest]:
    generated_forest = build_comparable_forest(get_transactions_graphs_generator(generated_dir), max_trees=max_trees)
    other_forest = generated_forest if raw_like_generated else None
    raw_forest = build_comparable_forest(
        get_transactions_graphs_generator(raw_dir), other_forest=other_forest, max_trees=max_trees
    )
    print(
        f"Found {raw_forest.size()} trees: "
        f"{len(raw_forest.forest)} structures in raw and {len(generated_forest.forest)} in generated."
    )
    return raw_forest, generated_forest


def compare_distributions(x: List[float], y: List[float]) -> float:
    # if len(x) != len(y):
    #     print("Note: comparing distributions with different lengths")
    #     length = min(len(x), len(y))
    #     x, y = x[:length], y[:length]
    return scipy.stats.wasserstein_distance(x, y)


min_time = 0
bulk_size = 0


def set_timespan(raw_forest: ComparableForest, generated_forest: ComparableForest, number_of_bulks: int) -> None:
    timestamps = [g.timestamp
                  for forest in [raw_forest, generated_forest]
                  for tree in forest.forest.values()
                  for g in tree.graphs]
    global bulk_size, min_time
    max_time = max(timestamps)
    min_time = min(timestamps)
    total_time = max_time - min_time
    bulk_size = total_time // number_of_bulks


def time_to_bulk_index(time: float) -> int:
    return int((time - min_time) // bulk_size)
