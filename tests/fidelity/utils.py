from typing import Any, Dict, List, Optional, Tuple

from fidelity.utils import SubGraph


def build_tx(
    edges: Optional[List[Tuple[str, str]]] = None, with_issue: bool = False
) -> SubGraph:
    edges = edges or [("1", "2")]
    nodes_data = {}
    for e in edges:
        nodes_data[e[0]] = {"id": e[0], "resource": {"name": e[0]}, "issues": with_issue}
        nodes_data[e[1]] = {"id": e[1], "resource": {"name": e[1]}}
    return SubGraph(
        nodes=nodes_data,
        edges=edges,
    )
