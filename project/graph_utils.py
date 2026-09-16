from pathlib import Path
from typing import NamedTuple

import cfpq_data
from networkx.drawing.nx_pydot import to_pydot


class GraphInfo(NamedTuple):
    number_of_vertices: int
    number_of_edges: int
    labels: set[str]


def get_graph_info(graph_name: str) -> GraphInfo:
    graph_path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_path)
    labels = {label for _, _, label in graph.edges(data="label") if label is not None}

    return GraphInfo(graph.number_of_nodes(), graph.number_of_edges(), labels)


def create_two_cycles_graph(
    first_cycle_nodes: int,
    second_cycle_nodes: int,
    first_label: str,
    second_label: str,
    output_path: str | Path,
) -> None:
    graph = cfpq_data.labeled_two_cycles_graph(
        first_cycle_nodes,
        second_cycle_nodes,
        labels=(first_label, second_label),
    )
    to_pydot(graph).write_raw(str(output_path))
