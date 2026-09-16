from unittest.mock import patch

import networkx
import pydot
import pytest

from project.graph_utils import GraphInfo, create_two_cycles_graph, get_graph_info


def test_get_graph_info() -> None:
    graph = networkx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 0, label="a")

    with (
        patch(
            "project.graph_utils.cfpq_data.download", return_value="graph.csv"
        ) as download,
        patch(
            "project.graph_utils.cfpq_data.graph_from_csv", return_value=graph
        ) as load,
    ):
        graph_info = get_graph_info("test_graph")

    assert isinstance(graph_info, GraphInfo)
    assert graph_info.number_of_vertices == 3
    assert graph_info.number_of_edges == 3
    assert graph_info.labels == {"a", "b"}
    assert tuple(graph_info) == (3, 3, {"a", "b"})

    download.assert_called_once_with("test_graph")
    load.assert_called_once_with("graph.csv")


def test_get_graph_info_without_labeled_edges() -> None:
    graph = networkx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1)

    with (
        patch("project.graph_utils.cfpq_data.download", return_value="graph.csv"),
        patch("project.graph_utils.cfpq_data.graph_from_csv", return_value=graph),
    ):
        assert get_graph_info("test_graph") == (3, 1, set())


def test_get_graph_info_propagates_download_error() -> None:
    with (
        patch(
            "project.graph_utils.cfpq_data.download",
            side_effect=FileNotFoundError("unknown graph"),
        ),
        patch("project.graph_utils.cfpq_data.graph_from_csv") as load,
        pytest.raises(FileNotFoundError),
    ):
        get_graph_info("unknown_graph")

    load.assert_not_called()


def test_create_two_cycles_graph(tmp_path) -> None:
    output_path = tmp_path / "two_cycles.dot"

    create_two_cycles_graph(2, 3, "first", "second", output_path)

    dot_graphs = pydot.graph_from_dot_file(output_path)
    assert len(dot_graphs) == 1

    dot_graph = dot_graphs[0]
    assert len(dot_graph.get_nodes()) == 6
    assert len(dot_graph.get_edges()) == 7
    assert {edge.get_label().strip('"') for edge in dot_graph.get_edges()} == {
        "first",
        "second",
    }


def test_create_two_cycles_graph_preserves_file_on_invalid_size(tmp_path) -> None:
    output_path = tmp_path / "existing.dot"
    original_content = "digraph { original; }\n"
    output_path.write_text(original_content)

    with pytest.raises(networkx.NetworkXError):
        create_two_cycles_graph(-1, 2, "first", "second", output_path)

    assert output_path.read_text() == original_content
