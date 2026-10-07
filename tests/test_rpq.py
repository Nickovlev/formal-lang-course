import networkx as nx
import pytest

from project.rpq import tensor_based_rpq


@pytest.fixture
def graph():
    result = nx.MultiDiGraph()
    result.add_edge(10, 20, label="a")
    result.add_edge(10, 20, label="a")
    result.add_edge(10, 20, label="b")
    result.add_edge(20, 30, label="b")
    result.add_edge(30, 10, label="a")
    result.add_node(99)
    return result


@pytest.mark.parametrize(
    "regex, expected",
    [
        ("", set()),
        ("epsilon", {(10, 10), (20, 20), (30, 30), (99, 99)}),
        ("a b", {(10, 30), (30, 20)}),
        ("b", {(10, 20), (20, 30)}),
        ("c", set()),
        ("a*", {(10, 10), (20, 20), (30, 30), (99, 99), (10, 20), (30, 10), (30, 20)}),
        ("(a | b)*", {(u, v) for u in (10, 20, 30) for v in (10, 20, 30)} | {(99, 99)}),
    ],
)
def test_all_pairs(graph, regex, expected):
    assert tensor_based_rpq(regex, graph, set(graph), set(graph)) == expected


@pytest.mark.parametrize(
    "starts, finals, expected",
    [
        ({10}, {30}, {(10, 30)}),
        ({30}, {20}, {(30, 20)}),
        ({10}, {20}, set()),
        (set(), {30}, set()),
        ({10}, set(), set()),
        (set(), set(), set()),
    ],
)
def test_selected_vertices(graph, starts, finals, expected):
    assert tensor_based_rpq("a b", graph, starts, finals) == expected


def test_empty_graph():
    assert tensor_based_rpq("a*", nx.MultiDiGraph(), set(), set()) == set()


def test_isolated_vertex_and_empty_word(graph):
    assert tensor_based_rpq("a*", graph, {99}, {99}) == {(99, 99)}
    assert tensor_based_rpq("a", graph, {99}, {99}) == set()
    assert tensor_based_rpq("epsilon", graph, {10, 20}, {20, 30}) == {(20, 20)}


def test_multicharacter_label():
    graph = nx.MultiDiGraph()
    graph.add_edge(-5, 100, label="hello")

    assert tensor_based_rpq("hello", graph, {-5}, {100}) == {(-5, 100)}
    assert tensor_based_rpq("h e l l o", graph, {-5}, {100}) == set()


@pytest.mark.parametrize("starts, finals", [({1000}, {10}), ({10}, {1000})])
def test_unknown_vertices(graph, starts, finals):
    with pytest.raises(ValueError, match="vertices not in the graph"):
        tensor_based_rpq("a", graph, starts, finals)
