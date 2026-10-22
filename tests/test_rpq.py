import networkx as nx
import pytest

from project.rpq import ms_bfs_based_rpq, tensor_based_rpq


@pytest.fixture(params=[tensor_based_rpq, ms_bfs_based_rpq])
def rpq(request):
    return request.param


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
def test_all_pairs(rpq, graph, regex, expected):
    assert rpq(regex, graph, set(graph), set(graph)) == expected


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
def test_selected_vertices(rpq, graph, starts, finals, expected):
    assert rpq("a b", graph, starts, finals) == expected


def test_empty_graph(rpq):
    assert rpq("a*", nx.MultiDiGraph(), set(), set()) == set()


def test_isolated_vertex_and_empty_word(rpq, graph):
    assert rpq("a*", graph, {99}, {99}) == {(99, 99)}
    assert rpq("a", graph, {99}, {99}) == set()
    assert rpq("epsilon", graph, {10, 20}, {20, 30}) == {(20, 20)}


def test_multicharacter_label(rpq):
    graph = nx.MultiDiGraph()
    graph.add_edge(-5, 100, label="hello")

    assert rpq("hello", graph, {-5}, {100}) == {(-5, 100)}
    assert rpq("h e l l o", graph, {-5}, {100}) == set()


@pytest.mark.parametrize("starts, finals", [({1000}, {10}), ({10}, {1000})])
def test_unknown_vertices(rpq, graph, starts, finals):
    with pytest.raises(ValueError, match="vertices not in the graph"):
        rpq("a", graph, starts, finals)


def test_multiple_sources_keep_separate_answers(rpq):
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 30, label="a")
    graph.add_edge(20, 30, label="a")
    graph.add_edge(30, 40, label="b")
    graph.add_edge(20, 50, label="a")
    graph.add_edge(50, 60, label="b")

    assert rpq("a b", graph, {10, 20}, {40, 60}) == {
        (10, 40),
        (20, 40),
        (20, 60),
    }


def test_revisit_vertex_in_different_regex_states(rpq):
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 10, label="a")
    graph.add_edge(10, 20, label="b")

    assert rpq("a a b", graph, {10, 20}, {20}) == {(10, 20)}


def test_final_vertices_do_not_stop_traversal(rpq):
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")

    assert rpq("a | a b", graph, {10}, {20, 30}) == {(10, 20), (10, 30)}
