import networkx as nx
import pytest
from pyformlang.cfg import CFG

from project.cfpq import hellings_based_cfpq


@pytest.fixture(params=[nx.DiGraph, nx.MultiDiGraph])
def graph(request):
    result = request.param()
    result.add_edge(10, 20, label="a")
    result.add_edge(20, 30, label="a")
    result.add_edge(30, 40, label="b")
    result.add_edge(40, 50, label="b")
    result.add_node(99)
    return result


def test_recursive_grammar_and_empty_paths(graph):
    cfg = CFG.from_text("S -> a S b | epsilon")

    assert hellings_based_cfpq(cfg, graph) == {
        (20, 40),
        (10, 50),
    } | {(node, node) for node in graph}


@pytest.mark.parametrize(
    "starts, finals, expected",
    [
        ({10}, {50}, {(10, 50)}),
        ({10, 20}, {40, 50}, {(10, 50), (20, 40)}),
        ({20}, None, {(20, 20), (20, 40)}),
        (None, {40}, {(20, 40), (40, 40)}),
        ({99}, {99}, {(99, 99)}),
        ({10}, {40}, set()),
        (set(), None, set()),
        (None, set(), set()),
        (set(), set(), set()),
    ],
)
def test_selected_vertices(graph, starts, finals, expected):
    cfg = CFG.from_text("S -> a S b | epsilon")
    assert hellings_based_cfpq(cfg, graph, starts, finals) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", set()),
        ("S -> c", set()),
        ("S -> A\nA -> B\nB -> A | a", {(10, 20), (20, 30)}),
        ("S -> a A b b\nA -> a | epsilon", {(10, 50), (20, 50)}),
        ("S -> epsilon", {(n, n) for n in (10, 20, 30, 40, 50, 99)}),
    ],
)
def test_general_grammars(graph, text, expected):
    assert hellings_based_cfpq(CFG.from_text(text), graph) == expected


def test_empty_graph():
    assert hellings_based_cfpq(CFG.from_text("S -> epsilon"), nx.DiGraph()) == set()


def test_parallel_edges_and_multicharacter_labels():
    graph = nx.MultiDiGraph()
    graph.add_edge(-5, 100, label="hello")
    graph.add_edge(-5, 100, label="hello")
    graph.add_edge(-5, 100, label="a")
    graph.add_edge(100, 200, label="b")
    cfg = CFG.from_text("Root -> hello b | a", start_symbol="Root")

    assert hellings_based_cfpq(cfg, graph) == {(-5, 100), (-5, 200)}


def test_cycles_and_repeated_vertices():
    graph = nx.DiGraph()
    graph.add_edge(0, 0, label="a")
    graph.add_edge(0, 1, label="b")
    graph.add_edge(1, 0, label="b")
    cfg = CFG.from_text("S -> a S b | a b")

    assert hellings_based_cfpq(cfg, graph) == {(0, 0), (0, 1)}


@pytest.mark.parametrize("text", ["S -> S S | a", "S -> a S | a", "S -> S a | a"])
def test_joining_paths_in_both_directions(text):
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)], label="a")

    assert hellings_based_cfpq(CFG.from_text(text), graph) == {
        (source, target) for source in graph for target in graph
    }


@pytest.mark.parametrize("starts, finals", [({1000}, {10}), ({10}, {1000})])
def test_unknown_vertices(graph, starts, finals):
    with pytest.raises(ValueError, match="vertices not in the graph"):
        hellings_based_cfpq(CFG.from_text("S -> a"), graph, starts, finals)


def test_missing_edge_label():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)

    with pytest.raises(ValueError, match="has no label"):
        hellings_based_cfpq(CFG.from_text("S -> a"), graph)
