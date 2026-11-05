from copy import deepcopy

import networkx as nx
import pytest
from pyformlang.cfg import CFG

from project.rsm import cfg_to_rsm, ebnf_to_rsm
from project.tensor_cfpq import tensor_based_cfpq


@pytest.fixture(params=[cfg_to_rsm, ebnf_to_rsm])
def convert(request):
    def from_text(text):
        if request.param is cfg_to_rsm:
            return cfg_to_rsm(CFG.from_text(text))
        return ebnf_to_rsm(text)

    return from_text


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_nullable_calls_and_filters_do_not_restrict_internal_paths(convert, graph_type):
    graph = graph_type()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")
    graph.add_node(99)
    rsm = convert("S -> a A b\nA -> B\nB -> epsilon")

    assert tensor_based_cfpq(rsm, graph, {10}, {30}) == {(10, 30)}
    assert tensor_based_cfpq(rsm, graph, set(), None) == set()
    assert tensor_based_cfpq(rsm, graph, None, set()) == set()


def test_mutual_recursion_reaches_fixed_point(convert):
    graph = nx.DiGraph()
    for source in range(4):
        graph.add_edge(source, source + 1, label="a")
    rsm = convert("S -> A\nA -> B a | a\nB -> S")

    assert tensor_based_cfpq(rsm, graph) == {
        (source, target) for source in range(5) for target in range(source + 1, 5)
    }


@pytest.mark.parametrize("text", ["", "S -> S", "S -> epsilon"])
def test_empty_graph(convert, text):
    assert tensor_based_cfpq(convert(text), nx.DiGraph()) == set()


@pytest.mark.parametrize(
    "text, expected",
    [("", set()), ("S -> S", set()), ("S -> epsilon", {(99, 99)})],
)
def test_isolated_vertex(convert, text, expected):
    graph = nx.DiGraph()
    graph.add_node(99)

    assert tensor_based_cfpq(convert(text), graph) == expected


def test_parallel_edges_loops_and_repeated_calls_preserve_inputs(convert):
    graph = nx.MultiDiGraph()
    graph.add_edge(-5, -5, label="hello")
    graph.add_edge(-5, 100, label="hello")
    graph.add_edge(-5, 100, label="hello")
    graph.add_edge(-5, 100, label="other")
    rsm = convert("S -> hello S | epsilon")
    original_graph = deepcopy(graph)
    original_boxes = {label: box.dfa.copy() for label, box in rsm.boxes.items()}

    for _ in range(2):
        assert tensor_based_cfpq(rsm, graph) == {(-5, -5), (-5, 100), (100, 100)}
    assert nx.utils.graphs_equal(graph, original_graph)
    for label, automaton in original_boxes.items():
        assert rsm.boxes[label].dfa.is_equivalent_to(automaton)


def test_custom_start_symbol_and_undefined_variable():
    graph = nx.DiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="Unused")
    cfg = CFG.from_text("Root -> a | Unused", start_symbol="Root")

    assert tensor_based_cfpq(cfg_to_rsm(cfg), graph) == {(10, 20)}


@pytest.mark.parametrize("starts, finals", [({1000}, {10}), ({10}, {1000})])
def test_unknown_vertices(starts, finals):
    graph = nx.DiGraph()
    graph.add_node(10)

    with pytest.raises(ValueError, match="vertices not in the graph"):
        tensor_based_cfpq(ebnf_to_rsm("S -> a"), graph, starts, finals)
