import networkx as nx
import pytest
from pyformlang.cfg import CFG

from project.cfpq import matrix_based_cfpq


@pytest.fixture
def graph():
    result = nx.DiGraph()
    result.add_edge(10, 20, label="a")
    result.add_edge(20, 30, label="a")
    result.add_edge(30, 40, label="b")
    result.add_edge(40, 50, label="b")
    result.add_node(99)
    return result


def test_recursive_grammar(graph):
    cfg = CFG.from_text("S -> a S b | epsilon")

    assert matrix_based_cfpq(cfg, graph) == {
        (10, 50),
        (20, 40),
    } | {(node, node) for node in graph}


def test_selected_vertices(graph):
    cfg = CFG.from_text("S -> a S b | epsilon")

    assert matrix_based_cfpq(cfg, graph, {10}, {50}) == {(10, 50)}
    assert matrix_based_cfpq(cfg, graph, set(), None) == set()
    assert matrix_based_cfpq(cfg, graph, None, set()) == set()


def test_recursive_updates_reach_dependent_rules():
    graph = nx.DiGraph()
    for source, label in enumerate("abccc"):
        graph.add_edge(source, source + 1, label=label)
    cfg = CFG.from_text("S -> A B\nA -> a\nB -> B C | b\nC -> c")

    assert matrix_based_cfpq(cfg, graph) == {(0, 2), (0, 3), (0, 4), (0, 5)}
