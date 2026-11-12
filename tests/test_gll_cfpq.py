import networkx as nx
import pytest
from pyformlang.cfg import CFG

from project.gll_cfpq import gll_based_cfpq
from project.rsm import cfg_to_rsm, ebnf_to_rsm


@pytest.fixture(params=[cfg_to_rsm, ebnf_to_rsm])
def convert(request):
    def from_text(text):
        if request.param is cfg_to_rsm:
            return cfg_to_rsm(CFG.from_text(text))
        return ebnf_to_rsm(text)

    return from_text


def test_nullable_result_reused_by_later_callers(convert):
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="b")
    graph.add_edge(0, 2, label="c")
    rsm = convert("S -> A A b | B A c\nB -> A\nA -> epsilon")

    assert gll_based_cfpq(rsm, graph, {0}, {1, 2}) == {(0, 1), (0, 2)}


def test_nonempty_result_reused_by_later_caller(convert):
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="c")
    graph.add_edge(1, 3, label="d")
    rsm = convert("S -> A c | B A d\nA -> a\nB -> C\nC -> D\nD -> epsilon")

    assert gll_based_cfpq(rsm, graph, {0}, {2, 3}) == {(0, 2), (0, 3)}


def test_shared_call_returns_to_each_origin(convert):
    graph = nx.DiGraph()
    graph.add_edge(10, 30, label="a")
    graph.add_edge(20, 30, label="a")
    graph.add_edge(30, 40, label="b")
    graph.add_edge(40, 50, label="b")
    graph.add_edge(50, 60, label="c")
    graph.add_edge(50, 70, label="d")
    rsm = convert("S -> a A c | a A d\nA -> b A | epsilon")

    assert gll_based_cfpq(rsm, graph, {10, 20}, {60, 70}) == {
        (10, 60),
        (10, 70),
        (20, 60),
        (20, 70),
    }
