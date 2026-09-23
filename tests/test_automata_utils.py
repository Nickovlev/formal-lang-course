import cfpq_data
import networkx as nx
import pytest
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    Symbol,
)
from pyformlang.regular_expression import MisformedRegexError

from project.automata_utils import graph_to_nfa, regex_to_dfa


@pytest.mark.parametrize(
    "regex, state_count, accepted, rejected",
    [
        ("", 1, [], [[], ["a"]]),
        ("epsilon", 1, [[]], [["a"]]),
        ("$", 1, [[]], [["a"]]),
        ("a | a", 2, [["a"]], [[], ["b"], ["a", "a"]]),
        ("a b | a c", 3, [["a", "b"], ["a", "c"]], [[], ["a"], ["b"]]),
        ("(a | b)*", 1, [[], ["a", "b", "a"]], [["c"]]),
        (
            "first . (second + $)",
            3,
            [["first"], ["first", "second"]],
            [[], ["second"], list("first")],
        ),
    ],
)
def test_regex_to_dfa(regex, state_count, accepted, rejected) -> None:
    dfa = regex_to_dfa(regex)

    assert isinstance(dfa, DeterministicFiniteAutomaton)
    assert dfa.is_deterministic()
    assert len(dfa.states) == state_count
    for word in accepted:
        assert dfa.accepts(word)
    for word in rejected:
        assert not dfa.accepts(word)


def test_invalid_regex() -> None:
    with pytest.raises(MisformedRegexError):
        regex_to_dfa("a | *")


def test_graph_structure_and_language() -> None:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 0, label="a")
    graph.add_edge(0, 1, label="a")

    nfa = graph_to_nfa(graph, {0}, {1})

    assert isinstance(nfa, NondeterministicFiniteAutomaton)
    assert not nfa.is_deterministic()
    assert nfa.states == {0, 1, 2}
    assert nfa.start_states == {0}
    assert nfa.final_states == {1}
    assert nfa(0, "a") == {0, 1}
    assert nfa.accepts(["a"])
    assert nfa.accepts(["a", "a"])
    assert not nfa.accepts([])
    assert not nfa.accepts(["b"])


@pytest.mark.parametrize("start", [None, set(), {1}])
@pytest.mark.parametrize("final", [None, set(), {2}])
def test_start_and_final_defaults(start, final) -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    graph.add_node(3)

    nfa = graph_to_nfa(graph, start, final)

    assert nfa.start_states == (start or {1, 2, 3})
    assert nfa.final_states == (final or {1, 2, 3})
    assert nfa.accepts([]) == bool(nfa.start_states & nfa.final_states)
    assert nfa.accepts(["a"])


def test_empty_graph() -> None:
    nfa = graph_to_nfa(nx.MultiDiGraph())

    assert not nfa.states
    assert nfa.is_empty()
    assert not nfa.accepts([])


def test_isolated_vertex() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(7)

    nfa = graph_to_nfa(graph)

    assert nfa.states == nfa.start_states == nfa.final_states == {7}
    assert nfa.accepts([])
    assert not nfa.accepts(["a"])


@pytest.mark.parametrize("parameter", ["start_states", "final_states"])
def test_unknown_vertices(parameter) -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(0)

    with pytest.raises(
        ValueError, match=parameter + " contains vertices not in the graph"
    ):
        graph_to_nfa(graph, **{parameter: {1}})


@pytest.mark.parametrize("data", [{}, {"label": None}])
def test_missing_label(data) -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, **data)

    with pytest.raises(ValueError, match="has no label"):
        graph_to_nfa(graph)


@pytest.mark.parametrize("label", ["epsilon", "$", "a b", "a|b"])
def test_labels_are_literal_symbols(label) -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label=label)

    nfa = graph_to_nfa(graph, {0}, {1})

    assert nfa.accepts([Symbol(label)])
    assert not nfa.accepts([])


def test_cfpq_two_cycles() -> None:
    graph = cfpq_data.labeled_two_cycles_graph(2, 3, labels=("first", "second"))

    nfa = graph_to_nfa(graph, {0}, {0})

    assert nfa.states == set(graph.nodes)
    assert nfa.accepts(["first"] * 3)
    assert nfa.accepts(["second"] * 4)
    assert not nfa.accepts(["first"])
