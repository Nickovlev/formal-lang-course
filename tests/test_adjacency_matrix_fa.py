from itertools import product

import pytest
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from scipy import sparse

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.automata_utils import regex_to_dfa


@pytest.mark.parametrize("regex", ["", "epsilon", "a", "a*", "a b | b a", "(a | b)*"])
def test_dfa_language(regex):
    dfa = regex_to_dfa(regex)
    matrix_fa = AdjacencyMatrixFA(dfa)

    assert matrix_fa.is_empty() == dfa.is_empty()
    for length in range(5):
        for word in product("ab", repeat=length):
            assert matrix_fa.accepts(map(Symbol, word)) == dfa.accepts(word)
    assert not matrix_fa.accepts([Symbol("unknown")])


def test_nfa_structure_and_branching():
    nfa = NondeterministicFiniteAutomaton(
        states={"start", "other", "middle", "end", "isolated"},
        start_state={"start", "other"},
        final_states={"end", "isolated"},
    )
    nfa.add_transition("start", "a", "start")
    nfa.add_transition("start", "a", "middle")
    nfa.add_transition("middle", "b", "end")
    nfa.add_transition("other", "b", "end")
    matrix_fa = AdjacencyMatrixFA(nfa)

    assert set(matrix_fa.states) == nfa.states
    assert {matrix_fa.states[i] for i in matrix_fa.start_states} == nfa.start_states
    assert {matrix_fa.states[i] for i in matrix_fa.final_states} == nfa.final_states
    for symbol, matrix in matrix_fa.matrices.items():
        assert sparse.isspmatrix_csr(matrix)
        assert matrix.dtype == bool
        assert matrix.shape == (5, 5)
        for source in nfa.states:
            for target in nfa.states:
                assert bool(
                    matrix[
                        matrix_fa.state_to_index[source],
                        matrix_fa.state_to_index[target],
                    ]
                ) == (target in nfa(source, symbol))
    for length in range(5):
        for word in product("ab", repeat=length):
            assert matrix_fa.accepts(iter(word)) == nfa.accepts(word)
    assert not matrix_fa.is_empty()


@pytest.mark.parametrize(
    "starts, finals, edges, empty",
    [
        (set(), {0}, [], True),
        ({0}, set(), [], True),
        ({0}, {0}, [], False),
        ({0}, {2}, [(0, 1), (1, 0)], True),
        ({0}, {3}, [(0, 1), (1, 2), (2, 3)], False),
    ],
)
def test_emptiness(starts, finals, edges, empty):
    nfa = NondeterministicFiniteAutomaton(
        states={0, 1, 2, 3}, start_state=starts, final_states=finals
    )
    for source, target in edges:
        nfa.add_transition(source, "a", target)
    matrix_fa = AdjacencyMatrixFA(nfa)

    assert matrix_fa.is_empty() is empty
    assert matrix_fa.accepts([]) == bool(starts & finals)


def test_no_states():
    matrix_fa = AdjacencyMatrixFA(NondeterministicFiniteAutomaton())

    assert matrix_fa.transitive_closure().shape == (0, 0)
    assert matrix_fa.is_empty()
    assert not matrix_fa.accepts([])
    assert not matrix_fa.accepts([Symbol("a")])
    intersection = intersect_automata(matrix_fa, AdjacencyMatrixFA(regex_to_dfa("a*")))
    assert intersection.is_empty()
    assert not intersection.accepts([])


@pytest.mark.parametrize(
    "first, second",
    [
        ("a*", "b*"),
        ("a", "b"),
        ("", "a*"),
        ("epsilon", "a*"),
        ("a b | b a", "a (a | b)*"),
        ("(a | b)*", "(a b)*"),
    ],
)
def test_intersection_language(first, second):
    dfa1, dfa2 = regex_to_dfa(first), regex_to_dfa(second)
    matrix1, matrix2 = AdjacencyMatrixFA(dfa1), AdjacencyMatrixFA(dfa2)
    intersection = intersect_automata(matrix1, matrix2)

    assert intersection.is_empty() == dfa1.get_intersection(dfa2).is_empty()
    for length in range(5):
        for word in product("ab", repeat=length):
            assert intersection.accepts(word) == (
                dfa1.accepts(word) and dfa2.accepts(word)
            )
    assert len(intersection.states) == len(dfa1.states) * len(dfa2.states)
    for matrix in intersection.matrices.values():
        assert sparse.isspmatrix_csr(matrix)
        assert matrix.dtype == bool


def test_intersection_with_nfa():
    nfa = NondeterministicFiniteAutomaton(start_state={10, 20}, final_states={30})
    nfa.add_transition(10, "a", 10)
    nfa.add_transition(10, "a", 30)
    nfa.add_transition(20, "b", 30)
    intersection = intersect_automata(
        AdjacencyMatrixFA(nfa), AdjacencyMatrixFA(regex_to_dfa("a a*"))
    )

    assert intersection.accepts(["a"])
    assert intersection.accepts(["a", "a"])
    assert not intersection.accepts(["b"])
    assert not intersection.accepts([])
    assert not intersection.is_empty()
    assert any(first == State(30) for first, _ in intersection.states)
