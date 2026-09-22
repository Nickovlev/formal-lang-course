from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    Symbol,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph,
    start_states: set[int] | None = None,
    final_states: set[int] | None = None,
) -> NondeterministicFiniteAutomaton:
    states = set(graph.nodes)
    start_states = states if not start_states else set(start_states)
    final_states = states if not final_states else set(final_states)
    for name, selected in (
        ("start_states", start_states),
        ("final_states", final_states),
    ):
        unknown = selected - states
        if unknown:
            raise ValueError(f"{name} contains vertices not in the graph: {unknown}")

    nfa = NondeterministicFiniteAutomaton(
        states=states, start_state=start_states, final_states=final_states
    )
    for source, target, key, data in graph.edges(keys=True, data=True):
        if "label" not in data or data["label"] is None:
            raise ValueError(f"Edge ({source}, {target}, {key}) has no label")
        nfa.add_transition(source, Symbol(data["label"]), target)
    return nfa
