from collections import defaultdict, deque

import networkx as nx
from pyformlang.rsa import RecursiveAutomaton

from project.automata_utils import graph_to_nfa


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    start_nodes = set(graph) if start_nodes is None else set(start_nodes)
    final_nodes = set(graph) if final_nodes is None else set(final_nodes)
    if not start_nodes or not final_nodes:
        return set()

    automaton = graph_to_nfa(nx.MultiDiGraph(graph), start_nodes, final_nodes)
    graph_transitions = automaton.to_dict()
    rsm_start_states = {}
    rsm_final_states = set()
    rsm_transitions = {}
    for label, box in rsm.boxes.items():
        rsm_start_states[label] = {(label, state) for state in box.start_state}
        rsm_final_states.update((label, state) for state in box.final_states)
        for state, transitions in box.dfa.to_dict().items():
            rsm_transitions[label, state] = {
                symbol: (label, target) for symbol, target in transitions.items()
            }

    pending = deque()
    visited = set()
    gss_edges = defaultdict(set)
    popped = defaultdict(set)

    def add(vertex, state, stack_node):
        descriptor = (vertex, state, stack_node)
        if descriptor not in visited:
            visited.add(descriptor)
            pending.append(descriptor)

    def create(vertex, label, return_state, caller):
        callee = (label, vertex)
        edge = (return_state, caller)
        if edge not in gss_edges[callee]:
            gss_edges[callee].add(edge)
            for end in popped[callee]:
                add(end, return_state, caller)
        return callee

    def pop(vertex, stack_node):
        if vertex not in popped[stack_node]:
            popped[stack_node].add(vertex)
            for return_state, caller in gss_edges[stack_node]:
                add(vertex, return_state, caller)

    for vertex in automaton.start_states:
        stack_node = (rsm.initial_label, vertex)
        for state in rsm_start_states[rsm.initial_label]:
            add(vertex, state, stack_node)

    while pending:
        vertex, state, stack_node = pending.popleft()

        if state in rsm_final_states:
            pop(vertex, stack_node)

        for symbol, next_state in rsm_transitions.get(state, {}).items():
            if symbol in rsm.boxes:
                callee = create(vertex, symbol, next_state, stack_node)
                for entry_state in rsm_start_states[symbol]:
                    add(vertex, entry_state, callee)
            else:
                targets = graph_transitions.get(vertex, {}).get(symbol, ())
                for target in targets:
                    add(target, next_state, stack_node)

    answer = set()
    for source in automaton.start_states:
        stack_node = (rsm.initial_label, source)
        for target in popped[stack_node]:
            if target in automaton.final_states:
                answer.add((source.value, target.value))
    return answer
