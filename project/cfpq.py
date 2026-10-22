from collections import defaultdict, deque

import networkx as nx
from pyformlang.cfg import CFG, Production

from project.automata_utils import graph_to_nfa


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    generates_epsilon = cfg.generate_epsilon()
    normal_form = cfg.to_normal_form()
    productions = set(normal_form.productions)
    if generates_epsilon:
        productions.add(Production(cfg.start_symbol, []))
    return CFG(start_symbol=cfg.start_symbol, productions=productions)


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    start_nodes = set(graph) if start_nodes is None else set(start_nodes)
    final_nodes = set(graph) if final_nodes is None else set(final_nodes)
    if not start_nodes or not final_nodes:
        return set()

    automaton = graph_to_nfa(nx.MultiDiGraph(graph), start_nodes, final_nodes)
    grammar = cfg_to_weak_normal_form(cfg)
    terminal_rules = defaultdict(set)
    left_rules = defaultdict(list)
    right_rules = defaultdict(list)
    nullable = set()
    for production in grammar.productions:
        head, body = production.head, production.body
        if not body:
            nullable.add(head)
        elif len(body) == 1:
            terminal_rules[body[0].value].add(head)
        else:
            left, right = body
            left_rules[left].append((head, right))
            right_rules[right].append((head, left))

    reachable = set()
    pending = deque()
    outgoing = defaultdict(set)
    incoming = defaultdict(set)

    def add(variable, source, target):
        triple = (variable, source, target)
        if triple not in reachable:
            reachable.add(triple)
            pending.append(triple)
            outgoing[variable, source].add(target)
            incoming[variable, target].add(source)

    for variable in nullable:
        for state in automaton.states:
            add(variable, state.value, state.value)
    for source, transitions in automaton.to_dict().items():
        for label, targets in transitions.items():
            for variable in terminal_rules[label.value]:
                for target in targets:
                    add(variable, source.value, target.value)

    while pending:
        variable, source, target = pending.popleft()
        for head, right in left_rules[variable]:
            for end in tuple(outgoing[right, target]):
                add(head, source, end)
        for head, left in right_rules[variable]:
            for begin in tuple(incoming[left, source]):
                add(head, begin, target)

    return {
        (source, target)
        for variable, source, target in reachable
        if variable == grammar.start_symbol
        and source in start_nodes
        and target in final_nodes
    }
