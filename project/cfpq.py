from collections import defaultdict, deque

import networkx as nx
from pyformlang.cfg import CFG, Production
from pyformlang.finite_automaton import Symbol
from scipy import sparse

from project.adjacency_matrix_fa import AdjacencyMatrixFA
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


def matrix_based_cfpq(
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
    graph_fa = AdjacencyMatrixFA(automaton)
    grammar = cfg_to_weak_normal_form(cfg)
    size = len(graph_fa.states)
    matrices = {
        variable: sparse.csr_matrix((size, size), dtype=bool)
        for variable in grammar.variables | {grammar.start_symbol}
    }

    dependent_rules = defaultdict(list)
    for production in grammar.productions:
        head, body = production.head, production.body
        if not body:
            matrices[head] = matrices[head] + sparse.eye(size, format="csr", dtype=bool)
        elif len(body) == 1:
            terminal_matrix = graph_fa.matrices.get(Symbol(body[0].value))
            if terminal_matrix is not None:
                matrices[head] = matrices[head] + terminal_matrix
        else:
            left, right = body
            rule = (head, left, right)
            dependent_rules[left].append(rule)
            if right != left:
                dependent_rules[right].append(rule)

    pending = deque(variable for variable, matrix in matrices.items() if matrix.nnz)
    queued = set(pending)
    while pending:
        variable = pending.popleft()
        queued.remove(variable)
        for head, left, right in dependent_rules[variable]:
            new_paths = matrices[left] @ matrices[right]
            expanded = matrices[head] + new_paths
            if expanded.nnz > matrices[head].nnz:
                matrices[head] = expanded
                if head not in queued:
                    pending.append(head)
                    queued.add(head)

    answer = set()
    rows, columns = matrices[grammar.start_symbol].nonzero()
    for row, column in zip(rows, columns):
        source = graph_fa.states[row].value
        target = graph_fa.states[column].value
        if source in start_nodes and target in final_nodes:
            answer.add((source, target))
    return answer
