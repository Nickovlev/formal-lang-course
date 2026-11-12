import networkx as nx
from pyformlang.rsa import RecursiveAutomaton
from scipy import sparse

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.automata_utils import graph_to_nfa


def tensor_based_cfpq(
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
    graph_fa = AdjacencyMatrixFA(automaton)
    size = len(graph_fa.states)
    graph_fa.start_states = set(range(size))
    graph_fa.final_states = set(range(size))
    boxes = {label: AdjacencyMatrixFA(box.dfa) for label, box in rsm.boxes.items()}
    for label in boxes:
        graph_fa.matrices[label] = sparse.csr_matrix((size, size), dtype=bool)

    changed = True
    while changed:
        changed = False
        for label, box_fa in boxes.items():
            product = intersect_automata(graph_fa, box_fa)
            reachable = product.transitive_closure()
            rows, columns = [], []
            for source in product.start_states:
                graph_source, _ = product.states[source]
                for target in reachable.getrow(source).indices:
                    if target in product.final_states:
                        graph_target, _ = product.states[target]
                        rows.append(graph_fa.state_to_index[graph_source])
                        columns.append(graph_fa.state_to_index[graph_target])

            paths = sparse.csr_matrix(
                ([True] * len(rows), (rows, columns)),
                shape=(size, size),
                dtype=bool,
            )
            previous = graph_fa.matrices[label]
            expanded = previous + paths
            if expanded.nnz > previous.nnz:
                graph_fa.matrices[label] = expanded
                changed = True

    answer = set()
    rows, columns = graph_fa.matrices[rsm.initial_label].nonzero()
    for row, column in zip(rows, columns):
        source = graph_fa.states[row].value
        target = graph_fa.states[column].value
        if source in start_nodes and target in final_nodes:
            answer.add((source, target))
    return answer
