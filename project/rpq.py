from networkx import MultiDiGraph
from scipy import sparse

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.automata_utils import graph_to_nfa, regex_to_dfa


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    if not start_nodes or not final_nodes:
        return set()

    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    product = intersect_automata(graph_fa, regex_fa)
    reachable = product.transitive_closure()

    answer = set()
    for source in product.start_states:
        for target in reachable.getrow(source).indices:
            if target in product.final_states:
                graph_source, _ = product.states[source]
                graph_target, _ = product.states[target]
                answer.add((graph_source.value, graph_target.value))
    return answer


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    if not start_nodes or not final_nodes:
        return set()

    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    starts = list(start_nodes)
    shape = (len(starts), len(graph_fa.states))

    initial = sparse.csr_matrix(
        (
            [True] * len(starts),
            (
                list(range(len(starts))),
                [graph_fa.state_to_index[node] for node in starts],
            ),
        ),
        shape=shape,
        dtype=bool,
    )
    frontier = [
        initial.copy()
        if state in regex_fa.start_states
        else sparse.csr_matrix(shape, dtype=bool)
        for state in range(len(regex_fa.states))
    ]
    visited = [matrix.copy() for matrix in frontier]
    transitions = []
    for symbol in graph_fa.matrices.keys() & regex_fa.matrices.keys():
        sources, targets = regex_fa.matrices[symbol].nonzero()
        transitions.extend(
            (source, symbol, target) for source, target in zip(sources, targets)
        )

    while any(matrix.nnz for matrix in frontier):
        expanded = [sparse.csr_matrix(shape, dtype=bool) for _ in regex_fa.states]
        for source, symbol, target in transitions:
            expanded[target] = (
                expanded[target] + frontier[source] @ graph_fa.matrices[symbol]
            )

        frontier = [matrix > seen for matrix, seen in zip(expanded, visited)]
        visited = [seen + matrix for seen, matrix in zip(visited, frontier)]

    answer = set()
    for state in regex_fa.final_states:
        rows, columns = visited[state].nonzero()
        for row, column in zip(rows, columns):
            if column in graph_fa.final_states:
                answer.add((starts[row], graph_fa.states[column].value))
    return answer
