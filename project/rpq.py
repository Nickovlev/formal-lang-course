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
    regex_size = len(regex_fa.states)
    shape = (len(starts) * regex_size, len(graph_fa.states))

    rows, columns = [], []
    for source_index, start in enumerate(starts):
        for state in regex_fa.start_states:
            rows.append(source_index * regex_size + state)
            columns.append(graph_fa.state_to_index[start])
    frontier = sparse.csr_matrix(
        ([True] * len(rows), (rows, columns)), shape=shape, dtype=bool
    )
    visited = frontier.copy()

    symbols = graph_fa.matrices.keys() & regex_fa.matrices.keys()
    regex_transitions = {}
    for symbol in symbols:
        regex_transitions[symbol] = sparse.block_diag(
            [regex_fa.matrices[symbol].transpose()] * len(starts), format="csr"
        )

    while frontier.nnz:
        expanded = sparse.csr_matrix(shape, dtype=bool)
        for symbol in symbols:
            expanded = expanded + (
                regex_transitions[symbol] @ frontier @ graph_fa.matrices[symbol]
            )

        frontier = expanded > visited
        visited = visited + frontier

    answer = set()
    rows, columns = visited.nonzero()
    for row, column in zip(rows, columns):
        source_index, state = divmod(row, regex_size)
        if state in regex_fa.final_states and column in graph_fa.final_states:
            answer.add((starts[source_index], graph_fa.states[column].value))
    return answer
