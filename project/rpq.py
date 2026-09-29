from networkx import MultiDiGraph

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
