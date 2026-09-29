from collections.abc import Iterable

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from scipy import sparse


class AdjacencyMatrixFA:
    def __init__(self, automaton: NondeterministicFiniteAutomaton | None = None):
        self.states = [] if automaton is None else list(automaton.states)
        self.state_to_index = {state: i for i, state in enumerate(self.states)}
        self.start_states: set[int] = set()
        self.final_states: set[int] = set()
        self.matrices: dict[Symbol, sparse.csr_matrix] = {}
        if automaton is None:
            return

        self.start_states = {
            self.state_to_index[state] for state in automaton.start_states
        }
        self.final_states = {
            self.state_to_index[state] for state in automaton.final_states
        }
        transitions = {}
        for source, outgoing in automaton.to_dict().items():
            for symbol, targets in outgoing.items():
                if not isinstance(targets, set):
                    targets = {targets}
                rows, columns = transitions.setdefault(symbol, ([], []))
                for target in targets:
                    rows.append(self.state_to_index[source])
                    columns.append(self.state_to_index[target])

        size = len(self.states)
        for symbol, (rows, columns) in transitions.items():
            self.matrices[symbol] = sparse.csr_matrix(
                ([True] * len(rows), (rows, columns)),
                shape=(size, size),
                dtype=bool,
            )

    def accepts(self, word: Iterable[Symbol]) -> bool:
        starts = list(self.start_states)
        current = sparse.csr_matrix(
            ([True] * len(starts), ([0] * len(starts), starts)),
            shape=(1, len(self.states)),
            dtype=bool,
        )
        for symbol in word:
            matrix = self.matrices.get(Symbol(symbol))
            if matrix is None:
                return False
            current = current @ matrix
            if current.nnz == 0:
                return False
        return bool(self.final_states.intersection(current.indices))

    def transitive_closure(self) -> sparse.csr_matrix:
        reachable = sparse.eye(len(self.states), format="csr", dtype=bool)
        for matrix in self.matrices.values():
            reachable = reachable + matrix
        while True:
            expanded = reachable + reachable @ reachable
            if expanded.nnz == reachable.nnz:
                return expanded
            reachable = expanded

    def is_empty(self) -> bool:
        if not self.start_states or not self.final_states:
            return True
        reachable = self.transitive_closure()
        return reachable[list(self.start_states)][:, list(self.final_states)].nnz == 0


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    result = AdjacencyMatrixFA()
    result.states = [
        (first, second) for first in automaton1.states for second in automaton2.states
    ]
    result.state_to_index = {state: i for i, state in enumerate(result.states)}
    size2 = len(automaton2.states)
    result.start_states = {
        first * size2 + second
        for first in automaton1.start_states
        for second in automaton2.start_states
    }
    result.final_states = {
        first * size2 + second
        for first in automaton1.final_states
        for second in automaton2.final_states
    }
    for symbol in automaton1.matrices.keys() & automaton2.matrices.keys():
        result.matrices[symbol] = sparse.kron(
            automaton1.matrices[symbol], automaton2.matrices[symbol], format="csr"
        )
    return result
