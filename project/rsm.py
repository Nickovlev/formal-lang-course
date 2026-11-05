from pyformlang.cfg import CFG
from pyformlang.finite_automaton import Epsilon, EpsilonNFA
from pyformlang.rsa import Box, RecursiveAutomaton

from project.automata_utils import regex_to_dfa


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    automata = {
        variable: EpsilonNFA(start_state={0}, final_states={1})
        for variable in cfg.variables | {cfg.start_symbol}
    }
    for production_index, production in enumerate(cfg.productions):
        automaton = automata[production.head]
        if not production.body:
            automaton.add_transition(0, Epsilon(), 1)
            continue

        source = 0
        for position, symbol in enumerate(production.body):
            target = (
                1
                if position == len(production.body) - 1
                else (production_index, position)
            )
            automaton.add_transition(source, symbol.value, target)
            source = target

    boxes = {Box(automaton, variable.value) for variable, automaton in automata.items()}
    return RecursiveAutomaton(initial_label=cfg.start_symbol.value, boxes=boxes)


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    alternatives = {}
    for line in ebnf.splitlines():
        if not line.strip():
            continue
        head, body = line.split("->", 1)
        alternatives.setdefault(head.strip(), []).append(body.strip() or "epsilon")

    alternatives.setdefault("S", [])
    boxes = set()
    for head, bodies in alternatives.items():
        regex = " | ".join(f"({body})" for body in bodies)
        boxes.add(Box(regex_to_dfa(regex), head))
    return RecursiveAutomaton(initial_label="S", boxes=boxes)
