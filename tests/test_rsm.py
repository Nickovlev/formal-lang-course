import pytest
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import Symbol

from project.rsm import cfg_to_rsm, ebnf_to_rsm


def test_cfg_boxes_preserve_production_bodies_and_custom_start():
    cfg = CFG.from_text(
        "Root -> a A b | epsilon\nA -> hello | A A", start_symbol="Root"
    )
    original = set(cfg.productions)

    rsm = cfg_to_rsm(cfg)

    assert cfg.productions == original
    assert rsm.initial_label == Symbol("Root")
    root = rsm.boxes[Symbol("Root")].dfa
    assert root.accepts([])
    assert root.accepts(["a", "A", "b"])
    assert not root.accepts(["a", "hello", "b"])
    assert not root.accepts(["a", "A", "A", "b"])
    other = rsm.boxes[Symbol("A")].dfa
    assert other.accepts(["hello"])
    assert other.accepts(["A", "A"])
    assert not other.accepts([])


@pytest.mark.parametrize("text", ["", "S -> A"])
def test_cfg_variables_without_productions_have_empty_boxes(text):
    cfg = CFG.from_text(text)

    rsm = cfg_to_rsm(cfg)

    assert Symbol("S") in rsm.boxes
    if text:
        assert rsm.boxes[Symbol("A")].dfa.is_empty()
    else:
        assert rsm.boxes[Symbol("S")].dfa.is_empty()


def test_ebnf_repeated_heads_and_regular_operations():
    rsm = ebnf_to_rsm("\n S -> (a | b)* A\n S ->\n A -> hello\n")

    assert rsm.initial_label == Symbol("S")
    start = rsm.boxes[Symbol("S")].dfa
    assert start.accepts([])
    assert start.accepts(["A"])
    assert start.accepts(["a", "b", "a", "A"])
    assert not start.accepts(["a", "b"])
    assert rsm.boxes[Symbol("A")].dfa.accepts(["hello"])


def test_empty_ebnf_has_empty_start_box():
    rsm = ebnf_to_rsm("")

    assert rsm.boxes[Symbol("S")].dfa.is_empty()
