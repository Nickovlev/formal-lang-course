from itertools import product

import pytest
from pyformlang.cfg import CFG, Terminal, Variable

from project.cfpq import cfg_to_weak_normal_form


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", set()),
        ("S -> epsilon", {""}),
        ("S -> A\nA -> B\nB -> epsilon", {""}),
        ("S -> A\nA -> B\nB -> A | a", {"a"}),
        ("S -> a S b | epsilon", {"", "ab", "aabb"}),
        ("S -> a A b B\nA -> a | epsilon\nB -> b", {"abb", "aabb"}),
        ("S -> A B\nA -> a | epsilon\nB -> b | epsilon", {"", "a", "b", "ab"}),
        ("S -> a\nU -> b U\nV -> b", {"a"}),
        ("S -> S", set()),
    ],
)
def test_weak_normal_form_and_language(text, expected):
    cfg = CFG.from_text(text)
    original_productions = set(cfg.productions)

    result = cfg_to_weak_normal_form(cfg)

    assert result.start_symbol == cfg.start_symbol
    assert cfg.productions == original_productions
    for production in result.productions:
        body = production.body
        assert (
            len(body) == 0
            or (len(body) == 1 and isinstance(body[0], Terminal))
            or (len(body) == 2 and all(isinstance(s, Variable) for s in body))
        )
    for length in range(5):
        for word in product("ab", repeat=length):
            assert result.contains(word) == ("".join(word) in expected)


def test_custom_start_symbol():
    cfg = CFG.from_text("Root -> a Root | epsilon", start_symbol="Root")

    result = cfg_to_weak_normal_form(cfg)

    assert result.start_symbol == Variable("Root")
    assert result.contains([])
    assert result.contains(["a", "a", "a"])
    assert not result.contains(["b"])
