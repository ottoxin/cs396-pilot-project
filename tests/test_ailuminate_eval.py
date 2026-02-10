import pytest

from src.eval.ailuminate_eval import _extract_prompt


def test_extract_prompt_from_prompt():
    assert _extract_prompt({"prompt": "hello"}) == "hello"


def test_extract_prompt_from_input():
    assert _extract_prompt({"input": "hello"}) == "hello"


def test_extract_prompt_from_prompt_text():
    assert _extract_prompt({"prompt_text": "hello"}) == "hello"


def test_extract_prompt_raises_when_missing():
    with pytest.raises(KeyError):
        _extract_prompt({"hazard": "x"})
