from src.data.gsm8k import parse_gsm8k_answer


def test_parse_basic():
    assert parse_gsm8k_answer("... #### 123") == "123"


def test_parse_commas():
    assert parse_gsm8k_answer("answer is #### 1,234") == "1234"


def test_parse_negative_decimal():
    assert parse_gsm8k_answer("#### -3.5") == "-3.5"


def test_parse_missing():
    assert parse_gsm8k_answer("no hash") is None
