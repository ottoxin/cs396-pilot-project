from datasets import Dataset
from src.data.gsm8k import prepare_fewshot


def test_fewshot_deterministic(tmp_path):
    data = Dataset.from_dict({"question": ["q1", "q2", "q3", "q4"], "answer": ["a1", "a2", "a3", "a4"]})
    fs_path = tmp_path / "few.jsonl"
    remaining, few = prepare_fewshot(data, k=2, fewshot_path=str(fs_path), seed=42)
    remaining2, few2 = prepare_fewshot(data, k=2, fewshot_path=str(fs_path), seed=99)
    assert few == few2
    # MAX_FEWSHOT=8, but dataset has 4 rows => all 4 removed from training
    assert len(remaining) == len(data) - 4
    assert len(remaining2) == len(data) - 4
