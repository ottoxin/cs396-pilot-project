import json
import os
import random
from typing import Any, Dict, Iterable, List

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def save_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    import pandas as pd

    ensure_dir(os.path.dirname(path))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
