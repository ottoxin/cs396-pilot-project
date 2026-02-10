# CS396 Pilot Project Report

## Log
- `2026-02-10`: Nick completed and shared `simple` run training + GSM8K evaluation logs (AILuminate pending).

## Metrics Tracker

| Run | Baseline GSM8K Acc | Baseline Safety Rate | GSM8K Acc (Result) | Safety Rate (Result) | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| simple | 0.26 | 0.26 | 0.4890 | pending | GSM8K passed baseline; waiting on AILuminate safety result |
| medium | 0.31 | 0.34 | TBD | TBD | Placeholder |
| strong | 0.37 | 0.42 | TBD | TBD | Placeholder |

## Simple Run Summary

Command used in that run (previous repo layout):

```bash
HF_HOME="/content/drive/MyDrive/hf_cache" PYTHONPATH=. python scripts/run_experiment.py --config configs/simple.yaml
```

### Training
- Trainable params: `9,232,384 / 1,552,946,688` (`0.5945%`)
- GSM8K train loaded: `7473` examples
- GSM8K test loaded: `1319` examples
- Train runtime: `1925.5739s` (~`32.1 min`)
- Final train loss: `0.3707377507293505`
- Steps: `467/467`

### GSM8K Evaluation
- Total questions: `1319`
- Correct: `645`
- Accuracy: `48.90%`
- Eval runtime: `4:43:35` (~`12.90s/item`)

### Example Wrong Case
- Expected: `70000`
- Model output: `#### -38000`
- Topic: house flipping profit/loss
