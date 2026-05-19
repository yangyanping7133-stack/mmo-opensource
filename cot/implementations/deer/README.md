# DEER — Dynamic Early Exit in Reasoning

**Paper**: [arXiv:2504.15895](https://arxiv.org/abs/2504.15895)

## Method

DEER monitors "Wait" breakpoints in the reasoning process, injects trial answer prompts, evaluates confidence, and exits early when confidence is high enough.

### Key Steps
1. **Wait Breakpoint Detection**: Identify Wait/Hmm re-evaluation points in thinking
2. **Trial Answer Injection**: Inject trial answer prompt at breakpoints
3. **Confidence Evaluation**: Heuristic scoring of trial answer confidence
4. **Early Exit**: Terminate reasoning when confidence exceeds threshold
5. **Progressive Budget**: Support [0.5x, 0.8x, 1.0x] multi-level budgets

### Per-Dataset Parameters

| Dataset | Threshold | Max Steps | Baseline Think Tokens |
|---------|-----------|-----------|----------------------|
| GSM8K | 0.95 | 4 | 1144 |
| MATH500 | 0.98 | 8 | 2800 |
| GPQA | 0.95 | 4 | 651 |
| AMC | 0.98 | 12 | 4240 |
| AIME | 0.98 | 12 | 7986 |

## File Structure

```
implementations/deer/
├── src/
│   ├── __init__.py    — Module exports
│   ├── core.py        — Core engine: DEEREngine
│   ├── adapter.py     — Eval framework adapter: DEERAdapter
│   └── utils.py       — Utility functions
├── config.yaml        — Configuration
├── paper_info.yaml    — Paper metadata
└── README.md          — This file
```

## Usage

```python
from implementations.deer.src import DEEREngine

engine = DEEREngine()
messages = engine.build_initial_messages(question, dataset="GSM8K")
breakpoints = engine.detect_wait_breakpoint(think_text)
confidence = engine.evaluate_confidence(response)
should_stop = engine.should_exit(confidence, step=1)
budgets = engine.get_progressive_budgets("MATH500")
```

### Run Evaluation

```bash
python -m implementations.deer.src.adapter \
    --dataset benchmarks/datasets/eval/gsm8k.json \
    --output results/deer/gsm8k_deer.jsonl \
    --vllm-url http://localhost:8000 \
    --concurrency 256
```
