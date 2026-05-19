# TALE — Token-Budget-Aware LLM Reasoning

**Paper**: [arXiv:2412.18547](https://arxiv.org/abs/2412.18547) (ACL 2025 Findings)
**Open-source Reference**: https://github.com/GeniusHTX/TALE

## Core Method

TALE dynamically estimates token budgets and injects budget constraints into prompts to guide the model toward generating shorter reasoning processes.

### Key Steps
1. **Difficulty Classification**: Categorize questions into easy/medium/hard based on dataset
2. **Budget Estimation**: Preset token budgets for each difficulty level
3. **Prompt Injection**: Inject budget constraints into the prompt
4. **Reasoning Generation**: Model generates more concise reasoning under budget constraints

### Budget Tiers

| Difficulty | Datasets | Default Budget | Max Budget | Baseline Think Tokens |
|------------|----------|---------------|------------|----------------------|
| easy | GSM8K, AMC | 800 | 1500 | 1144 / 4240 |
| medium | GPQA | 500 | 800 | 651 |
| hard | MATH-500, AIME | 2000 | 4000 | 2800 / 7986 |

### Deployment Modes
- **preset**: Preset budgets by dataset (training-free, plug-and-play)
- **dynamic**: Fine-tune budgets based on question complexity (future extension)

## File Structure

```
implementations/tale/
├── src/
│   ├── __init__.py    — Module exports
│   ├── core.py        — Core engine: TALEEngine
│   ├── adapter.py     — Eval framework adapter: TALEAdapter
│   └── utils.py       — Utility functions
├── config.yaml        — Configuration parameters
├── paper_info.yaml    — Paper metadata
└── README.md          — This file
```

## Usage

```python
from implementations.tale.src import TALEEngine

engine = TALEEngine()
messages, budget = engine.build_messages(question, dataset="GSM8K")
prompt_text, budget = engine.build_prompt_text(question, dataset="MATH500")
```

### Run Evaluation

```bash
python -m implementations.tale.src.adapter \
    --dataset benchmarks/datasets/eval/gsm8k.json \
    --output results/tale/gsm8k_tale.jsonl \
    --vllm-url http://localhost:8000 \
    --concurrency 256
```
