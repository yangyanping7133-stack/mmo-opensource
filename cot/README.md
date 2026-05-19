# CoT Compression — Chain-of-Thought Token Reduction

Evaluation framework and method implementations for reducing CoT (Chain-of-Thought) token usage in LLM reasoning, without training.

## Project Goal

Reduce CoT thinking tokens by ≥30% and inference latency by ≥20%, while maintaining accuracy at or above baseline levels. All methods are **training-free** (plug-and-play).

## Evaluation Framework

### Core Library: `evaluation/run_eval.py`

A standardized async evaluation runner that supports:
- Concurrent evaluation with configurable parallelism
- Resume from existing results (JSONL)
- Model-based judging (LLM-as-judge) for math answers
- Exact match for multiple-choice questions
- Per-dataset statistics and summary output

### Usage

```bash
# Set environment variables
export VLLM_URL="http://your-server:8000"
export VLLM_API_KEY="your-api-key"
export MODEL_NAME="Qwen/Qwen3-32B"

# Run baseline evaluation
python -m cot.evaluation.run_eval \
    --dataset data/gsm8k.json \
    --output results/baseline_gsm8k.jsonl \
    --vllm-url $VLLM_URL \
    --concurrency 32 \
    --method baseline

# View summary from existing results
python -m cot.evaluation.run_eval \
    --dataset data/gsm8k.json \
    --output results/baseline_gsm8k.jsonl \
    --summary-only
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_URL` | vLLM server URL | `http://localhost:8000` |
| `VLLM_API_KEY` | API key for authentication | `YOUR_API_KEY` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen3-32B` |

## Implemented Methods

### DEER — Dynamic Early Exit in Reasoning
- **Paper**: [arXiv:2504.15895](https://arxiv.org/abs/2504.15895)
- **Method**: Monitors "Wait" breakpoints in reasoning, injects trial answer prompts, evaluates confidence via heuristic scoring, and exits early when confidence exceeds threshold.
- **Location**: `implementations/deer/`

### TALE — Token-Budget-Aware LLM Reasoning
- **Paper**: [arXiv:2412.18547](https://arxiv.org/abs/2412.18547) (ACL 2025 Findings)
- **Method**: Estimates token budget per question difficulty and injects budget constraints into the prompt to guide shorter reasoning.
- **Location**: `implementations/tale/`

### Don't Overthink — Short-m@k Majority Voting
- **Paper**: [arXiv:2412.21194](https://arxiv.org/abs/2412.21194)
- **Method**: Generates k short responses, uses early-stopping (first m completions), and applies majority voting for the final answer.
- **Location**: `implementations/dontoverthink/`

## Benchmark Datasets

The evaluation suite uses 5 datasets totaling 1,270 questions:

| Dataset | Count | Type |
|---------|-------|------|
| GSM8K | 500 | Math |
| MATH-500 | 500 | Math |
| GPQA | 200 | Multiple Choice |
| AMC | 40 | Math |
| AIME | 30 | Math |

Dataset files are not included in this repository. See `benchmarks/manifest.json` for the expected format.

## Documentation

- `docs/INSIGHT_REPORT.md` — Survey of 22+ CoT compression papers (2026)
- `docs/ADR-002-2026-cot-compress.md` — Architecture Decision Record for method selection

## Dependencies

```
httpx
aiohttp
pyyaml
```
