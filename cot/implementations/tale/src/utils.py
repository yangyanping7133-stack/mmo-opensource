import re
import json


BASELINE_THINK_TOKENS = {
    "GSM8K": 1144,
    "gsm8k": 1144,
    "MATH500": 2800,
    "MATH-500": 2800,
    "math500": 2800,
    "GPQA": 651,
    "gpqa": 651,
    "AMC": 4240,
    "amc": 4240,
    "AIME": 7986,
    "aime": 7986,
}

DATASET_DIFFICULTY = {
    "GSM8K": "easy",
    "gsm8k": "easy",
    "MATH500": "hard",
    "MATH-500": "hard",
    "math500": "hard",
    "GPQA": "medium",
    "gpqa": "medium",
    "AIME": "hard",
    "aime": "hard",
    "AMC": "easy",
    "amc": "easy",
}


def estimate_think_reduction(dataset: str, actual_think_tokens: int) -> dict:
    baseline = BASELINE_THINK_TOKENS.get(dataset, 2609)
    reduction = baseline - actual_think_tokens
    reduction_pct = (reduction / baseline * 100) if baseline > 0 else 0
    return {
        "dataset": dataset,
        "baseline_think_tokens": baseline,
        "actual_think_tokens": actual_think_tokens,
        "reduction": reduction,
        "reduction_pct": round(reduction_pct, 1),
    }


def count_think_tokens(text: str) -> int:
    # Match <think...>content</think...> — > after think is optional (Qwen3: <think\n...\n</think\n>)
    think_match = re.search(r"<think\s*>?(.*?)</think\s*>?", text, re.DOTALL)
    if think_match:
        return len(think_match.group(1).split())
    think_match = re.search(r"<think\s*>?(.*?)(?:</think\s*>?|$)", text, re.DOTALL)
    if think_match:
        return len(think_match.group(1).split())
    return 0


def analyze_results(results_jsonl: str) -> dict:
    results = []
    with open(results_jsonl) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        return {"error": "no results"}

    total = len(results)
    correct = sum(1 for r in results if r.get("judge_correct"))
    avg_tokens = sum(r.get("completion_tokens", 0) for r in results) / total
    avg_think = sum(r.get("thinking_tokens_est", 0) for r in results) / total
    avg_time = sum(r.get("total_time", 0) for r in results) / total

    by_dataset = {}
    for r in results:
        ds = r.get("dataset", "unknown")
        if ds not in by_dataset:
            by_dataset[ds] = {"total": 0, "correct": 0, "tokens": 0, "think": 0, "time": 0}
        by_dataset[ds]["total"] += 1
        if r.get("judge_correct"):
            by_dataset[ds]["correct"] += 1
        by_dataset[ds]["tokens"] += r.get("completion_tokens", 0)
        by_dataset[ds]["think"] += r.get("thinking_tokens_est", 0)
        by_dataset[ds]["time"] += r.get("total_time", 0)

    report = {
        "method": "tale",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1),
        "avg_completion_tokens": round(avg_tokens),
        "avg_thinking_tokens": round(avg_think),
        "avg_time_ms": round(avg_time),
        "by_dataset": {},
    }

    for ds, stats in by_dataset.items():
        n = stats["total"]
        reduction = estimate_think_reduction(ds, round(stats["think"] / n))
        report["by_dataset"][ds] = {
            "correct": stats["correct"],
            "total": n,
            "accuracy": round(stats["correct"] / n * 100, 1),
            "avg_completion_tokens": round(stats["tokens"] / n),
            "avg_thinking_tokens": round(stats["think"] / n),
            "avg_time_ms": round(stats["time"] / n),
            "think_reduction_pct": reduction["reduction_pct"],
        }

    return report


def validate_config(config: dict) -> list:
    warnings = []
    params = config.get("params", {})

    mode = params.get("mode", "preset")
    if mode not in ("preset", "dynamic"):
        warnings.append(f"Unknown mode: {mode}")

    token_budget = config.get("token_budget", {})
    for difficulty in ("easy", "medium", "hard"):
        if difficulty not in token_budget:
            warnings.append(f"Missing token_budget for difficulty: {difficulty}")
        else:
            default = token_budget[difficulty].get("default", 0)
            max_val = token_budget[difficulty].get("max", 0)
            if default <= 0:
                warnings.append(f"Invalid default budget for {difficulty}: {default}")
            if max_val < default:
                warnings.append(f"max budget < default for {difficulty}")

    return warnings
