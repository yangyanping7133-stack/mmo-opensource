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
    avg_confidence = sum(r.get("confidence", 0) for r in results) / total
    avg_wait_breakpoints = sum(r.get("wait_breakpoints", 0) for r in results) / total

    by_dataset = {}
    for r in results:
        ds = r.get("dataset", "unknown")
        if ds not in by_dataset:
            by_dataset[ds] = {"total": 0, "correct": 0, "tokens": 0, "think": 0, "time": 0, "confidence": 0, "breakpoints": 0}
        by_dataset[ds]["total"] += 1
        if r.get("judge_correct"):
            by_dataset[ds]["correct"] += 1
        by_dataset[ds]["tokens"] += r.get("completion_tokens", 0)
        by_dataset[ds]["think"] += r.get("thinking_tokens_est", 0)
        by_dataset[ds]["time"] += r.get("total_time", 0)
        by_dataset[ds]["confidence"] += r.get("confidence", 0)
        by_dataset[ds]["breakpoints"] += r.get("wait_breakpoints", 0)

    report = {
        "method": "deer",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1),
        "avg_completion_tokens": round(avg_tokens),
        "avg_thinking_tokens": round(avg_think),
        "avg_time_ms": round(avg_time),
        "avg_confidence": round(avg_confidence, 3),
        "avg_wait_breakpoints": round(avg_wait_breakpoints, 1),
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
            "avg_confidence": round(stats["confidence"] / n, 3),
            "avg_wait_breakpoints": round(stats["breakpoints"] / n, 1),
            "think_reduction_pct": reduction["reduction_pct"],
        }

    return report


def validate_config(config: dict) -> list:
    warnings = []
    params = config.get("params", {})

    threshold = params.get("threshold", 0.95)
    if not (0.0 < threshold <= 1.0):
        warnings.append(f"Invalid threshold: {threshold} (must be in (0, 1])")

    max_steps = params.get("max_steps", 4)
    if max_steps < 1:
        warnings.append(f"Invalid max_steps: {max_steps} (must be >= 1)")

    think_ratio = params.get("think_ratio", 0.8)
    if not (0.0 < think_ratio <= 1.0):
        warnings.append(f"Invalid think_ratio: {think_ratio} (must be in (0, 1])")

    per_dataset = config.get("per_dataset", {})
    for ds, ds_params in per_dataset.items():
        ds_threshold = ds_params.get("threshold", threshold)
        if not (0.0 < ds_threshold <= 1.0):
            warnings.append(f"Invalid threshold for {ds}: {ds_threshold}")
        ds_max_steps = ds_params.get("max_steps", max_steps)
        if ds_max_steps < 1:
            warnings.append(f"Invalid max_steps for {ds}: {ds_max_steps}")

    baseline = config.get("baseline_think_tokens", {})
    for ds in ("GSM8K", "MATH500", "GPQA", "AMC", "AIME"):
        if ds not in baseline:
            warnings.append(f"Missing baseline_think_tokens for {ds}")
        elif baseline[ds] <= 0:
            warnings.append(f"Invalid baseline_think_tokens for {ds}: {baseline[ds]}")

    progressive = config.get("progressive_budget", {})
    if progressive.get("enabled", False):
        levels = progressive.get("levels", [])
        if not levels:
            warnings.append("Progressive budget enabled but no levels defined")
        for level in levels:
            if not (0.0 < level <= 1.0):
                warnings.append(f"Invalid progressive budget level: {level}")

    return warnings
