import sys
import os
import re
import json
import asyncio
import time
import argparse
import httpx
from collections import Counter

# Note: requires baee/utils.py in the evaluation framework path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baee", "src"))
from utils import (
    extract_math_answer,
    extract_choice_answer,
    count_thinking_tokens,
    format_summary,
    judge_answer,
    BASELINE_THINK_TOKENS,
)

API_KEY = os.environ.get("VLLM_API_KEY", "YOUR_API_KEY")
DEFAULT_K = 8
DEFAULT_M = 3
TEMPERATURE = 0.6
MAX_TOKENS = 32768

MODEL = None

DATASET_PARAMS = {
    "gsm8k": {"k": 8, "m": 3},
    "math500": {"k": 16, "m": 5},
    "gpqa": {"k": 16, "m": 5},
    "amc": {"k": 8, "m": 3},
    "aime": {"k": 8, "m": 3},
}

DATASET_MAP = {
    "gsm8k": "./benchmarks/datasets/eval/gsm8k.json",
    "math500": "./benchmarks/datasets/eval/math500.json",
    "gpqa": "./benchmarks/datasets/eval/gpqa.json",
    "amc": "./benchmarks/datasets/eval/amc.json",
    "aime": "./benchmarks/datasets/eval/aime.json",
}


async def get_model_id(client, base_url):
    global MODEL
    if MODEL:
        return MODEL
    resp = await client.get(
        f"{base_url}/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    data = resp.json()
    models = data.get("data", [])
    if models:
        MODEL = models[0]["id"]
    else:
        MODEL = "default"
    print(f"Model: {MODEL}")
    return MODEL


async def api_call(client, base_url, question):
    model = await get_model_id(client, base_url)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    resp = await client.post(
        f"{base_url}/v1/chat/completions", json=payload, headers=headers
    )
    elapsed = time.perf_counter() - t0
    data = resp.json()
    if "choices" not in data:
        raise RuntimeError(f"API error: {json.dumps(data)[:300]}")
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return {
        "content": content,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "elapsed": round(elapsed, 2),
    }


async def run_one(client, base_url, question, qtype, k=DEFAULT_K, m=DEFAULT_M):
    tasks = [asyncio.ensure_future(api_call(client, base_url, question)) for _ in range(k)]

    results = []
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if not t.exception():
                results.append(t.result())

        while len(results) < m and pending:
            done2, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED, timeout=600
            )
            if not done2:
                break
            for t in done2:
                if not t.exception():
                    results.append(t.result())
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.sleep(0.1)

    if not results:
        return "", [], []

    if qtype == "mc":
        answers = [extract_choice_answer(r["content"]) for r in results]
    else:
        answers = [extract_math_answer(r["content"]) for r in results]

    final_answer = majority_vote(answers)
    return final_answer, results, answers


def majority_vote(answers):
    valid = [a for a in answers if a]
    if not valid:
        return ""
    counts = Counter(valid)
    return counts.most_common(1)[0][0]


async def run_dataset(
    dataset_path, output_path, concurrency, vllm_url, summary_path=None,
    override_k=None, override_m=None, no_resume=False,
):
    with open(dataset_path) as f:
        cases = json.load(f)

    for case in cases:
        if "dataset" not in case:
            cid = case.get("id", "")
            for prefix in ["gsm8k", "math500", "gpqa", "amc", "aime"]:
                if cid.startswith(prefix):
                    case["dataset"] = prefix
                    break
            else:
                case["dataset"] = os.path.basename(dataset_path).replace(".json", "")

    ds_name = os.path.basename(dataset_path).replace(".json", "")
    ds_params = DATASET_PARAMS.get(ds_name, {"k": DEFAULT_K, "m": DEFAULT_M})
    k = override_k if override_k is not None else ds_params["k"]
    m = override_m if override_m is not None else ds_params["m"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    completed_ids = set()
    if not no_resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        completed_ids.add(json.loads(line).get("id", ""))
                    except json.JSONDecodeError:
                        pass
    elif no_resume and os.path.exists(output_path):
        os.remove(output_path)

    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(900.0, connect=30.0)
    limits = httpx.Limits(max_connections=max(concurrency * k, 60), max_keepalive_connections=max(concurrency * k, 40))

    base_url = vllm_url.rstrip("/")

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await get_model_id(client, base_url)

        async def process_one(case, idx):
            sid = case["id"]
            if sid in completed_ids:
                return None

            async with semaphore:
                question = case["question"]
                gt = case["answer"]
                ds = case["dataset"]
                qtype = case.get("type", "math")

                t0 = time.perf_counter()
                try:
                    final_answer, sample_results, sample_answers = await run_one(
                        client, base_url, question, qtype, k=k, m=m
                    )
                except Exception as e:
                    return {
                        "id": sid,
                        "dataset": ds,
                        "question": question[:200],
                        "ground_truth": gt,
                        "method": "dontoverthink",
                        "qtype": qtype,
                        "error": str(e),
                        "total_time": round(time.perf_counter() - t0, 2),
                        "judge_correct": False,
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "thinking_tokens_est": 0,
                    }

                total_time = time.perf_counter() - t0

                if not sample_results:
                    return {
                        "id": sid,
                        "index": idx,
                        "dataset": ds,
                        "question": question[:200],
                        "ground_truth": gt,
                        "method": "dontoverthink",
                        "qtype": qtype,
                        "error": "no successful samples",
                        "total_time": round(total_time, 2),
                        "judge_correct": False,
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "thinking_tokens_est": 0,
                    }

                if qtype == "mc":
                    predicted_clean = final_answer.strip() if final_answer else ""
                    gt_clean = gt.strip().upper()
                    correct = predicted_clean.upper() == gt_clean
                else:
                    correct, predicted_clean = await judge_answer(final_answer, gt)

                avg_completion = round(
                    sum(r["completion_tokens"] for r in sample_results)
                    / len(sample_results)
                )
                avg_think = round(
                    sum(count_thinking_tokens(r["content"]) for r in sample_results)
                    / len(sample_results)
                )

                result = {
                    "id": sid,
                    "index": idx,
                    "dataset": ds,
                    "question": question[:200],
                    "ground_truth": gt,
                    "method": "dontoverthink",
                    "qtype": qtype,
                    "total_time": round(total_time, 2),
                    "k": k,
                    "m": m,
                    "temperature": TEMPERATURE,
                    "final_answer": predicted_clean,
                    "judge_correct": correct,
                    "completion_tokens": avg_completion,
                    "prompt_tokens": sample_results[0].get("prompt_tokens", 0),
                    "thinking_tokens_est": avg_think,
                    "sample_answers": sample_answers,
                    "sample_think_tokens": [
                        count_thinking_tokens(r["content"]) for r in sample_results
                    ],
                    "sample_times": [r["elapsed"] for r in sample_results],
                }
                return result

        tasks = []
        for idx, case in enumerate(cases):
            if case["id"] not in completed_ids:
                tasks.append(process_one(case, idx))

        ds_name = os.path.basename(dataset_path).replace(".json", "")
        print(
            f"Don't Overthink short-m@k: k={k}, m={m}, temp={TEMPERATURE}"
        )
        print(
            f"Dataset: {ds_name} | {len(tasks)} cases (skipped {len(cases) - len(tasks)} completed)"
        )
        print(
            f"Outer concurrency: {concurrency} (max API calls: {concurrency * k})"
        )
        print()

        done_count = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is None:
                continue
            done_count += 1

            with open(output_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            judge_str = "OK" if result["judge_correct"] else "MISS"
            if "error" in result:
                print(
                    f"  [{done_count}/{len(tasks)}] id={result['id']} "
                    f"ERROR: {result['error'][:80]}",
                    flush=True,
                )
            else:
                print(
                    f"  [{done_count}/{len(tasks)}] id={result['id']} "
                    f"time={result['total_time']:.1f}s "
                    f"tok={result['completion_tokens']} "
                    f"think={result['thinking_tokens_est']} "
                    f"answer={result['final_answer'][:30]} "
                    f"samples={result['sample_answers']} "
                    f"{judge_str}",
                    flush=True,
                )

    all_results = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    summary = format_summary(all_results, method="dontoverthink")
    summary["params"] = {
        "k": k,
        "m": m,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "outer_concurrency": concurrency,
    }

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {summary['method']}")
    print(
        f"  Overall: {summary['correct']}/{summary['total']} = {summary['accuracy']}%"
    )
    print(f"  Avg latency: {summary['avg_time_ms']:.1f}s")
    print(f"  Avg completion tokens: {summary['avg_completion_tokens']}")
    print(f"  Avg thinking tokens: {summary['avg_thinking_tokens']}")

    baseline_acc = {
        "gsm8k": 96.0,
        "math500": 88.8,
        "gpqa": 94.0,
        "amc": 47.5,
        "aime": 20.0,
    }
    for ds, stats in summary.get("by_dataset", {}).items():
        bl_acc = baseline_acc.get(ds, 0)
        bl_think = BASELINE_THINK_TOKENS.get(ds, 0)
        red = stats.get("think_reduction_pct", 0)
        delta = stats["accuracy"] - bl_acc
        print(
            f"  {ds:15s}: {stats['correct']}/{stats['total']} = {stats['accuracy']}% "
            f"(baseline={bl_acc}%, delta={delta:+.1f}%) "
            f"think={stats['avg_thinking_tokens']} (baseline={bl_think}) "
            f"reduction={red}%"
        )

    if summary_path:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to {summary_path}")

    return summary


async def main():
    parser = argparse.ArgumentParser(description="Don't Overthink short-m@k evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a single dataset JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run all 5 datasets sequentially",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dontoverthink",
        help="Directory for output files (used with --all-datasets)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Outer concurrency (each case sends K requests)",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=os.environ.get("VLLM_URL", "http://localhost:8000"),
        help="vLLM server URL",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override k (number of samples) for all datasets",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="Override m (majority vote threshold) for all datasets",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore/remove existing output file",
    )
    args = parser.parse_args()

    if args.all_datasets:
        for ds_name in ["gsm8k", "math500", "gpqa", "amc", "aime"]:
            ds_path = DATASET_MAP.get(ds_name)
            if not ds_path or not os.path.exists(ds_path):
                print(f"SKIP {ds_name}: file not found at {ds_path}")
                continue
            out_path = os.path.join(
                args.output_dir, f"{ds_name}_dontoverthink.jsonl"
            )
            sum_path = os.path.join(
                args.output_dir, f"{ds_name}_dontoverthink_summary.json"
            )
            print(f"\n{'#' * 60}")
            print(f"# Running {ds_name} ({ds_path})")
            print(f"# Output: {out_path}")
            print(f"{'#' * 60}\n")
            await run_dataset(
                ds_path, out_path, args.concurrency, args.vllm_url, sum_path,
                override_k=args.k, override_m=args.m, no_resume=args.no_resume,
            )
    elif args.dataset:
        out_path = args.output or os.path.join(
            args.output_dir,
            f"{os.path.basename(args.dataset).replace('.json', '')}_dontoverthink.jsonl",
        )
        sum_path = out_path.replace(".jsonl", "_summary.json")
        await run_dataset(
            args.dataset, out_path, args.concurrency, args.vllm_url, sum_path,
            override_k=args.k, override_m=args.m, no_resume=args.no_resume,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
