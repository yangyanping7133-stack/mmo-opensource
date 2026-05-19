#!/usr/bin/env python3
"""Standardized evaluation runner for CoT compression experiments.

Usage:
    python3 run_eval.py --config config.json --output results.jsonl [--vllm-url http://localhost:8000]
"""

import argparse
import json
import os
import re
import sys
import time
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


def load_dataset(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "test_cases" in data:
            return data["test_cases"]
        items = []
        for ds in data.get("datasets", []):
            ds_file = ds.get("file")
            if ds_file:
                ds_path = Path(path).parent / ds_file
                if ds_path.exists():
                    with open(ds_path) as df:
                        items.extend(json.load(df))
        return items
    return []


def _extract_boxed_content(text: str) -> str:
    """Extract content from \\boxed{...} using bracket balancing (handles deep nesting)."""
    # Try \boxed{...} format (last occurrence)
    idx = text.rfind('\\boxed{')
    if idx != -1:
        start = idx + 7  # len('\\boxed{')
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                if depth == 0:
                    return text[start:i].strip()
                depth -= 1
    # Try \boxed(...) format
    idx = text.rfind('\\boxed(')
    if idx != -1:
        start = idx + 7  # len('\\boxed(')
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                if depth == 0:
                    return text[start:i].strip()
                depth -= 1
    return ""


def _extract_math_from_text(text: str) -> str:
    """Shared logic for extracting math answers from text (both model output and ground truth)."""
    # 1. Try \boxed{...} with bracket balancing
    boxed = _extract_boxed_content(text)
    if boxed:
        return _normalize_number_str(boxed)

    # 2. Try #### separator (GSM8K style)
    hash_match = re.search(r'####\s*(-?[\d,\.]+)', text)
    if hash_match:
        return hash_match.group(1).replace(",", "")

    # 3. Try "the answer is" pattern
    ans_match = re.search(
        r'(?:the answer is|answer is|=)\s*\$?(.+?)\$?(?:\.|\.\s*$|\n|$)',
        text, re.IGNORECASE | re.MULTILINE,
    )
    if ans_match:
        candidate = ans_match.group(1).strip().rstrip(".")
        if re.match(r'^[\-+]?\d[\d,]*\.?\d*$', candidate):
            return _normalize_number_str(candidate)
        nums = re.findall(r'[\-+]?\d+\.?\d*', candidate)
        if nums:
            return _normalize_number_str(nums[-1])
        return candidate.strip()

    # 4. Fallback: try LaTeX-aware extraction and comma-number parsing
    # Bug B: If text contains LaTeX commands, normalize and return directly
    if re.search(r'\\frac|\\sqrt|\\pi|\\cdot|\\times|\\pm', text):
        return _normalize_for_judge(text)

    # Bug A: Try stripping commas/punctuation and parsing as a single number
    cleaned = re.sub(r'[,\\!\s]', '', text)
    try:
        val = float(cleaned)
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, TypeError):
        pass

    # Last resort: last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return _normalize_number_str(numbers[-1]) if numbers else ""


def _normalize_for_judge(s: str) -> str:
    """Normalize LaTeX math expressions for model-based comparison."""
    s = s.strip().strip("$").strip()
    s = re.sub(r'\\left\s*|\\right\s*', '', s)
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    frac_match = re.match(r'^\\frac\{([^{}]+)\}\{([^{}]+)\}$', s)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            val = num / den
            if val == int(val):
                return str(int(val))
            return str(val)
        except (ValueError, ZeroDivisionError):
            pass
    s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'\1/\2', s)
    s = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', s)
    s = re.sub(r'\\pi', 'pi', s)
    s = re.sub(r'\\cdot', '*', s)
    s = re.sub(r'\\times', '*', s)
    s = re.sub(r'\\pm', '+-', s)
    s = re.sub(r'[\\{}]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _normalize_number_str(s: str) -> str:
    """Normalize a number string: remove commas, convert to int if whole."""
    s = s.strip().replace(",", "")
    try:
        val = float(s)
        if val == int(val) and "." not in s:
            return str(int(val))
        return s
    except (ValueError, TypeError):
        return s.strip()


def extract_answer(text: str, qtype: str) -> str:
    if qtype == "mc":
        match = re.search(r'\b([A-D])\b', text[-200:])
        return match.group(1) if match else ""
    return _extract_math_from_text(text)


def extract_ground_truth(answer: str, qtype: str) -> str:
    if qtype == "mc":
        match = re.search(r'\b([A-D])\b', answer)
        return match.group(1) if match else answer.strip()[0].upper() if answer else ""
    return _extract_math_from_text(answer)


_judge_client = None


async def _judge_with_model(predicted: str, ground_truth: str) -> bool:
    """Use the LLM as judge for all math comparisons (async)."""
    global _judge_client
    if not predicted or not ground_truth:
        return False
    import httpx
    if _judge_client is None or _judge_client.is_closed:
        _judge_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_connections=32),
        )
    prompt = """/no_think
Are these two mathematical answers equivalent? Answer ONLY "YES" or "NO".

Ground Truth: """
    prompt += ground_truth + "\nPredicted: " + predicted
    try:
        resp = await _judge_client.post(
            os.environ.get("VLLM_URL", "http://localhost:8000") + "/v1/chat/completions",
            json={
                "model": os.environ.get("MODEL_NAME", "Qwen/Qwen3-32B"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.0,
            },
            headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'YOUR_API_KEY')}"},
        )
        content = resp.json()["choices"][0]["message"]["content"]
        if "</think" in content:
            content = content.split("</think", 1)[-1].lstrip(">").strip()
        return content.strip().upper().startswith("YES")
    except Exception:
        return False


async def judge(predicted: str, ground_truth: str, qtype: str) -> bool:
    if qtype == "mc":
        return predicted.strip().upper() == ground_truth.strip().upper()
    return await _judge_with_model(predicted, ground_truth)


async def call_vllm(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    model: str = "Qwen3-32B",
    max_tokens: int = 8192,
    timeout: int = 600,
    temperature: float = 0.6,
    use_chat: bool = True,
    stop: list = None,
    extra_body: dict = None,
) -> dict:
    t0 = time.time()
    try:
        if use_chat:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1.0,
            }
            if stop:
                payload["stop"] = stop
            if extra_body:
                payload.update(extra_body)
            endpoint = f"{url}/v1/chat/completions"
        else:
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1.0,
            }
            if stop:
                payload["stop"] = stop
            if extra_body:
                payload.update(extra_body)
            endpoint = f"{url}/v1/completions"

        headers = {"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'YOUR_API_KEY')}"}
        async with session.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            body = await resp.json()
            if "error" in body:
                return {"error": body["error"], "elapsed": time.time() - t0}
            choice = body["choices"][0]
            usage = body.get("usage", {})
            if use_chat:
                text = choice.get("message", {}).get("content", "")
            else:
                text = choice.get("text", "")
            finish = choice.get("finish_reason", "")
            return {
                "text": text,
                "finish_reason": finish,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "elapsed": time.time() - t0,
            }
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - t0}


async def run_eval(
    test_cases: list,
    vllm_url: str,
    concurrency: int,
    output_path: str,
    resume: bool = True,
    method: str = "baseline",
    max_tokens: int = 8192,
    prompt_fn: Optional[callable] = None,
    model_name: str = os.environ.get("MODEL_NAME", "Qwen/Qwen3-32B"),
    temperature: float = 0.6,
    use_chat: bool = True,
) -> dict:
    done_ids = set()
    results = []

    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            # Use dict to deduplicate: keep only the last result for each id
            resume_dict = {}
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    resume_dict[r.get("id", "")] = r
            results = list(resume_dict.values())
            done_ids = set(resume_dict.keys())
        print(f"[resume] {len(done_ids)} existing results loaded (deduplicated)")

    pending = [tc for tc in test_cases if tc.get("id", "") not in done_ids]
    print(f"[eval] {len(pending)} pending / {len(test_cases)} total, concurrency={concurrency}")

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def eval_one(tc):
            async with sem:
                tc_id = tc.get("id", "unknown")
                question = tc.get("question", "")
                ground_truth_raw = tc.get("answer", "")
                qtype = tc.get("type", tc.get("qtype", "math"))
                dataset = tc.get("dataset", tc.get("dataset_key", "unknown"))

                prompt = prompt_fn(question) if prompt_fn else question

                resp = await call_vllm(
                    session, vllm_url, prompt,
                    model=model_name, max_tokens=max_tokens,
                    temperature=temperature, use_chat=use_chat,
                )

                if "error" in resp:
                    err_result = {
                        "id": tc_id, "dataset": dataset, "qtype": qtype,
                        "question": question[:200], "ground_truth": ground_truth_raw,
                        "method": method, "error": str(resp["error"]),
                        "total_time": round(resp.get("elapsed", 0) * 1000, 1),
                        "judge_correct": False,
                    }
                    with open(output_path, "a") as f:
                        f.write(json.dumps(err_result, ensure_ascii=False) + "\n")
                    return err_result

                full_text = resp.get("text", "")
                predicted = extract_answer(full_text, qtype)
                gt = extract_ground_truth(ground_truth_raw, qtype)
                correct = await judge(predicted, gt, qtype)
                if qtype == "mc":
                    judge_method = "exact_match" if correct else "exact_match_failed"
                else:
                    judge_method = "model" if correct else "model_failed"

                think_tokens = 0
                answer_only = full_text
                # Match <think...>content</think...> — the > after think/closing is optional
                # (Qwen3 uses <think\n...\n</think\n> format without closing >)
                think_match = re.search(r"<think\s*>?(.*?)</think\s*>?", full_text, re.DOTALL)
                if think_match:
                    think_tokens = len(think_match.group(1).split())
                    end_pos = think_match.end()
                    answer_only = full_text[end_pos:].strip()
                else:
                    think_match = re.search(r"<think\s*>?(.*?)(?:</think\s*>?|$)", full_text, re.DOTALL)
                    if think_match:
                        think_tokens = len(think_match.group(1).split())

                result = {
                    "id": tc_id,
                    "dataset": dataset,
                    "qtype": qtype,
                    "question": question[:200],
                    "ground_truth": gt,
                    "predicted": predicted,
                    "method": method,
                    "total_time": round(resp.get("elapsed", 0) * 1000, 1),
                    "ttft": 0,
                    "completion_tokens": resp.get("completion_tokens", 0),
                    "prompt_tokens": resp.get("prompt_tokens", 0),
                    "thinking_tokens_est": think_tokens,
                    "answer_content": full_text[:500],
                    "judge_correct": correct,
                    "judge_method": judge_method,
                    "stop_reason": resp.get("finish_reason", ""),
                }

                with open(output_path, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                return result

        completed = 0
        tasks = []
        for tc in pending:
            tasks.append(eval_one(tc))

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                correct_so_far = sum(1 for r in results if r.get("judge_correct"))
                print(f"[progress] {completed}/{len(pending)} ({completed/len(pending)*100:.0f}%) | accuracy={correct_so_far}/{len(results)}={correct_so_far/len(results)*100:.1f}%")

    total = len(results)
    correct = sum(1 for r in results if r.get("judge_correct"))
    avg_time = sum(r.get("total_time", 0) for r in results) / max(total, 1)
    avg_tokens = sum(r.get("completion_tokens", 0) for r in results) / max(total, 1)
    avg_think = sum(r.get("thinking_tokens_est", 0) for r in results) / max(total, 1)
    errors = sum(1 for r in results if r.get("error"))

    by_dataset = {}
    for r in results:
        ds = r.get("dataset", "unknown")
        if ds not in by_dataset:
            by_dataset[ds] = {"total": 0, "correct": 0, "total_time": 0, "total_tokens": 0, "total_think": 0}
        by_dataset[ds]["total"] += 1
        if r.get("judge_correct"):
            by_dataset[ds]["correct"] += 1
        by_dataset[ds]["total_time"] += r.get("total_time", 0)
        by_dataset[ds]["total_tokens"] += r.get("completion_tokens", 0)
        by_dataset[ds]["total_think"] += r.get("thinking_tokens_est", 0)

    for ds, stats in by_dataset.items():
        stats["accuracy"] = round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] else 0
        stats["avg_time_ms"] = round(stats["total_time"] / stats["total"], 1) if stats["total"] else 0
        stats["avg_tokens"] = round(stats["total_tokens"] / stats["total"], 0) if stats["total"] else 0
        stats["avg_think_tokens"] = round(stats["total_think"] / stats["total"], 0) if stats["total"] else 0

    # Close async judge client
    global _judge_client
    if _judge_client and not _judge_client.is_closed:
        await _judge_client.aclose()
        _judge_client = None

    summary = {
        "method": method,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1) if total else 0,
        "avg_time_ms": round(avg_time, 1),
        "avg_completion_tokens": round(avg_tokens, 0),
        "avg_thinking_tokens": round(avg_think, 0),
        "errors": errors,
        "by_dataset": by_dataset,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://localhost:8000"), help="vLLM server URL")
    parser.add_argument("--concurrency", type=int, default=32, help="Concurrent requests")
    parser.add_argument("--method", default="baseline", help="Method name")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens per response")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary from existing results")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-32B"), help="Model name for vLLM")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument("--use-completions", action="store_true", help="Use /v1/completions instead of chat")
    args = parser.parse_args()

    test_cases = load_dataset(args.dataset)
    if not test_cases:
        print(f"No test cases found in {args.dataset}")
        sys.exit(1)

    if args.summary_only:
        results = []
        if os.path.exists(args.output):
            with open(args.output) as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        correct = sum(1 for r in results if r.get("judge_correct"))
        print(f"Results: {correct}/{len(results)} = {correct/len(results)*100:.1f}%" if results else "No results")
        for ds_name in sorted(set(r.get("dataset", "") for r in results)):
            ds_items = [r for r in results if r.get("dataset") == ds_name]
            ds_correct = sum(1 for r in ds_items if r.get("judge_correct"))
            avg_t = sum(r.get("total_time", 0) for r in ds_items) / len(ds_items)
            print(f"  {ds_name:15s}: {ds_correct}/{len(ds_items)} = {ds_correct/len(ds_items)*100:.1f}%  avg={avg_t:.0f}ms")
        return

    print(f"Dataset: {args.dataset} ({len(test_cases)} cases)")
    print(f"Method: {args.method}")
    print(f"Output: {args.output}")
    print(f"Concurrency: {args.concurrency}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    summary = asyncio.run(run_eval(
        test_cases=test_cases,
        vllm_url=args.vllm_url,
        concurrency=args.concurrency,
        output_path=args.output,
        resume=not args.no_resume,
        method=args.method,
        max_tokens=args.max_tokens,
        model_name=args.model,
        temperature=args.temperature,
        use_chat=not args.use_completions,
    ))

    print(f"\n{'='*60}")
    print(f"SUMMARY: {summary['method']}")
    print(f"  Overall: {summary['correct']}/{summary['total']} = {summary['accuracy']}%")
    print(f"  Avg latency: {summary['avg_time_ms']}ms")
    print(f"  Avg tokens: {summary['avg_completion_tokens']}")
    for ds, stats in summary.get("by_dataset", {}).items():
        print(f"  {ds:15s}: {stats['correct']}/{stats['total']} = {stats['accuracy']}%  avg={stats['avg_time_ms']}ms")

    summary_path = args.output.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
