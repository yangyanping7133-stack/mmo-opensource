import sys
import os
import json
import argparse
import asyncio

sys.path.insert(0, os.path.dirname(__file__))

from core import DEEREngine
from utils import estimate_think_reduction

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "worker"))
    from run_eval import run_eval as _run_eval, load_dataset as _load_dataset  # noqa: F401
except ImportError:
    pass


class DEERAdapter:
    def __init__(self, config_path: str = None):
        config = None
        if config_path:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
        self.engine = DEEREngine(config)
        self.config_path = config_path

    def get_prompt_fn(self, dataset: str = "unknown"):
        return self.engine.get_prompt_fn(dataset)

    def get_params(self, dataset: str = "unknown") -> dict:
        return self.engine.get_params(dataset)


def run_deer_eval(args):
    try:
        from run_eval import run_eval, load_dataset

        adapter = DEERAdapter(args.config)

        test_cases = load_dataset(args.dataset)
        if not test_cases:
            print(f"No test cases found in {args.dataset}")
            sys.exit(1)

        dataset = os.path.splitext(os.path.basename(args.dataset))[0]
        prompt_fn = adapter.get_prompt_fn(dataset)
        params = adapter.get_params(dataset)

        print(f"Dataset: {args.dataset} ({len(test_cases)} cases)")
        print(f"Method: deer, threshold: {params['threshold']}, max_steps: {params['max_steps']}")
        print(f"Max tokens: {params['max_tokens']}")

        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        summary = asyncio.run(run_eval(
            test_cases=test_cases,
            vllm_url=args.vllm_url,
            concurrency=args.concurrency,
            output_path=args.output,
            resume=not args.no_resume,
            method="deer",
            max_tokens=params["max_tokens"],
            prompt_fn=prompt_fn,
            temperature=params["temperature"],
        ))

        print(f"\n{'='*60}")
        print(f"SUMMARY: {summary['method']}")
        print(f"  Overall: {summary['correct']}/{summary['total']} = {summary['accuracy']}%")
        print(f"  Avg latency: {summary['avg_time_ms']:.0f}ms")
        print(f"  Avg tokens: {summary['avg_completion_tokens']:.0f}")
        print(f"  Avg think tokens: {summary['avg_thinking_tokens']:.0f}")
        for ds, stats in summary.get("by_dataset", {}).items():
            reduction = estimate_think_reduction(ds, round(stats["avg_thinking_tokens"]))
            print(f"  {ds:15s}: {stats['correct']}/{stats['total']} = {stats['accuracy']}%  "
                  f"avg={stats['avg_time_ms']:.0f}ms think={stats['avg_thinking_tokens']:.0f} "
                  f"reduction={reduction['reduction_pct']}%")

        summary_path = args.output.replace(".jsonl", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to {summary_path}")

    except ImportError:
        _run_standalone_eval(args)


def _run_standalone_eval(args):
    import aiohttp
    import time

    adapter = DEERAdapter(args.config)
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    params = adapter.get_params(dataset_name)

    with open(args.dataset) as f:
        test_cases = json.load(f)

    print(f"Dataset: {args.dataset} ({len(test_cases)} cases)")
    print(f"Method: deer (standalone mode)")
    print(f"Params: threshold={params['threshold']}, max_steps={params['max_steps']}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    completed = set()
    if not args.no_resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed.add(r.get("question_id", r.get("id", "")))
        print(f"Resuming: {len(completed)} already done")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def eval_one(session, case):
        qid = case.get("question_id", case.get("id", ""))
        question = case.get("question", case.get("problem", ""))
        answer = case.get("answer", case.get("ground_truth", ""))

        async with semaphore:
            messages = adapter.engine.build_initial_messages(question, dataset_name)
            payload = {
                "model": os.environ.get("MODEL_NAME", "Qwen/Qwen3-32B"),
                "messages": messages,
                "max_tokens": params["max_tokens"],
                "temperature": params["temperature"],
                "chat_template_kwargs": {"enable_thinking": True},
            }

            t0 = time.time()
            async with session.post(
                f"{args.vllm_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                result = await resp.json()
            elapsed = time.time() - t0

            choice = result.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            reasoning = choice.get("message", {}).get("reasoning_content", "")
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)

            think_tokens = len(reasoning.split()) if reasoning else 0

            wait_breakpoints = adapter.engine.detect_wait_breakpoint(reasoning)
            confidence = adapter.engine.evaluate_confidence(content)

            record = {
                "question_id": qid,
                "question": question,
                "ground_truth": answer,
                "response": content,
                "reasoning": reasoning,
                "confidence": confidence,
                "wait_breakpoints": len(wait_breakpoints),
                "completion_tokens": completion_tokens,
                "thinking_tokens_est": think_tokens,
                "total_time": round(elapsed * 1000),
                "dataset": dataset_name,
                "method": "deer",
            }

            return record

    async def main():
        pending = []
        for case in test_cases:
            qid = case.get("question_id", case.get("id", ""))
            if qid not in completed:
                pending.append(case)

        print(f"Running {len(pending)} evaluations...")

        async with aiohttp.ClientSession() as session:
            tasks = [eval_one(session, case) for case in pending]
            results = []
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                try:
                    result = await coro
                    results.append(result)
                    with open(args.output, "a") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if (i + 1) % 50 == 0:
                        print(f"  Progress: {i + 1}/{len(pending)}")
                except Exception as e:
                    print(f"  Error: {e}")

        return results

    results = asyncio.run(main())

    if results:
        total = len(results)
        print(f"\n{'='*60}")
        print(f"Completed: {total} evaluations")
        print(f"Avg think tokens: {sum(r['thinking_tokens_est'] for r in results) / total:.0f}")
        print(f"Avg latency: {sum(r['total_time'] for r in results) / total:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="DEER Evaluation Runner")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://localhost:8000"))
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.config is None:
        args.config = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

    run_deer_eval(args)


if __name__ == "__main__":
    main()
