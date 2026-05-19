import sys
import os
import json
import argparse
import asyncio

sys.path.insert(0, os.path.dirname(__file__))

from core import TALEEngine
from utils import estimate_think_reduction

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "worker"))
    from run_eval import run_eval as _run_eval, load_dataset as _load_dataset  # noqa: F401
except ImportError:
    pass


class TALEAdapter:
    def __init__(self, config_path: str = None):
        config = None
        if config_path:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
        self.engine = TALEEngine(config)
        self.config_path = config_path

    def get_prompt_fn(self, dataset: str = "unknown"):
        return self.engine.get_prompt_fn(dataset)

    def get_params(self, dataset: str = "unknown") -> dict:
        return {
            "max_tokens": self.engine.get_max_tokens_for_dataset(dataset),
            "temperature": self.engine.temperature,
            "use_chat": True,
            "method": "tale",
        }


def run_tale_eval(args):
    from run_eval import run_eval, load_dataset

    adapter = TALEAdapter(args.config)

    test_cases = load_dataset(args.dataset)
    if not test_cases:
        print(f"No test cases found in {args.dataset}")
        sys.exit(1)

    dataset = os.path.splitext(os.path.basename(args.dataset))[0]
    prompt_fn = adapter.get_prompt_fn(dataset)
    params = adapter.get_params(dataset)

    print(f"Dataset: {args.dataset} ({len(test_cases)} cases)")
    print(f"Method: tale, mode: {adapter.engine.mode}")
    print(f"Budget for {dataset}: {adapter.engine.get_budget_for_dataset(dataset)} tokens")
    print(f"Max tokens: {params['max_tokens']}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    summary = asyncio.run(run_eval(
        test_cases=test_cases,
        vllm_url=args.vllm_url,
        concurrency=args.concurrency,
        output_path=args.output,
        resume=not args.no_resume,
        method="tale",
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
              f"avg={stats['avg_time_ms']:.0f}ms think={stats['avg_think_tokens']:.0f} "
              f"reduction={reduction['reduction_pct']}%")

    summary_path = args.output.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="TALE Evaluation Runner")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://localhost:8000"))
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.config is None:
        args.config = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

    run_tale_eval(args)


if __name__ == "__main__":
    main()
