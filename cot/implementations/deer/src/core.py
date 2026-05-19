import yaml
import os
import re
from typing import Optional


CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


class DEEREngine:
    WAIT_PATTERN = re.compile(
        r"\b(?:Wait|wait|WAIT|Hmm|hmm|But wait|but wait)\b[,:\.\!\?\;\s]"
    )

    BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}")

    FINAL_ANSWER_PATTERNS = [
        re.compile(r"(?:the answer is|final answer[:\s]*)\s*(.+?)(?:\.|$)", re.IGNORECASE),
        re.compile(r"(?:therefore|thus|so|hence)[,\s]+(?:the answer is\s+)?(.+?)(?:\.|$)", re.IGNORECASE),
        re.compile(r"=\s*\\boxed\{([^}]+)\}"),
        re.compile(r"\\boxed\{([^}]+)\}"),
    ]

    TRIAL_ANSWER_PROMPT = (
        "\n\nBased on your reasoning so far, what is your answer? "
        "Please provide your final answer clearly. "
        "If you are confident, state 'The answer is <answer>'. "
        "If you are unsure, state 'I need to think more'."
    )

    SYSTEM_PROMPT = (
        "You are a precise problem solver. Think step by step. "
        "When you reach the answer, output it clearly."
    )

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()
        self.config = config
        params = config.get("params", {})
        self.threshold = params.get("threshold", 0.95)
        self.max_steps = params.get("max_steps", 4)
        self.think_ratio = params.get("think_ratio", 0.8)
        self.max_tokens = params.get("max_tokens", 8192)
        self.temperature = params.get("temperature", 0.6)
        self.per_dataset = config.get("per_dataset", {})
        self.baseline_think_tokens = config.get("baseline_think_tokens", {})
        self.progressive_budget = config.get("progressive_budget", {})

    def get_per_dataset_params(self, dataset: str) -> dict:
        overrides = self.per_dataset.get(dataset, {})
        return {
            "threshold": overrides.get("threshold", self.threshold),
            "max_steps": overrides.get("max_steps", self.max_steps),
            "think_ratio": self.think_ratio,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def build_initial_messages(self, question: str, dataset: str = "unknown") -> list:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return messages

    def detect_wait_breakpoint(self, text: str) -> list:
        positions = []
        for m in self.WAIT_PATTERN.finditer(text):
            positions.append({
                "start": m.start(),
                "end": m.end(),
                "match": m.group(),
                "context": text[max(0, m.start() - 50):m.end() + 50],
            })
        return positions

    def build_trial_answer_prompt(self, think_content: str) -> str:
        return think_content + self.TRIAL_ANSWER_PROMPT

    def evaluate_confidence(self, response: str) -> float:
        if not response or not response.strip():
            return 0.0

        score = 0.0

        boxed_match = self.BOXED_PATTERN.search(response)
        if boxed_match:
            score += 0.4
            answer_text = boxed_match.group(1)
            if answer_text.strip():
                score += 0.1

        for pattern in self.FINAL_ANSWER_PATTERNS:
            match = pattern.search(response)
            if match:
                score += 0.3
                answer_text = match.group(1)
                if answer_text and answer_text.strip().lower() not in ("", "i need to think more", "not sure"):
                    score += 0.1
                break

        if "I need to think more" in response or "not sure" in response.lower():
            score -= 0.3

        negative_indicators = ["maybe", "might be", "could be", "possibly", "i think", "perhaps"]
        for indicator in negative_indicators:
            if indicator in response.lower():
                score -= 0.05

        confidence_indicators = [
            "the answer is", "therefore", "thus", "clearly",
            "definitely", "exactly", "must be",
        ]
        for indicator in confidence_indicators:
            if indicator in response.lower():
                score += 0.05

        if re.search(r"^-?\d+\.?\d*$", response.strip().split("\n")[-1].strip()):
            score += 0.2

        return max(0.0, min(1.0, score))

    def should_exit(self, confidence: float, step: int, max_steps: int = None) -> bool:
        if max_steps is None:
            max_steps = self.max_steps
        if step >= max_steps:
            return True
        return confidence >= self.threshold

    def get_progressive_budgets(self, dataset: str) -> list:
        if not self.progressive_budget.get("enabled", False):
            return [1.0]

        baseline = self.baseline_think_tokens.get(dataset, 2609)
        levels = self.progressive_budget.get("levels", [0.5, 0.8, 1.0])
        return [int(baseline * level) for level in levels]

    def get_prompt_fn(self, dataset: str = "unknown"):
        def prompt_fn(question: str) -> str:
            messages = self.build_initial_messages(question, dataset)
            parts = []
            for msg in messages:
                if msg["role"] == "system":
                    parts.append(f"Instructions: {msg['content']}\n")
                elif msg["role"] == "user":
                    parts.append(f"Problem: {msg['content']}")
            return "\n".join(parts)

        return prompt_fn

    def get_params(self, dataset: str = "unknown") -> dict:
        ds_params = self.get_per_dataset_params(dataset)
        return {
            "max_tokens": ds_params["max_tokens"],
            "temperature": ds_params["temperature"],
            "use_chat": True,
            "method": "deer",
            "threshold": ds_params["threshold"],
            "max_steps": ds_params["max_steps"],
            "think_ratio": ds_params["think_ratio"],
        }
