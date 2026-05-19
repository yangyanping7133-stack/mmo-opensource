import yaml
import os
import re
from typing import Optional


CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


class TALEEngine:
    BUDGET_SYSTEM_PROMPT = (
        "You are a precise problem solver. Solve the problem step by step. "
        "You should limit your reasoning to the token budget provided. "
        "Focus on the key reasoning steps and avoid unnecessary elaboration, "
        "repetition, backtracking, or exploring alternatives. "
        "When you reach the answer, output it clearly."
    )

    BUDGET_USER_TEMPLATE = (
        "{question}\n\n"
        "Please solve this problem concisely. You have approximately {budget} tokens "
        "for your thinking process. Focus on the key reasoning steps and avoid "
        "unnecessary elaboration."
    )

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()
        self.config = config
        params = config.get("params", {})
        self.mode = params.get("mode", "preset")
        self.max_tokens = params.get("max_tokens", 8192)
        self.temperature = params.get("temperature", 0.6)
        self.use_system_prompt = params.get("use_system_prompt", True)

        self.token_budget = config.get("token_budget", {})
        self.dataset_mapping = config.get("dataset_mapping", {})
        self.baseline_think_tokens = config.get("baseline_think_tokens", {})
        self.prompt_template = config.get("prompt_template", "")
        self.budget_inject_patterns = config.get("budget_inject_patterns", [])

    def classify_difficulty(self, dataset: str) -> str:
        return self.dataset_mapping.get(dataset, "medium")

    def estimate_token_budget(self, question: str, dataset: str) -> int:
        difficulty = self.classify_difficulty(dataset)
        budget_config = self.token_budget.get(difficulty, {})
        default_budget = budget_config.get("default", 1500)
        max_budget = budget_config.get("max", 3000)

        if self.mode == "preset":
            return default_budget

        q_len = len(question)
        has_complex_math = bool(
            re.search(
                r"\\frac|\\sqrt|\\int|\\sum|\\prod|\\alpha|\\beta|\\theta",
                question,
            )
        )
        has_multi_step = any(
            kw in question.lower()
            for kw in ["respectively", "each of", "how many", "find all", "prove"]
        )

        complexity_score = 0
        if q_len > 500:
            complexity_score += 2
        elif q_len > 200:
            complexity_score += 1
        if has_complex_math:
            complexity_score += 2
        if has_multi_step:
            complexity_score += 1

        if difficulty == "easy":
            if complexity_score >= 3:
                budget = int(default_budget * 1.3)
            else:
                budget = default_budget
        elif difficulty == "hard":
            if complexity_score <= 1:
                budget = int(default_budget * 0.7)
            else:
                budget = default_budget
        else:
            budget = default_budget

        return min(budget, max_budget)

    def inject_budget_prompt(self, question: str, budget: int) -> str:
        for pattern_config in self.budget_inject_patterns:
            pattern = pattern_config.get("pattern", "")
            replacement = pattern_config.get("replacement", "").format(budget=budget)
            if pattern in question:
                return question.replace(pattern, replacement)

        return self.BUDGET_USER_TEMPLATE.format(question=question, budget=budget)

    def build_messages(self, question: str, dataset: str) -> tuple:
        budget = self.estimate_token_budget(question, dataset)
        prompt_text = self.inject_budget_prompt(question, budget)

        messages = []
        if self.use_system_prompt:
            messages.append({"role": "system", "content": self.BUDGET_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": prompt_text})

        return messages, budget

    def build_prompt_text(self, question: str, dataset: str) -> tuple:
        messages, budget = self.build_messages(question, dataset)

        parts = []
        for msg in messages:
            if msg["role"] == "system":
                parts.append(f"Instructions: {msg['content']}\n")
            elif msg["role"] == "user":
                parts.append(f"Problem: {msg['content']}")

        prompt = "\n".join(parts)
        return prompt, budget

    def get_prompt_fn(self, dataset: str = "unknown"):
        def prompt_fn(question: str) -> str:
            prompt, _ = self.build_prompt_text(question, dataset)
            return prompt

        return prompt_fn

    def get_budget_for_dataset(self, dataset: str) -> int:
        budget_config = self.token_budget.get(
            self.classify_difficulty(dataset), {}
        )
        return budget_config.get("default", 1500)

    def get_max_tokens_for_dataset(self, dataset: str) -> int:
        budget = self.get_budget_for_dataset(dataset)
        return min(budget + 512, self.max_tokens)
