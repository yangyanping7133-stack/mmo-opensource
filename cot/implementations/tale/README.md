# TALE — Token-Budget-Aware LLM Reasoning

**论文**: [arXiv:2412.18547](https://arxiv.org/abs/2412.18547) (ACL 2025 Findings)
**开源参考**: https://github.com/GeniusHTX/TALE

## 核心方法

TALE 通过动态估计 token 预算并在 prompt 中注入预算约束，引导模型生成更短的推理过程。

### 关键步骤
1. **难度分类**: 根据数据集将问题分为 easy/medium/hard
2. **预算估计**: 为每个难度级别预设 token 预算
3. **Prompt 注入**: 在 prompt 中注入预算约束
4. **推理生成**: 模型在预算约束下生成更简洁的推理

### 预算分档

| 难度 | 数据集 | 默认预算 | 最大预算 | Baseline Think Tokens |
|------|--------|---------|---------|---------------------|
| easy | GSM8K, AMC | 800 | 1500 | 1144 / 4240 |
| medium | GPQA | 500 | 800 | 651 |
| hard | MATH-500, AIME | 2000 | 4000 | 2800 / 7986 |

### 部署模式
- **preset**: 按数据集预设预算（免训练，即插即用）
- **dynamic**: 基于问题复杂度微调预算（未来扩展）

## 文件结构

```
implementations/tale/
├── src/
│   ├── __init__.py    — 模块导出
│   ├── core.py        — 核心引擎: TALEEngine
│   ├── adapter.py     — 评测框架适配: TALEAdapter
│   └── utils.py       — 工具函数
├── config.yaml        — 配置参数
├── paper_info.yaml    — 论文信息
└── README.md          — 本文件
```

## 使用方式

```python
from implementations.tale.src import TALEEngine

engine = TALEEngine()
messages, budget = engine.build_messages(question, dataset="GSM8K")
prompt_text, budget = engine.build_prompt_text(question, dataset="MATH500")
```

### 运行评测

```bash
python -m implementations.tale.src.adapter \
    --dataset benchmarks/datasets/eval/gsm8k.json \
    --output results/tale/gsm8k_tale.jsonl \
    --vllm-url http://localhost:8000 \
    --concurrency 256
```
