# ADR-002: 2026 CoT 压缩方案重新裁决

> 由 CTO-Agent 生成
> 生成时间：2026-05-15
> 前置：ADR-001 已废弃（TALE 试点精度崩溃，全部基于 2024-2025 论文）
> 本决策完全基于 INSIGHT_REPORT_v2.md 中的 2026 年论文

## Status: Proposed

## Context

### 项目背景（不变）
- **模型**: Qwen3-32B，推理框架 vLLM-Ascend（昇腾 NPU）
- **硬件**: 8 × 昇腾 YOUR_NPU_DEVICE
- **核心目标**: CoT token 减少 ≥30%，延迟降低 ≥20%
- **硬性红线**: 准确率不允许低于 baseline
- **约束**: 免训练优先（仅允许阈值/prompt调优）；可接受极低成本 SFT（≤1K样本）
- **评测规模**: 5 数据集 1270 条，128 并发

### ADR-001 回顾与教训

| 方案 | 结果 | 教训 |
|------|------|------|
| TALE | ❌ 试点精度崩溃（GPQA 60% vs 94%） | 固定 token budget 不适用于难度差异大的数据集 |
| Concise Prompt | ✅ GSM8K 10样本 90%（持平） | 简单 prompt 有效但不够系统化 |
| max_tokens 截断 | ❌ 破坏精度 | 粗粒度截断不可行 |
| DEER | 作为对比基线已实现 | confidence-based exit 不可靠 |

### 2026 论文洞察总结

22 篇 2026 年论文，关键发现：

1. **CRISP** (ACL 2026): 利用 `</think` 注意力锚点，50-60% 压缩，精度不降
2. **ASC**: Steering vector 注入，67% 压缩，32B 已验证，100 样本
3. **SAT** (ACL 2026 Main): FSM 4模式 step-level 剪枝，≤40% 压缩
4. **Whisper**: 黑盒说服性 prompt，~40% 压缩，Qwen3 系列验证
5. **DTR**: 不是压缩方法，是推理质量度量，可作评估指标

### ATAM 质量属性优先级

| 属性 | 优先级 | 目标 |
|------|--------|------|
| 准确率 | P0（一票否决） | ≥ baseline |
| Token 减少 | P1 | ≥30% |
| vLLM-Ascend 兼容 | P0 | 必须可运行 |
| 延迟降低 | P1 | ≥20% |
| 实现成本 | P1 | 免训练或极低成本 |
| 实现周期 | P2 | ≤2周 |

## Decision: 三轨并行 + 瀑布回退

### 轨道 A: ASC — Steering Vector 注入（主攻）
- **论文**: Activation-Steered Compression (arXiv 2507.04742)
- **原理**: 100 对 verbose/concise 样本 → 提取 steering vector → 推理时注入隐藏层
- **预期**: 50-67% token 减少，精度不降，端到端 2.7x 加速
- **优势**:
  - ✅ 32B 模型已验证
  - ✅ MATH500 + GSM8K 已验证（与我们评测集对齐）
  - ✅ 仅需 100 对样本，零训练
  - ✅ 代码开源 (https://github.com/ArminAzizi98/ASC)
- **风险**:
  - ⚠️ 需要修改推理流程注入 steering vector
  - ⚠️ vLLM-Ascend 是否支持 hidden state manipulation 未知
  - **缓解**: 先做 vLLM-Ascend 兼容性 PoC（Day 1）
- **Go/No-Go**: Day 1 完成 PoC，确认 steering vector 可注入 → Go

### 轨道 B: CRISP — 内在显著性剪枝（并行主攻）
- **论文**: CRISP (arXiv 2604.17297, ACL 2026 Findings)
- **原理**: 利用 `</think` 终止 token 注意力模式 → 识别冗余 → 原子级压缩
- **预期**: 50-60% token 减少，精度不降
- **优势**:
  - ✅ ACL 2026 peer-reviewed
  - ✅ 代码开源
  - ✅ 跨模型验证
  - ✅ 原子级操作保留逻辑连贯性（不是粗粒度截断）
- **风险**:
  - ⚠️ 需要访问注意力权重
  - ⚠️ vLLM-Ascend 是否暴露注意力权重未知
  - **缓解**: 先做 vLLM-Ascend 注意力访问 PoC（Day 1）
- **Go/No-Go**: Day 1 完成 PoC，确认注意力权重可获取 → Go

### 轨道 C: Whisper — 黑盒说服性 Prompt（保底 + Day 1 快速验证）
- **论文**: Whisper (arXiv 2510.10528)
- **原理**: 迭代优化"说服性 prompt"，引导模型生成更简洁推理
- **预期**: ~40% token 减少，精度不降
- **优势**:
  - ✅ 纯 prompt 方法，零侵入性，100% vLLM-Ascend 兼容
  - ✅ Qwen3 系列已验证
  - ✅ 可立即开始，Day 1 即有结果
  - ✅ 可与任何其他方法叠加
- **风险**:
  - ⚠️ 效果可能因数据集差异波动
  - ⚠️ 压缩比可能不够（目标 ≥30%，Whisper 可能卡在边界）
- **Go/No-Go**: Day 1: 50 条快速验证，token 减少 ≥25% 且精度 ≥baseline → Go

### 回退方案

| 回退级 | 方案 | 触发条件 |
|--------|------|----------|
| F1 | SAT (step-level FSM 剪枝) | 轨道 A+B 均 No-Go |
| F2 | Hint Tuning (1K SFT) | 轨道 A+B+C 均不达标 |
| F3 | Dynamic Early Exit | 所有方法不达标，取最大压缩保精度 |

### 评估增强

- **DTR 指标集成**: 对所有方法的输出计算 Deep-Thinking Token Ratio，作为质量评估维度
- **Trustworthiness 维度**: 参考 "Shorter but Still Trustworthy" 的评估框架

### 执行计划

```
Day 1:
  - ASC PoC: 验证 vLLM-Ascend steering vector 注入
  - CRISP PoC: 验证 vLLM-Ascend 注意力权重访问
  - Whisper 50条快速验证（纯 prompt，无需 PoC）

Day 2-3:
  - 主攻方向确定（基于 Day 1 PoC 结果）
  - ASC/CRISP 至少一个全量实现
  - Whisper 继续优化 prompt

Day 4-5:
  - 全量评测（128 并发，5 数据集 1270 条）
  - Baseline 完成（MATH500 从 199/500 继续 + GPQA + AMC）

Day 6-7:
  - 如需回退，启动 F1 (SAT)
  - 结果对比分析
  - 最终裁决
```

### 决策树

```
Day 1 PoC
├─ ASC ✅ + CRISP ✅ → 双轨并行，选最佳
├─ ASC ✅ + CRISP ❌ → ASC 主攻 + Whisper 保底
├─ ASC ❌ + CRISP ✅ → CRISP 主攻 + Whisper 保底
└─ ASC ❌ + CRISP ❌ → Whisper 主攻 + F1(SAT) 回退

Day 2 全量评测
├─ Token减少 ≥30% + 精度 ≥ baseline → ACCEPT ✅
├─ Token减少 ≥25% + 精度 ≥ baseline → 微调参数，重新评测
└─ 精度 < baseline → No-Go → 回退到下一方案
```

## Consequences

### 正面
- 基于最新 2026 研究，方法更成熟
- 三轨并行降低风险：至少 Whisper 纯 prompt 可立即产出
- DTR 指标提供科学评估框架
- ASC 和 CRISP 已有开源代码，实现成本低

### 负面
- ASC/CRISP 都依赖 vLLM-Ascend 的特定能力，PoC 风险
- 同时做 3 轨需要更多并行资源
- 2026 论文普遍较新，复现可能有坑

### 与 ADR-001 的关键变化
1. TALE → 被证明不可行，完全移除
2. CRISP（2026）→ 新发现，基于注意力锚点的精细剪枝
3. ASC（2026）→ 新发现，steering vector 方法，32B 已验证
4. Whisper（2026 update）→ 替代原来的 concise prompt，更系统化
5. DTR → 新增为评估指标（不是压缩方法）
6. 并发 256 → 128（硬件稳定性考虑）

---

*ADR-002 supersedes ADR-001*
