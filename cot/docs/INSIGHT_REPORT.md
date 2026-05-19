# CoT Compression Insight Report v2 (2026 Papers Only)

> Generated: 2026-05-15
> Focus: Papers published January 2026 onwards, specifically relevant to Qwen3-32B + vLLM-Ascend
> Filter: Training-free methods prioritized; RL-based methods included for comparison

---

## Executive Summary

搜索到 **22 篇 2026 年论文**，按方法论分为 5 大类：

| 类别 | 论文数 | 最佳候选 | 训练要求 |
|------|--------|----------|----------|
| **A. 推理时压缩 (Inference-Time)** | 6 | CRISP, SAT, Hint Tuning, Whisper | 免训练 |
| **B. RL 训练压缩** | 7 | ExpThink, LEAD, SmartThinker, STACK | 需RL训练 |
| **C. 早期退出 (Early Exit)** | 3 | EAT, RCPD, Dynamic Early Exit | 免训练 |
| **D. 理论/度量** | 3 | DTR (Think Deep), CIB, Shorter but Trustworthy | 免训练/分析 |
| **E. 架构创新** | 3 | Draft-Thinking, ASC, MemoSight | 需SFT/修改 |

**关键发现**: DTR (Deep-Thinking Token Ratio) 不是压缩方法，而是**推理质量度量 + 采样策略**。对项目有重大参考价值。

---

## Tier 1: Top Candidates for cot-compress (Training-Free)

### 1. CRISP — Compressing Redundancy in CoT via Intrinsic Saliency Pruning
- **ArXiv**: [2604.17297](https://arxiv.org/abs/2604.17297) (Apr 2026)
- **Venue**: ACL 2026 Findings
- **Method**: 利用 `</think` 终止 token 的注意力模式作为信息锚点，识别 CoT 中的冗余部分并原子级压缩
- **Results**: **50-60% token 减少，精度不降**
- **Training-Free**: ✅ (推理时方法)
- **Code**: Open-sourced
- **适用性分析**:
  - ⭐ **高度适用**: 纯推理时方法，零训练成本
  - 核心洞察：`</think` token 的注意力分布能区分关键推理 vs 冗余
  - 原子级操作（不是粗粒度截断），保留逻辑连贯性
  - 跨模型验证（多种 backbone + 数学数据集）
  - **风险**: 需要 vLLM 暴露注意力权重 — 需验证 vLLM-Ascend 兼容性
- **优先级**: **P0 — 首选实现目标**

### 2. SAT — Stepwise Adaptive Thinking
- **ArXiv**: [2604.07922](https://arxiv.org/abs/2604.07922) (Apr 2026)
- **Venue**: ACL 2026 Main Conference
- **Method**: 将推理建模为有限状态机 (FSM)，4 种思考模式 (Slow/Normal/Fast/Skip)，用轻量 PRM 动态切换
- **Results**: **最高 40% token 减少，精度持平或提升**
- **Training-Free**: ✅ (推理时方法，使用预训练 PRM)
- **适用性分析**:
  - ⭐ **高度适用**: Step-level 难度感知剪枝
  - 9 个 LRM × 7 个 benchmark 全面验证
  - FSM 模式优雅：简单题跳步，难题保深度
  - **风险**: 需要预训练的 Process Reward Model (PRM)
- **优先级**: **P0 — 并行首选**

### 3. Hint Tuning — Less Data Makes Better Reasoners
- **ArXiv**: [2605.08665](https://arxiv.org/abs/2605.08665) (May 2026)
- **Method**: 用 instruct model 作为难度探针，自动构建 No-Hint/Sparse-Hint/Full-Hint 三档训练数据
- **Results**: **24-66% token 减少（均值 31.5%），精度竞争性保持**
- **Training**: SFT (仅需 1K 自标注样本)
- **适用性分析**:
  - ⭐ **高适用性**: 支持 Qwen3-Thinking 32B，仅 1K 样本
  - 核心洞察：instruct model = 天然难度探针
  - 三档难度分类：No-Hint (直接答) → Sparse-Hint → Full-Hint
  - **风险**: 需要 SFT，但训练成本极低
- **优先级**: **P1**

### 4. Whisper — Black-box Persuasive Prompting
- **ArXiv**: [2510.10528](https://arxiv.org/abs/2510.10528) (Jan 2026 update)
- **Method**: 迭代优化"说服性 prompt"，引导模型生成更简洁的推理
- **Results**: **~40% token 减少，精度不降；Qwen3 系列验证；GSM8K 简单题 3x 缩减**
- **Training-Free**: ✅ (纯 prompt 方法)
- **适用性分析**:
  - ⭐ **高度适用**: 黑盒方法，零侵入性
  - 已验证 Qwen3 系列兼容性
  - 迭代式 prompt 优化框架
  - **风险**: 效果可能因数据集差异波动
- **优先级**: **P1 — 与我们 concise prompt 实验类似但更系统化**

### 5. ASC — Activation-Steered Compression
- **ArXiv**: [2507.04742](https://arxiv.org/abs/2507.04742) (Jul 2025, indexed 2026)
- **Method**: 提取 verbose ↔ concise 的 steering vector，推理时注入到隐藏表示中
- **Results**: **最高 67.43% CoT 缩减，精度不降；MATH500/GSM8K 验证；7B/8B/32B 均有效**
- **Training-Free**: ✅ (仅需 100 个配对样本提取 steering vector)
- **Code**: https://github.com/ArminAzizi98/ASC
- **适用性分析**:
  - ⭐ **极高适用性**: 32B 已验证，MATH500/GSM8K 对齐我们评测集
  - 100 个样本即可提取 vector
  - 2.73x 端到端加速 (8B model)
  - **风险**: 需要修改推理流程注入 steering vector — 需验证 vLLM-Ascend 兼容性
- **优先级**: **P0 — 与 CRISP 并列首选**

---

## Tier 2: High-Value RL-Based Methods (Training Required)

> 这些方法需要 RL 训练，不适合直接使用，但其设计思想可借鉴到 training-free 方案。

### 6. ExpThink — Experience-Guided RL for Adaptive CoT Compression
- **ArXiv**: [2605.07501](https://arxiv.org/abs/2605.07501) (May 2026)
- **Results**: **77% 平均长度缩减 + 精度提升**
- **Key Insight**: experience-guided reward shaping — 追踪每题最短正确解，三档奖励
- **借鉴价值**: 三档奖励思想可用于 prompt 设计 (concise/normal/detailed)

### 7. LEAD — Length-Efficient Adaptive and Dynamic Reasoning
- **ArXiv**: [2605.09806](https://arxiv.org/abs/2605.09806) (May 2026)
- **Results**: 最高精度 + Accuracy-Efficiency Score
- **Key Insight**: 动态校准 correctness-efficiency trade-off + per-problem adaptive target length
- **借鉴价值**: per-problem target length 思想 → 可用于 budget-aware prompt

### 8. SmartThinker — Progressive CoT Length Calibration
- **ArXiv**: [2603.08000](https://arxiv.org/abs/2603.08000) (Mar 2026)
- **Results**: **52.5% 压缩 + 精度提升 + AIME25 上 16.6% 精度提升**
- **Key Insight**: 动态估计最优长度 + 动态调整奖励系数
- **借鉴价值**: "最优长度估计"思想 → prompt 中嵌入 target token count

### 9. STACK — State-Aware Reasoning Compression with Knowledge Guidance
- **ArXiv**: [2604.09150](https://arxiv.org/abs/2604.09150) (Apr 2026)
- **Results**: **59.9% 平均长度缩减 + 4.8% 精度提升**
- **Key Insight**: step-wise 压缩 + RAG 辅助 + answer-convergence 早期停止
- **借鉴价值**: answer-convergence early stopping 可直接用于推理时

### 10. ICR — Implicit Compression Regularization
- **ArXiv**: [2605.07316](https://arxiv.org/abs/2605.07316) (May 2026)
- **Key Insight**: 当 length-accuracy 负相关时 = overthinking，用最短正确解作为隐式压缩信号
- **借鉴价值**: length-accuracy correlation 作为 overthinking 检测器

---

## Tier 3: Early Exit Methods (Training-Free)

### 11. EAT — Entropy After `</think`
- **ArXiv**: [2509.26522](https://arxiv.org/abs/2509.26522) (Apr 2026 update)
- **Method**: 监控 `</think` 后 token 的 entropy 变化，当 entropy 稳定后提前退出
- **Results**: **12-22% token 减少，精度不降**
- **Training-Free**: ✅
- **Black-box**: ✅ (可用 proxy model)
- **适用性**: 中等 — 压缩幅度不够大（目标 ≥30%）

### 12. RCPD — Reasoning Completion Point Detector
- **ArXiv**: [2508.17627](https://arxiv.org/abs/2508.17627) (Jan 2026 update)
- **Method**: 监控终止 token 的 rank dynamics，检测推理完成点
- **Results**: **最高 44% token 减少，精度不降**
- **Training-Free**: ✅
- **适用性**: ⭐ 高 — 在 Qwen3 和 DeepSeek-R1 上验证

### 13. Dynamic Early Exit in Reasoning Models
- **ArXiv**: [2504.15895](https://arxiv.org/abs/2504.15895) (Sep 2025, indexed 2026)
- **Method**: 在推理转换点监控模型行为，高置信度时动态终止
- **Results**: **19.1%-80.1% CoT 长度缩减 + 0.3%-5.0% 精度提升**
- **Training-Free**: ✅
- **11 个 LRM × 10 个 benchmark** 全面验证
- **适用性**: ⭐⭐ 极高 — 跨模型验证最广

---

## Tier 4: Theoretical/Metric Contributions

### 14. DTR — Think Deep, Not Just Long (Deep-Thinking Token Ratio)
- **ArXiv**: [2602.13517](https://arxiv.org/abs/2602.13517) (Feb 2026)
- **这不是压缩方法！** 这是一个**推理质量度量 + 采样策略**
- **Core Concept**: Deep-thinking tokens = 内部预测在深层经历显著修正的 token
  - Deep-thinking ratio (DTR) = deep-thinking tokens 占总 token 的比例
  - DTR 与 accuracy 正相关（比 raw token count 和 confidence 都好）
- **Application**: Think@n — 优先采样 DTR 高的样本，early reject 低 DTR 前缀
- **对项目的价值**:
  - ⭐ **极高参考价值**: 可作为我们实验的**质量度量标准**
  - 验证任何压缩方法的 DTR 是否保持/提升
  - Think@n 策略可用于我们的 multi-sample 推理
  - **不直接解决压缩问题，但提供了科学的评估框架**
- **优先级**: P1 — 作为评估指标集成

### 15. CIB — Conditional Information Bottleneck for Reasoning
- **ArXiv**: [2603.08462](https://arxiv.org/abs/2603.08462) (Mar 2026)
- **理论贡献**: 将高效推理形式化为有损压缩问题，统一 budget forcing
- **Key Result**: semantic prior (token surprisal) > naive length penalty
- **借鉴价值**: 理论框架解释为什么 CRISP/ASC 等 semantic 方法优于 length truncation

### 16. Shorter, but Still Trustworthy?
- **ArXiv**: [2604.04120](https://arxiv.org/abs/2604.04120) (Apr 2026)
- **贡献**: 首个系统性研究 CoT 压缩对 trustworthiness (安全/幻觉/多语言) 的影响
- **Key Finding**: 不同压缩方法有不同的 trustworthiness 退化特征
- **对项目价值**: 评估指标参考 — 压缩方法不仅要保精度，还要保可信度

---

## Tier 5: Architecture/SFT Innovations

### 17. Draft-Thinking
- **ArXiv**: [2603.00578](https://arxiv.org/abs/2603.00578) (Feb 2026)
- **Method**: 学习 concise draft-style 推理结构 + progressive curriculum learning
- **Results**: MATH500 上 82.6% 预算缩减，仅 2.6% 精度降
- **需 SFT + RL**: 不适合直接使用

### 18. MemoSight — Unified Context Compression + Multi-Token Prediction
- **ArXiv**: [2604.14889](https://arxiv.org/abs/2604.14889) (Apr 2026)
- **Results**: 66% KV cache 缩减 + 1.56x 加速
- **需架构修改**: 不适合 vLLM-Ascend

---

## Comparison Matrix: Training-Free Methods

| Method | Token Reduction | Accuracy | GSM8K | MATH500 | GPQA | AIME | Code | vLLM Compatible |
|--------|----------------|----------|-------|---------|------|------|------|-----------------|
| **CRISP** | 50-60% | 不降 | ✅ | ✅ | ❓ | ❓ | ✅ | 需验证(attention) |
| **SAT** | ≤40% | 持平/提升 | ✅ | ✅ | ✅ | ✅ | ❓ | 需验证(PRM) |
| **ASC** | ≤67% | 不降 | ✅ | ✅ | ❓ | ❓ | ✅ | 需验证(steering) |
| **Whisper** | ~40% | 不降 | ✅ | ✅ | ❓ | ❓ | ❌ | ✅ (pure prompt) |
| **RCPD** | ≤44% | 不降 | ❓ | ❓ | ✅ | ✅ | ❌ | 需验证 |
| **Dyn. Early Exit** | 19-80% | 提升 | ✅ | ✅ | ✅ | ✅ | ❌ | 需验证 |
| **EAT** | 12-22% | 不降 | ❓ | ✅ | ❓ | ✅ | ✅ | 需验证 |
| **Don't Overthink** | ≤40% | 持平/提升 | ✅ | ✅ | ✅ | ❓ | ❌ | ✅ |

---

## Recommendation for cot-compress

### Primary Implementation Priority (Training-Free)

1. **CRISP** (P0) — 最高压缩比 + 精度不降，ACL 2026，有代码
2. **ASC** (P0) — 67% 压缩，32B 已验证，100 样本即可用
3. **SAT** (P0) — ACL 2026 Main，step-level 粒度最精细

### Secondary (Low-effort, Quick Wins)

4. **Whisper** (P1) — 纯 prompt 方法，可立即验证
5. **Dynamic Early Exit** (P1) — 最广泛验证的方法
6. **Hint Tuning** (P1) — 1K 样本 SFT 即可

### Evaluation Enhancement

7. **DTR** (P1) — 集成为质量评估指标，不作为压缩方法
8. **Trustworthiness** (P2) — 参考 Shorter but Trustworthy 的评估维度

### RL-Based (Phase 3 考虑)

9. **ExpThink** — 77% 压缩 + 精度提升，最强 RL 方法
10. **SmartThinker** — 52.5% 压缩 + AIME25 显著提升

---

## Papers Not Recommended

- **MemoSight**: 架构修改，不兼容 vLLM-Ascend
- **Draft-Thinking**: 需要 SFT+RL pipeline
- **CIB**: 理论框架，不是实现方案
- **Neural Garbage Collection**: KV cache 管理，非 CoT 压缩
- **SketchThinker-R1**: 多模态，不适用纯文本推理

---

*End of INSIGHT_REPORT_v2.md*
