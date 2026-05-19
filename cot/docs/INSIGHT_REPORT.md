# CoT Compression Insight Report v2 (2026 Papers Only)

> Generated: 2026-05-15
> Focus: Papers published January 2026 onwards, specifically relevant to Qwen3-32B + vLLM-Ascend
> Filter: Training-free methods prioritized; RL-based methods included for comparison

---

## Executive Summary

Found **22 papers from 2026**, categorized into 5 groups by methodology:

| Category | Papers | Top Candidates | Training Required |
|----------|--------|----------------|-------------------|
| **A. Inference-Time Compression** | 6 | CRISP, SAT, Hint Tuning, Whisper | Training-free |
| **B. RL-Based Compression** | 7 | ExpThink, LEAD, SmartThinker, STACK | Requires RL training |
| **C. Early Exit** | 3 | EAT, RCPD, Dynamic Early Exit | Training-free |
| **D. Theory/Metrics** | 3 | DTR (Think Deep), CIB, Shorter but Trustworthy | Training-free/Analysis |
| **E. Architecture Innovation** | 3 | Draft-Thinking, ASC, MemoSight | Requires SFT/Modification |

**Key Finding**: DTR (Deep-Thinking Token Ratio) is not a compression method, but a **reasoning quality metric + sampling strategy**. Extremely valuable as a reference for the project.

---

## Tier 1: Top Candidates for cot-compress (Training-Free)

### 1. CRISP — Compressing Redundancy in CoT via Intrinsic Saliency Pruning
- **ArXiv**: [2604.17297](https://arxiv.org/abs/2604.17297) (Apr 2026)
- **Venue**: ACL 2026 Findings
- **Method**: Uses the attention pattern of `</think` termination tokens as information anchors to identify redundant parts in CoT and perform atom-level compression
- **Results**: **50-60% token reduction with no accuracy drop**
- **Training-Free**: Yes (inference-time method)
- **Code**: Open-sourced
- **Applicability Analysis**:
  - Highly applicable: Pure inference-time method, zero training cost
  - Core insight: `</think` token attention distribution distinguishes key reasoning vs. redundancy
  - Atom-level operations (not coarse truncation), preserves logical coherence
  - Cross-model validation (multiple backbones + math datasets)
  - **Risk**: Requires vLLM to expose attention weights — need to verify vLLM-Ascend compatibility
- **Priority**: **P0 — Primary implementation target**

### 2. SAT — Stepwise Adaptive Thinking
- **ArXiv**: [2604.07922](https://arxiv.org/abs/2604.07922) (Apr 2026)
- **Venue**: ACL 2026 Main Conference
- **Method**: Models reasoning as a finite state machine (FSM) with 4 thinking modes (Slow/Normal/Fast/Skip), dynamically switching via a lightweight PRM
- **Results**: **Up to 40% token reduction with maintained or improved accuracy**
- **Training-Free**: Yes (inference-time method, uses pre-trained PRM)
- **Applicability Analysis**:
  - Highly applicable: Step-level difficulty-aware pruning
  - 9 LRM x 7 benchmarks comprehensive validation
  - FSM design is elegant: simple problems skip steps, hard problems preserve depth
  - **Risk**: Requires pre-trained Process Reward Model (PRM)
- **Priority**: **P0 — Parallel primary target**

### 3. Hint Tuning — Less Data Makes Better Reasoners
- **ArXiv**: [2605.08665](https://arxiv.org/abs/2605.08665) (May 2026)
- **Method**: Uses instruct model as a difficulty probe to automatically construct 3-tier training data: No-Hint/Sparse-Hint/Full-Hint
- **Results**: **24-66% token reduction (mean 31.5%), competitive accuracy maintained**
- **Training**: SFT (only 1K self-annotated samples needed)
- **Applicability Analysis**:
  - High applicability: Supports Qwen3-Thinking 32B, only 1K samples
  - Core insight: instruct model = natural difficulty probe
  - 3-tier difficulty classification: No-Hint (direct answer) -> Sparse-Hint -> Full-Hint
  - **Risk**: Requires SFT, but training cost is extremely low
- **Priority**: **P1**

### 4. Whisper — Black-box Persuasive Prompting
- **ArXiv**: [2510.10528](https://arxiv.org/abs/2510.10528) (Jan 2026 update)
- **Method**: Iteratively optimizes "persuasive prompts" to guide the model toward generating more concise reasoning
- **Results**: **~40% token reduction with no accuracy drop; Qwen3 series verified; GSM8K simple problems 3x reduction**
- **Training-Free**: Yes (pure prompt method)
- **Applicability Analysis**:
  - Highly applicable: Black-box method, zero invasiveness
  - Qwen3 series compatibility verified
  - Iterative prompt optimization framework
  - **Risk**: Effectiveness may vary across datasets
- **Priority**: **P1 — Similar to our concise prompt experiments but more systematic**

### 5. ASC — Activation-Steered Compression
- **ArXiv**: [2507.04742](https://arxiv.org/abs/2507.04742) (Jul 2025, indexed 2026)
- **Method**: Extracts verbose vs. concise steering vectors and injects them into hidden representations at inference time
- **Results**: **Up to 67.43% CoT reduction with no accuracy drop; MATH500/GSM8K verified; effective on 7B/8B/32B**
- **Training-Free**: Yes (only needs 100 paired samples to extract steering vector)
- **Code**: https://github.com/ArminAzizi98/ASC
- **Applicability Analysis**:
  - Extremely high applicability: 32B already verified, MATH500/GSM8K aligns with our eval suite
  - 100 samples sufficient to extract vector
  - 2.73x end-to-end speedup (8B model)
  - **Risk**: Requires modifying inference pipeline to inject steering vector — need to verify vLLM-Ascend compatibility
- **Priority**: **P0 — Tied with CRISP as top choice**

---

## Tier 2: High-Value RL-Based Methods (Training Required)

> These methods require RL training and are not suitable for direct use, but their design ideas can inform training-free approaches.

### 6. ExpThink — Experience-Guided RL for Adaptive CoT Compression
- **ArXiv**: [2605.07501](https://arxiv.org/abs/2605.07501) (May 2026)
- **Results**: **77% average length reduction + accuracy improvement**
- **Key Insight**: Experience-guided reward shaping — tracks shortest correct solution per problem, 3-tier reward
- **Reference Value**: 3-tier reward concept applicable to prompt design (concise/normal/detailed)

### 7. LEAD — Length-Efficient Adaptive and Dynamic Reasoning
- **ArXiv**: [2605.09806](https://arxiv.org/abs/2605.09806) (May 2026)
- **Results**: Highest accuracy + Accuracy-Efficiency Score
- **Key Insight**: Dynamic calibration of correctness-efficiency trade-off + per-problem adaptive target length
- **Reference Value**: Per-problem target length concept -> usable in budget-aware prompts

### 8. SmartThinker — Progressive CoT Length Calibration
- **ArXiv**: [2603.08000](https://arxiv.org/abs/2603.08000) (Mar 2026)
- **Results**: **52.5% compression + accuracy improvement + 16.6% accuracy gain on AIME25**
- **Key Insight**: Dynamic estimation of optimal length + dynamic reward coefficient adjustment
- **Reference Value**: "Optimal length estimation" concept -> embed target token count in prompts

### 9. STACK — State-Aware Reasoning Compression with Knowledge Guidance
- **ArXiv**: [2604.09150](https://arxiv.org/abs/2604.09150) (Apr 2026)
- **Results**: **59.9% average length reduction + 4.8% accuracy improvement**
- **Key Insight**: Step-wise compression + RAG assistance + answer-convergence early stopping
- **Reference Value**: Answer-convergence early stopping directly applicable at inference time

### 10. ICR — Implicit Compression Regularization
- **ArXiv**: [2605.07316](https://arxiv.org/abs/2605.07316) (May 2026)
- **Key Insight**: When length-accuracy correlation is negative = overthinking; use shortest correct solution as implicit compression signal
- **Reference Value**: Length-accuracy correlation as overthinking detector

---

## Tier 3: Early Exit Methods (Training-Free)

### 11. EAT — Entropy After `</think`
- **ArXiv**: [2509.26522](https://arxiv.org/abs/2509.26522) (Apr 2026 update)
- **Method**: Monitors entropy changes in tokens after `</think`, exiting early when entropy stabilizes
- **Results**: **12-22% token reduction with no accuracy drop**
- **Training-Free**: Yes
- **Black-box**: Yes (can use proxy model)
- **Applicability**: Moderate — compression range insufficient (target is >=30%)

### 12. RCPD — Reasoning Completion Point Detector
- **ArXiv**: [2508.17627](https://arxiv.org/abs/2508.17627) (Jan 2026 update)
- **Method**: Monitors rank dynamics of termination tokens to detect reasoning completion points
- **Results**: **Up to 44% token reduction with no accuracy drop**
- **Training-Free**: Yes
- **Applicability**: High — verified on Qwen3 and DeepSeek-R1

### 13. Dynamic Early Exit in Reasoning Models
- **ArXiv**: [2504.15895](https://arxiv.org/abs/2504.15895) (Sep 2025, indexed 2026)
- **Method**: Monitors model behavior at reasoning transition points, dynamically terminating when confidence is high
- **Results**: **19.1%-80.1% CoT length reduction + 0.3%-5.0% accuracy improvement**
- **Training-Free**: Yes
- **11 LRM x 10 benchmarks** comprehensive validation
- **Applicability**: Extremely high — broadest cross-model validation

---

## Tier 4: Theoretical/Metric Contributions

### 14. DTR — Think Deep, Not Just Long (Deep-Thinking Token Ratio)
- **ArXiv**: [2602.13517](https://arxiv.org/abs/2602.13517) (Feb 2026)
- **This is NOT a compression method!** It is a **reasoning quality metric + sampling strategy**
- **Core Concept**: Deep-thinking tokens = tokens whose internal predictions undergo significant revision in deep layers
  - Deep-thinking ratio (DTR) = proportion of deep-thinking tokens to total tokens
  - DTR positively correlates with accuracy (better than raw token count and confidence)
- **Application**: Think@n — prioritize sampling high-DTR responses, early reject low-DTR prefixes
- **Value for the Project**:
  - Extremely high reference value: can serve as a **quality metric** for our experiments
  - Verify whether any compression method maintains/improves DTR
  - Think@n strategy applicable to our multi-sample inference
  - Does not directly solve compression, but provides a scientific evaluation framework
- **Priority**: P1 — integrate as evaluation metric

### 15. CIB — Conditional Information Bottleneck for Reasoning
- **ArXiv**: [2603.08462](https://arxiv.org/abs/2603.08462) (Mar 2026)
- **Theoretical Contribution**: Formalizes efficient reasoning as a lossy compression problem, unifying budget forcing
- **Key Result**: Semantic prior (token surprisal) > naive length penalty
- **Reference Value**: Theoretical framework explaining why semantic methods like CRISP/ASC outperform length truncation

### 16. Shorter, but Still Trustworthy?
- **ArXiv**: [2604.04120](https://arxiv.org/abs/2604.04120) (Apr 2026)
- **Contribution**: First systematic study on the impact of CoT compression on trustworthiness (safety/hallucination/multilingual)
- **Key Finding**: Different compression methods exhibit different trustworthiness degradation characteristics
- **Project Value**: Evaluation metric reference — compression methods must preserve not only accuracy but also trustworthiness

---

## Tier 5: Architecture/SFT Innovations

### 17. Draft-Thinking
- **ArXiv**: [2603.00578](https://arxiv.org/abs/2603.00578) (Feb 2026)
- **Method**: Learns concise draft-style reasoning structure + progressive curriculum learning
- **Results**: 82.6% budget reduction on MATH500 with only 2.6% accuracy drop
- **Requires SFT + RL**: Not suitable for direct use

### 18. MemoSight — Unified Context Compression + Multi-Token Prediction
- **ArXiv**: [2604.14889](https://arxiv.org/abs/2604.14889) (Apr 2026)
- **Results**: 66% KV cache reduction + 1.56x speedup
- **Requires Architecture Modification**: Not compatible with vLLM-Ascend

---

## Comparison Matrix: Training-Free Methods

| Method | Token Reduction | Accuracy | GSM8K | MATH500 | GPQA | AIME | Code | vLLM Compatible |
|--------|----------------|----------|-------|---------|------|------|------|-----------------|
| **CRISP** | 50-60% | No drop | Yes | Yes | ? | ? | Yes | Needs verification (attention) |
| **SAT** | <=40% | Maintained/Improved | Yes | Yes | Yes | Yes | ? | Needs verification (PRM) |
| **ASC** | <=67% | No drop | Yes | Yes | ? | ? | Yes | Needs verification (steering) |
| **Whisper** | ~40% | No drop | Yes | Yes | ? | ? | No | Yes (pure prompt) |
| **RCPD** | <=44% | No drop | ? | ? | Yes | Yes | No | Needs verification |
| **Dyn. Early Exit** | 19-80% | Improved | Yes | Yes | Yes | Yes | No | Needs verification |
| **EAT** | 12-22% | No drop | ? | Yes | ? | Yes | Yes | Needs verification |
| **Don't Overthink** | <=40% | Maintained/Improved | Yes | Yes | Yes | ? | No | Yes |

---

## Recommendation for cot-compress

### Primary Implementation Priority (Training-Free)

1. **CRISP** (P0) — Highest compression ratio + no accuracy drop, ACL 2026, code available
2. **ASC** (P0) — 67% compression, 32B verified, 100 samples to deploy
3. **SAT** (P0) — ACL 2026 Main, finest step-level granularity

### Secondary (Low-effort, Quick Wins)

4. **Whisper** (P1) — Pure prompt method, immediately verifiable
5. **Dynamic Early Exit** (P1) — Most broadly validated method
6. **Hint Tuning** (P1) — 1K sample SFT sufficient

### Evaluation Enhancement

7. **DTR** (P1) — Integrate as quality evaluation metric, not as a compression method
8. **Trustworthiness** (P2) — Reference evaluation dimensions from "Shorter but Still Trustworthy"

### RL-Based (Phase 3 Consideration)

9. **ExpThink** — 77% compression + accuracy improvement, strongest RL method
10. **SmartThinker** — 52.5% compression + significant AIME25 improvement

---

## Papers Not Recommended

- **MemoSight**: Architecture modification, incompatible with vLLM-Ascend
- **Draft-Thinking**: Requires SFT+RL pipeline
- **CIB**: Theoretical framework, not an implementation approach
- **Neural Garbage Collection**: KV cache management, not CoT compression
- **SketchThinker-R1**: Multimodal, not applicable to text-only reasoning

---

*End of INSIGHT_REPORT_v2.md*
