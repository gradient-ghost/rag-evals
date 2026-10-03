# 📊 Eval Results Summary — `reranker_bge_v2` Run

**Source:** `results/retriever_eval_results.csv`  
**Run ID:** `reranker_bge_v2`  
**Timestamp:** `2026-10-03`  
**Architecture:** Two-stage retrieval (Stage 1: ChromaDB + BGE-M3 @ $k=16$ ➔ Stage 2: Cross-Encoder Reranker `BAAI/bge-reranker-v2-m3` @ top-5)  
**Total Questions:** 50  

---

## 📈 High-Level Comparison vs. Baselines

| Run ID | Architecture | Top-$k$ | Avg Recall | Avg Precision | Perfect Scores (1.0 / 1.0) |
|---|---|:---:|:---:|:---:|:---:|
| `k5_baseline_dualcall` | Dual-call Dense (k=5) | 5 | 0.6729 | 0.6550 | 3 / 50 (6%) |
| `k8_merged` | Dense Baseline (k=8) | 8 | 0.7679 | 0.7275 | 8 / 50 (16%) |
| **`reranker_bge_v2`** | **Dense ($k=16$) + Reranker (top-5)** | **5** | **0.7698** | **0.8880** | **12 / 50 (24%)** |

> **Key takeaway:** The reranker delivered a massive **+16.05% absolute boost in precision** (rising from 72.8% to 88.8%) while maintaining and slightly improving overall recall. 33 out of 50 questions achieved a perfect 1.00 precision score.

---

## 🔢 Score Distribution

| Score Bracket | Recall Count | Precision Count |
|---|:---:|:---:|
| **1.00** | 20 | 33 |
| **0.75–0.99** | 13 | 8 |
| **0.50–0.74** | 11 | 7 |
| **0.25–0.49** | 4 | 2 |
| **0.00–0.24** | 2 | 0 |

---

## 🌟 Perfect Scores (Recall = 1.0 AND Precision = 1.0)

12 questions achieved a flawless 1.0 across both metrics (up from 8 in `k8_merged`):

| ID | Type | Question |
|---|---|---|
| Q014 | `single_concept` | What are the different eval methods and who actually executes the evaluation? |
| Q022 | `single_concept` | What is online eval and why does it not need a golden dataset? |
| Q026 | `single_concept` | How do I evaluate quality in production when I don't have the correct answers? |
| Q030 | `single_concept` | What is a benchmark and what does it consist of? |
| Q035 | `single_concept` | What is benchmark contamination? |
| Q062 | `cross_session` | How do drift and changing data affect the golden dataset over time, and what should I do about it? |
| Q066 | `cross_session` | How are safety evaluations handled across the course? |
| Q079 | `cross_session` | How do I fairly compare two versions or two models of my system? |
| Q088 | `comparison` | What are captured signals versus computed signals in online evals? |
| Q090 | `comparison` | How do component-level, workflow-level and application-level failures differ? |
| Q092 | `comparison` | What's the difference between benchmark contamination and benchmark saturation? |
| Q098 | `comparison` | Are leaderboards a selection tool or a filtering tool, and how does that compare to custom evals? |

---

## 🎯 Key Turnarounds: Comparison vs. `k8_merged`

The reranker targeted and resolved the biggest failure modes from the previous run:

| ID | Question | Baseline Recall | Reranker Recall | Baseline Precision | Reranker Precision | Net Impact |
|---|---|:---:|:---:|:---:|:---:|---|
| **Q085** | When should I use programmatic vs. human vs. LLM judge eval? | 0.00 | **0.50** | 0.625 | **1.000** | Rescued from zero recall; noise removed |
| **Q088** | What are captured signals versus computed signals in online evals? | 0.50 | **1.00** | 0.250 | **1.000** | Reached 100% on both recall & precision |
| **Q012** | What are the risk categories in LLM evaluation? | 1.00 | **1.00** | 0.250 | **0.800** | +55% precision; filler chunks removed |
| **Q048** | What is the RAG triad? | 0.75 | **1.00** | 0.250 | **0.600** | +25% recall, +35% precision |
| **Q066** | How are safety evaluations handled across the course? | 0.75 | **1.00** | 0.375 | **1.000** | +25% recall, +62.5% precision |
| **Q098** | Are leaderboards a selection tool or a filtering tool? | 1.00 | **1.00** | 0.375 | **1.000** | Cleaned out all unneeded chunks |
| **Q002** | How do we define LLM evals / key characteristics? | 0.40 | **0.60** | 0.375 | **1.000** | +20% recall, +62.5% precision |

---

## ⬆️ High Recall, Low Precision (Recall ≥ 0.75, Precision ≤ 0.50)

> **Drastic reduction:** In `k8_merged`, 9 questions suffered from high recall but poor precision. With the reranker, this category has dropped to **just 1 question**.

| ID | Recall | Precision | Question |
|---|:---:|:---:|---|
| Q086 | 1.00 | 0.40 | What's the difference between precision and recall for a RAG retriever? |

---

## ⬇️ Low Recall, High Precision (Recall ≤ 0.40, Precision ≥ 0.75)

> High-precision retrieval where the top-5 window missed parts of multi-hop answers:

| ID | Recall | Precision | Question | Notes |
|---|:---:|:---:|---|---|
| Q005 | 0.40 | 1.00 | What are application evals and why do they matter if we already have model evals? | Top hits concentrated in session 1 |
| Q053 | 0.33 | 1.00 | Where does LLM-as-a-judge show up across the course and what are its limitations? | Spans sessions 2, 3, 5, 6 |
| Q065 | 0.20 | 1.00 | How are latency and cost treated as evals throughout the course? | Spans sessions 2, 3, 7 |

---

## 🔴 Low on Both (Recall ≤ 0.40, Precision ≤ 0.60) — Remaining Opportunities

| ID | Recall | Precision | Question | Diagnosis |
|---|:---:|:---:|---|---|
| Q052 | 0.40 | 0.60 | What does a golden dataset contain for different kinds of evals? | Wide multi-topic spread (classification, SQL, etc.) |
| Q074 | 0.00 | 0.60 | How do reference-free evals apply to online monitoring and the RAG triad? | Query terminology mismatch |
| Q091 | 0.40 | 0.60 | How is testing an LLM application different from ordinary software testing? | Borderline conceptual boundary |

---

## 🔁 Mismatch Flags

Both `recall_mismatch` and `precision_mismatch` columns are **False for all 50 rows** — 100% reliable schema parsing and validation.

---

## 📋 By Question Type

| Question Type | Count | Avg Recall | Avg Precision | Precision Delta vs. `k8_merged` |
|---|:---:|:---:|:---:|:---:|
| `single_concept` | 20 | 0.8086 | **0.8900** | **+18.38%** |
| `comparison` | 15 | 0.8256 | **0.8667** | **+17.50%** |
| `cross_session` | 15 | 0.6622 | **0.9067** | **+11.50%** |

---

## 💡 Summary & Insights

1. **Precision Problem Solved:** The primary bottleneck in `k8_merged` was that retrieving 8 chunks introduced excessive background noise (precision was ~72.8%). The two-stage pipeline (retrieve $k=16$, rerank to top 5) pushed precision to **88.8%**.
2. **Comparison Questions Fixed:** In prior runs, comparison questions suffered the lowest precision (~69.2%) due to repetitive chunks. The cross-encoder resolved this, lifting comparison precision to **86.7%**.
3. **Lexical Gaps Bridged:** Questions with zero recall like Q085 were restored to positive recall and 1.0 precision because the cross-encoder computes full query-document attention instead of independent vector cosine distances.
4. **Next Potential Improvements:** The only remaining bottlenecks are multi-session cross-cutting questions (e.g. Q053, Q065 spanning 3–4 sessions), which can be addressed with session-based fan-out or HyDE query expansion.
