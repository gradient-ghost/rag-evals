# 📊 Eval Results Summary — `k8_merged` Run

**Source:** `results/retriever_eval_results.csv`
**Run ID:** `k8_merged`
**Timestamp:** `2026-09-26`
**Total Questions:** 50

---

## 🔢 Score Distribution

| Score | Recall Count | Precision Count |
|---|---|---|
| **1.00** | 23 | 12 |
| **0.75–0.99** | 8 | 11 |
| **0.50–0.74** | 8 | 10 |
| **0.25–0.49** | 7 | 11 |
| **0.00–0.24** | 4 | 6 |

---

## 🌟 Perfect Scores (Recall = 1.0 AND Precision = 1.0)

| ID | Question |
|---|---|
| Q018 | What are offline evals and what are their main benefits? |
| Q021 | What is drift and why does it make my offline eval setup obsolete? |
| Q037 | What does MMLU measure, and what does it not measure? |
| Q039 | What is GPQA and why is it called Google-proof? |
| Q055 | What is faithfulness (groundedness) and how is it used across offline, online and RAG evaluation? |
| Q060 | Why aren't benchmark scores and leaderboard rankings enough to pick a model? |
| Q081 | What's the difference between model evals and application evals? |
| Q090 | How do component-level, workflow-level and application-level failures differ? |

---

## ⬆️ High Recall, Low Precision (Recall ≥ 0.75, Precision ≤ 0.40)

> Many correct chunks were retrieved but precision was poor — too much noise returned.

| ID | Recall | Precision | Question |
|---|---|---|---|
| Q012 | 1.00 | 0.25 | What are the risk categories in LLM evaluation? |
| Q096 | 1.00 | 0.375 | What are pass@1, pass@k and majority@k? |
| Q098 | 1.00 | 0.375 | Are leaderboards a selection tool or a filtering tool? |
| Q048 | 0.75 | 0.25 | What is the RAG triad? |
| Q088 | 0.50 | 0.25 | What are captured signals versus computed signals in online evals? |
| Q066 | 0.75 | 0.375 | How are safety evaluations handled across the course? |
| Q010 | 1.00 | 0.50 | What are the three levels at which an LLM application can fail? |
| Q017 | 1.00 | 0.50 | When should I use LLM-as-a-judge instead of code or humans? |
| Q087 | 1.00 | 0.625 | How do context relevance, faithfulness and answer relevance differ? |

---

## ⬇️ Low Recall, High Precision (Recall ≤ 0.40, Precision ≥ 0.75)

> Retrieved chunks were accurate but key content was missed — under-retrieval.

| ID | Recall | Precision | Question |
|---|---|---|---|
| Q052 | 0.20 | 0.875 | What does a golden dataset contain for different kinds of evals? |
| Q074 | 0.40 | 1.00 | How do reference-free evals apply to online monitoring and the RAG triad? |
| Q043 | 0.4286 | 1.00 | Why shouldn't I blindly trust LLM leaderboards? |
| Q007 | 0.40 | 1.00 | What is a golden dataset and how is one created? |
| Q005 | 0.60 | 1.00 | What are application evals and why do they matter? *(borderline)* |

---

## 🔴 Low on Both (Recall ≤ 0.40, Precision ≤ 0.50) — Worst Performers

| ID | Recall | Precision | Question |
|---|---|---|---|
| Q085 | 0.00 | 0.625 | When should I use programmatic vs. human vs. LLM judge eval? |
| Q002 | 0.40 | 0.375 | How do we define LLM evals / key characteristics? |
| Q065 | 0.20 | 0.50 | How are latency and cost treated as evals throughout the course? |

---

## 🔁 Mismatch Flags

Both `recall_mismatch` and `precision_mismatch` columns are **False for all 50 rows** — no flagged mismatches in this run.

---

## 📋 By Question Type

| Type | Count | Avg Recall | Avg Precision |
|---|---|---|---|
| `single_concept` | 20 | ~0.77 | ~0.71 |
| `cross_session` | 16 | ~0.70 | ~0.76 |
| `comparison` | 14 | ~0.84 | ~0.69 |

> **Note:** `comparison` questions have the highest recall but lowest precision — the model tends to over-retrieve when comparing concepts.
