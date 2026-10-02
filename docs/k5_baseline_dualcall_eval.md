# 📊 Eval Results Summary — `k5_baseline_dualcall` Run

**Source:** `results/k5_baseline_dualcall_ret_eval.csv`
**Run ID:** `k5_baseline_dualcall`
**Total Questions:** 50

---

## 🔢 Score Distribution

| Score | Recall Count | Precision Count |
|---|---|---|
| **1.00** | 11 | 8 |
| **0.75–0.99** | 12 | 19 |
| **0.50–0.74** | 13 | 13 |
| **0.25–0.49** | 10 | 6 |
| **0.00–0.24** | 4 | 4 |

---

## 🌟 Perfect Scores (Recall = 1.0 AND Precision = 1.0)

| ID | Question |
|---|---|
| Q030 | What is a benchmark and what does it consist of? |
| Q084 | What's the difference between reference-based and reference-free evaluation? |
| Q090 | How do component-level, workflow-level and application-level failures differ? |

---

## ⬆️ High Recall, Low Precision (Recall ≥ 0.75, Precision ≤ 0.40)

> Many correct chunks were retrieved but precision was poor — too much noise returned.

| ID | Recall | Precision | Question |
|---|---|---|---|
| Q079 | 1.00 | **0.00** ⚠️ | How do I fairly compare two versions or two models of my system? |
| Q017 | 0.80 | **0.20** | When should I use LLM-as-a-judge instead of code or humans? |
| Q022 | 1.00 | **0.40** | What is online eval and why does it not need a golden dataset? |
| Q048 | 0.75 | **0.20** | What is the RAG triad? |
| Q096 | 1.00 | **0.20** | What are pass@1, pass@k and majority@k? |
| Q087 | 0.75 | **0.40** | How do context relevance, faithfulness and answer relevance differ? |
| Q077 | 0.60 | **0.40** | What would a good structured answer to 'How do you evaluate a RAG chatbot?' include? |

---

## ⬇️ Low Recall, High Precision (Recall ≤ 0.40, Precision ≥ 0.75)

> Retrieved chunks were accurate but key content was missed — under-retrieval.

| ID | Recall | Precision | Question |
|---|---|---|---|
| Q005 | **0.40** | 0.80 | What are application evals and why do they matter if we already have model evals? |
| Q056 | **0.40** | 0.80 | How does regression testing connect offline evals, baselines and CI? |
| Q058 | **0.40** | 1.00 | How do multiple failure points and risk categories translate into an actual RAG eval suite? |
| Q039 | **0.40** | 0.80 | What is GPQA and why is it called Google-proof? |
| Q049 | **0.40** | 0.60 | What is regression testing for an LLM app and what levels of it exist? *(borderline)* |

---

## 🔴 Low on Both (Recall ≤ 0.40, Precision ≤ 0.40) — Worst Performers

| ID | Recall | Precision | Question |
|---|---|---|---|
| Q052 | **0.40** | **0.00** | What does a golden dataset contain for different kinds of evals? |
| Q074 | **0.40** | **0.20** | How do reference-free evals apply to online monitoring and the RAG triad? |
| Q088 | **0.25** | **0.20** | What are captured signals versus computed signals in online evals? |
| Q091 | **0.40** | **0.20** | How is testing an LLM application different from ordinary software testing? |
| Q086 | **0.25** | 0.60 | What's the difference between precision and recall for a RAG retriever? *(borderline)* |
| Q065 | **0.20** | 0.60 | How are latency and cost treated as evals throughout the course? *(borderline)* |

---

## 🔁 Mismatch Flags

One flagged mismatch found:

| ID | Recall Mismatch | Precision Mismatch | Question |
|---|---|---|---|
| Q079 | False | **True** ⚠️ | How do I fairly compare two versions or two models of my system? |

> This question scored perfect recall (1.0) but zero precision (0.00) — a stark mismatch between the two signals, and the only row flagged in this run.

---

## 📋 By Question Type

| Type | Count | Avg Recall | Avg Precision |
|---|---|---|---|
| `single_concept` | 20 | ~0.65 | ~0.68 |
| `cross_session` | 16 | ~0.63 | ~0.64 |
| `comparison` | 14 | ~0.74 | ~0.63 |

> **Note:** Overall scores are noticeably lower than the `k8_merged` run across all question types, especially for `cross_session` questions — suggesting the baseline dual-call approach retrieves less completely across multi-session topics.
