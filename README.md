# rag-evals

Evaluating the retriever component of a RAG system built for an LLM Evals course.

**Stack:** ChromaDB vector store + BGE-M3 embeddings, judged by `gemini-flash-lite` against a 50-question golden dataset.

**What's being measured:**
- **Recall** — do the retrieved chunks cover the ground-truth answer?
- **Precision** — are the retrieved chunks actually relevant, or just noise?

```bash
python evals/retriever_eval.py
```

---

## Runs

| Run | k | Judge calls | Avg Recall | Avg Precision |
|---|---|---|---|---|
| `k5_baseline_dualcall` | 5 | 2/question | 0.659 | 0.648 |
| `k8_merged` (current) | 8 | 1/question | **0.768** | **0.728** |

Raw results: [`evals/k5_baseline_dualcall_eval.csv`](evals/k5_baseline_dualcall_eval.csv), [`evals/eval_results.csv`](evals/eval_results.csv)
Full analysis: [`docs/k5_baseline_dualcall_eval.md`](docs/k5_baseline_dualcall_eval.md), [`docs/k8_merged_eval_results.md`](docs/k8_merged_eval_results.md)

---

## Current Failure Modes

### 1. Over-retrieval — High Recall, Low Precision
The retriever finds the right content but pulls too many irrelevant chunks alongside it.
Consistent across both runs:

| ID | Question | k8 Precision | k5 Precision |
|---|---|---|---|
| Q012 | What are the risk categories in LLM evaluation? | 0.25 | 0.60 |
| Q048 | What is the RAG triad? | 0.25 | 0.20 |
| Q096 | What are pass@1, pass@k and majority@k? | 0.375 | 0.20 |
| Q098 | Are leaderboards a selection tool or a filtering tool? | 0.375 | 0.60 |
| Q017 | When should I use LLM-as-a-judge instead of code or humans? | 0.50 | 0.20 |

### 2. Under-retrieval — Low Recall, High Precision
Returned chunks are clean but the retriever misses key content entirely.
Persistent across both runs:

| ID | Question | k8 Recall | k5 Recall |
|---|---|---|---|
| Q052 | What does a golden dataset contain for different kinds of evals? | 0.20 | 0.40 |
| Q065 | How are latency and cost treated as evals throughout the course? | 0.20 | 0.20 |
| Q074 | How do reference-free evals apply to online monitoring and the RAG triad? | 0.40 | 0.40 |

### 3. Hard Failures — Low on Both
Both recall and precision are weak; the retriever fundamentally struggles with these questions.

| ID | Question | k8 Recall | k8 Precision | k5 Recall | k5 Precision |
|---|---|---|---|---|---|
| Q085 | When should I use programmatic vs. human vs. LLM judge? | 0.00 | 0.625 | 0.75 | 0.80 |
| Q088 | What are captured vs. computed signals in online evals? | 0.50 | 0.25 | 0.25 | 0.20 |
| Q091 | How is testing an LLM app different from ordinary software testing? | 0.40 | 0.50 | 0.40 | 0.20 |

### 4. Cross-Session Questions Are the Weakest Category
Questions that require synthesising content across multiple course sessions consistently score lowest.

| Run | Avg Recall | Avg Precision |
|---|---|---|
| `k8_merged` | ~0.70 | ~0.76 |
| `k5_baseline_dualcall` | ~0.63 | ~0.64 |

### 5. Precision Mismatch Flag
One extreme outlier in the k5 baseline run — perfect recall but zero precision:

| ID | Question | Recall | Precision |
|---|---|---|---|
| Q079 | How do I fairly compare two versions or two models of my system? | 1.00 | 0.00 |

---

## Next
- Test `k=10` vs **query decomposition** (splitting cross-session questions into per-session sub-queries) — targeting cross-session recall specifically