# rag-evals

Hands-on project for an **LLM Evals course** — evaluating RAG system components.

## Retriever Eval

Checks if the retriever (ChromaDB + BGE-M3) pulls relevant context for a question, judged by `gemini-3.1-flash-lite` against a 50-question golden dataset (recall = are ground-truth claims covered; precision = are retrieved chunks actually relevant).

```bash
python evals/retriever_eval.py
```

### Progress

| Run | k | Judge calls | Avg Recall | Avg Precision |
|---|---|---|---|---|
| `k5_baseline` | 5 | 2/question (recall + precision separate) | 0.659 | 0.648 |
| `k8_merged` (current) | 8 | 1/question (merged) | **0.768** | **0.728** |
| `k10_merged` or `query_decomp` (next) | — | 1/question | — | — |

**Why it changed:** merged the two judge calls into one (chunks were being sent twice per question for no reason), then raised k from 5→8. Both scores improved; cross_session questions (answers spanning multiple course sessions) are still the weakest category — next run tests whether that's fixed by more chunks (k=10) or needs query decomposition (splitting the question into per-session sub-queries) instead.

Each run's raw results are kept as separate CSVs (`k5_baseline_dualcall_eval.csv`, `eval_results.csv` with `run_id` column) for this comparison.

## Next
- Decide k=10 vs query decomposition based on cross_session recall specifically