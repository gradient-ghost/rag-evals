# rag-evals

A hands-on project from an **LLM Evals course**, focused on evaluating LLM-powered applications — specifically RAG systems.

---

## What's in here

**RAG Component Eval — Retriever**

Evaluates how well the retriever (ChromaDB + BGE-M3 embeddings) fetches relevant context for a given question.

Two metrics are judged by `gemini-3.1-flash-lite` against a 50-question golden dataset:

- **Recall** — do the retrieved chunks cover the ground-truth claims?
- **Precision** — are the retrieved chunks actually relevant to the question?

| Metric | Score |
|---|---|
| Avg Recall | 0.6594 |
| Avg Precision | 0.6480 |

---

## How it works

```
goldens/claude_golden_dataset.jsonl   ← 50 hand-labelled Q&A pairs with claims
        ↓  Phase 1 (run once)
evals/retrieved_context.jsonl         ← top-5 chunks per question from ChromaDB
        ↓  Phase 2 (resumable, rate-limited)
evals/eval_results.csv                ← per-question recall & precision scores
```

Run it from the project root:
```bash
python evals/retriever_eval.py
```

---

## What's next

- **Generator Eval** — evaluate the LLM's answer quality given the retrieved context
