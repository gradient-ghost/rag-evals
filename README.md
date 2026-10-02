# rag-evals

Component-level evaluation of a RAG system built for an LLM Evals course. Evaluates both the **Retriever** and **Generator** independently using crash-safe pipelines and LLM-as-a-judge.

---

## 1. Retriever Evaluation

Evaluates the vector store and embedding search against a 50-question golden dataset.

* **Stack:** ChromaDB + BGE-M3 embeddings, evaluated with `gemini-3.5-flash-lite` judge using structured outputs.
* **Metrics:**
  * **Context Recall:** Did retrieved chunks contain the ground-truth claims?
  * **Context Precision:** Were retrieved chunks relevant, or noisy/tangential?
* **Method:** Two-phase execution with intermediate retrieval cached to JSONL, evaluated items flushed immediately to CSV with resume support.

### Results

| Run | k | Judge Calls | Avg Recall | Avg Precision | Raw Results |
|---|---|---|---|---|---|
| `k5_baseline_dualcall` | 5 | 2 / question | 0.659 | 0.648 | [`evals/k5_baseline_dualcall_ret_eval.csv`](evals/k5_baseline_dualcall_ret_eval.csv) |
| `k8_merged` (current) | 8 | 1 / question | **0.768** | **0.728** | [`evals/retriever_eval_results.csv`](evals/retriever_eval_results.csv) |

*Detailed reports: [`docs/k5_baseline_dualcall_eval.md`](docs/k5_baseline_dualcall_eval.md), [`docs/k8_merged_eval_results.md`](docs/k8_merged_eval_results.md).*

### Key Findings
* Increasing $k$ from 5 to 8 yielded a **+16.5% boost in recall** and **+12.3% boost in precision** with merged single-call judging.
* Weakest category: Cross-session questions that require synthesizing context across multiple course lectures.

---

## 2. Generator Evaluation

Evaluates the LLM generation component on answer quality and hallucination prevention using 21 golden questions (`baseline_21`).

* **Stack:**
  * **Generator:** `gemini-3.5-flash` ($T=0.0$) with strict anti-hallucination and negative-constraint system prompt.
  * **Judge:** `gemini-3.8-flash` with structured Pydantic output, decomposing answers into atomic claims.
* **Metrics (Pure Python Scoring):**
  * **Faithfulness:** Proportion of generated claims strictly supported by the golden context.
  * **Answer Relevancy:** Proportion of generated claims that directly address the question.
  * **Clean Pass Rate:** Percentage of answers with zero hallucinations (`has_hallucination == False`).
* **Method:** Crash-safe, resumable pipeline with 15s rate-throttling (5 RPM quota), Tenacity exponential backoffs, and immediate atomic disk flushing (`os.fsync`).

### Results

| Metric | Overall ($n=21$) | Answered Subset ($n=18$) | Notes |
|---|---|---|---|
| **Clean Pass Rate** | **100.0% (21/21)** | **100.0% (18/18)** | **Zero hallucinations** generated |
| **Avg Faithfulness** | **1.000** | **1.000** | Perfect grounding across all outputs |
| **Avg Relevancy** | **0.851** | **0.993** | 17/18 answered queries scored a perfect 1.0 |
| **Abstentions / Refusals** | **3 (14.3%)** | — | Triggered negative constraint due to context gaps |

*Raw data: [`evals/generated_answers.jsonl`](evals/generated_answers.jsonl), [`evals/generator_evals.csv`](evals/generator_evals.csv).*
*Detailed report: [`docs/generator_baseline_eval.md`](docs/generator_baseline_eval.md).*

### Key Findings
* **Zero Hallucinations:** The prompt’s strict grounding and negative-constraint rules achieved a 100% clean pass rate.
* **Controlled Abstention:** For questions with incomplete source context (`q010`, `q011`, `q013`), the generator correctly refused to speculate (`"I cannot answer this question based on the provided context"`), scoring 1.0 Faithfulness and 0.0 Relevancy.
* **High Relevancy on Answered Queries:** Scored **0.993**, with only a minor deduction on `q002` (0.875) due to an extra tangential comparison.

---

## Quickstart

```bash
# 1. Run retriever evaluation
python evals/retriever_eval.py

# 2. Run generator answer generation
python evals/generate_answers.py

# 3. Run generator evaluation judge
python evals/generator_eval.py
```