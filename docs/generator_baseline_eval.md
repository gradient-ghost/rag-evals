# 📊 Generator Baseline Evaluation Analysis — `baseline_21` Run

**Source:** [`evals/generator_evals.csv`](../evals/generator_evals.csv)  
**Run ID:** `baseline_21`  
**Generator Model:** `gemini-3.5-flash` ($T=0.0$)  
**Judge Model:** `gemini-3.8-flash`  
**Total Questions:** 21  

---

## 1. Executive Summary & Core Metrics

| Metric | Overall (All 21 Questions) | Answered Subset ($n=18$) | Notes |
| :--- | :---: | :---: | :--- |
| **Total Evaluated** | **21** | 18 | `baseline_21` run |
| **Clean Pass Rate (Zero Hallucination)** | **100.0% (21/21)** | **100.0% (18/18)** | Not a single ungrounded claim was generated |
| **Average Faithfulness** | **1.000** | **1.000** | Flawless grounding against provided context |
| **Average Answer Relevancy** | **0.851** | **0.993** | 17/18 answered questions scored a perfect 1.0 |
| **Refusal / Abstention Count** | **3 (14.3%)** | — | Triggered the prompt's negative constraint rule |

---

## 2. Breakdown by Performance Category

* **Perfect Scores (`Faithfulness: 1.0`, `Relevancy: 1.0`) — 17 / 21 questions (81.0%)**:
  * Questions: `q001`, `q003`–`q009`, `q012`, `q014`–`q021`.
  * Across all standard RAG topics (vibe testing, drift, sampling, metrics, triad, safety vs. quality boundaries), every claim generated was both factually grounded in the golden context and relevant to the user question.

* **Refusals (`Faithfulness: 1.0`, `Relevancy: 0.0`) — 3 / 21 questions (14.3%)**:
  * Questions: `q010`, `q011`, and `q013`.
  * Each returned the prompt's negative constraint: `"I cannot answer this question based on the provided context."`
  * Under the deterministic scoring rules, a refusal receives **`1.0` Faithfulness** (it refrained from hallucinating) and **`0.0` Relevancy** (it produced 0 answer claims).

* **Minor Relevancy Penalty (`Faithfulness: 1.0`, `Relevancy: 0.875`) — 1 / 21 questions (4.8%)**:
  * Question: `q002` (*"what is the mmlu benchmark?"*).

---

## 3. Deep Dive into Non-1.0 Cases

### A. The 3 Refusals: Why Did the Model Abstain?
Inspecting the context chunks in [`evals/generated_answers.jsonl`](../evals/generated_answers.jsonl) reveals that the model **correctly abstained** due to information gaps in the golden context:

1. **`q010`** (*"how are safety issues like toxicity evaluated differently from quality issues?"*):
   * **Context Content**: Mentions that safety has separate pipelines (code injection, toxicity) and quality has separate pipelines (hallucination, correctness), but does not detail *how* the evaluation methodology differs.
   * **Verdict**: Refusal is factually correct; answering would require external knowledge.
2. **`q011`** (*"does training on more data always make an llm more knowledgeable?"*):
   * **Context Content**: Discusses knowledge capabilities and MMLU's history, but never makes a claim about whether adding more data *always* increases knowledge.
   * **Verdict**: Refusal is factually correct.
3. **`q013`** (*"how can thumbs up and thumbs down signals help build a golden dataset?"*):
   * **Context Content**: Discusses tracking thumbs up/down as an operational user signal to detect degradation, but does not describe the workflow for turning them into a golden dataset.
   * **Verdict**: Refusal is factually correct.

### B. The Partial Relevancy Score on `q002` (0.875)
* **Question**: *"what is the mmlu benchmark?"*
* **Generated Answer**: Defined MMLU (57 subjects), explained MMLU Pro, and added that other benchmarks like GPQA were introduced to test knowledge breadth.
* **Judge Deduction**: 7 of 8 decomposed claims directly answered the question about MMLU; 1 claim regarding GPQA was flagged as tangential to the specific question ($7 / 8 = 0.875$).

---

## 4. Operational & Engineering Observations

* **Resume Logic & Crash Safety Verified**:
  * In [`evals/generator_evals.csv`](../evals/generator_evals.csv), questions `q001`–`q004` completed in the first run (~15:28 UTC), and `q005`–`q021` resumed cleanly at 16:09 UTC without re-running earlier items or losing any data.
* **Strict Grounding Adherence**:
  * The generator's system prompt successfully prevented extrapolation and speculation across the board. The model exhibited zero false confidence or hallucination on difficult/incomplete contexts.

---

## 5. Key Takeaways & Recommendations

1. **Production Readiness of Prompt**: The anti-hallucination prompt is performing well: 100% clean pass rate with zero hallucinations across 21 test queries.
2. **Context Enrichment in Dataset**: For questions `q010`, `q011`, and `q013`, if you intend for the RAG assistant to answer them, the underlying golden chunks need to be expanded to include the missing explanatory text from the transcripts.
