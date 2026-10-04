# 📊 Contextual Relevancy Evaluation Analysis — `ctx_relevancey_v1` Run

**Source:** [`results/ctx_relevancy_eval.csv`](../results/ctx_relevancy_eval.csv)  
**Context Source:** [`data/contexts/ctx_relevancy.jsonl`](../data/contexts/ctx_relevancy.jsonl)  
**Run ID:** `ctx_relevancey_v1`  
**Timestamp:** `2026-10-04`  
**Judge Model:** `gemini-3.5-flash-lite` ($T=0.0$, Structured JSON Output)  
**Retriever Architecture:** Two-stage hybrid (Stage 1: ChromaDB + BGE-M3 @ $k=16 \rightarrow$ Stage 2: Cross-Encoder Reranker `BAAI/bge-reranker-v2-m3` @ top-5)  
**Total Questions Evaluated:** 20  

---

## 1. Executive Summary & Core Metrics

Contextual Relevancy is the **first metric of the RAG Triad**, evaluating the signal-to-noise ratio of the retrieved context relative to the input query ($\frac{\text{Relevant Claims}}{\text{Total Claims}}$). This reference-free evaluation uses Gemini 3.8 Flash to decompose the top-5 retrieved chunks into discrete factual claims and classify each as relevant or non-relevant, scored deterministically in pure Python.

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Total Questions Evaluated** | **20** | Randomly sampled from golden pools (seed `42`) |
| **Mean Relevancy Score (Macro)** | **0.7897 (79.0%)** | Average query-level context relevancy |
| **Median Relevancy Score** | **0.8000 (80.0%)** | 50% of queries achieved $\ge 80\%$ relevancy |
| **Aggregate Relevancy (Micro)** | **0.8049 (80.5%)** | 99 of 123 extracted claims were relevant |
| **Perfect Retrieval Rate (Score = 1.0)** | **30.0% (6 / 20)** | Zero irrelevant claims in retrieved context |
| **High Retrieval Rate ($\ge 0.80$)** | **65.0% (13 / 20)** | Dense, high-signal context retrieval |
| **Total Outlier Failures (Score = 0.0)** | **5.0% (1 / 20)** | Query `Q074` (cross-session concept mismatch) |

> **Key Takeaway:** The hybrid retrieval architecture (BGE-M3 + BGE Cross-Encoder Reranker) delivers an **80.5% aggregate precision rate**, consistently filtering out off-topic chunks. The remaining 19.5% noise stems primarily from conversational lecture transcript filler rather than semantic hallucination.

---

## 2. Complete Score Distribution

```
Score Bracket         Count   Percentage   Qualitative Performance
─────────────────────────────────────────────────────────────────────────────
1.000 (Perfect)         6       30.0%      Flawless — 100% of context claims relevant
0.800 – 0.999 (High)    7       35.0%      Strong — high signal with minor conversational noise
0.600 – 0.799 (Mid)     6       30.0%      Moderate — core answer retrieved with lecture tangents
0.000 – 0.599 (Poor)    1        5.0%      Failure — cross-concept retrieval divergence
```

---

## 3. Detailed Results by Question

| ID | Question | Score | Relevant / Total Claims | Tier |
| :--- | :--- | :---: | :---: | :---: |
| **Q037** | What does MMLU measure, and what does it not measure? | **1.000** | 8 / 8 | Perfect |
| **Q066** | How are safety evaluations handled across the course? | **1.000** | 5 / 5 | Perfect |
| **Q060** | Why aren't benchmark scores and leaderboard rankings enough to pick a model? | **1.000** | 6 / 6 | Perfect |
| **Q044** | What's the process for choosing an LLM for my application using custom model evals? | **1.000** | 6 / 6 | Perfect |
| **Q059** | What's the full workflow from benchmarks to choosing a model for my app? | **1.000** | 4 / 4 | Perfect |
| **Q034** | What is an eval harness and why use one? | **1.000** | 7 / 7 | Perfect |
| **Q007** | What is a golden dataset and how is one created? | **0.875** | 7 / 8 | High |
| **Q081** | What's the difference between model evals and application evals? | **0.857** | 6 / 7 | High |
| **Q084** | What's the difference between reference-based and reference-free evaluation? | **0.833** | 5 / 6 | High |
| **Q010** | What are the three levels at which an LLM application can fail and be evaluated? | **0.800** | 4 / 5 | High |
| **Q030** | What is a benchmark and what does it consist of? | **0.800** | 4 / 5 | High |
| **Q005** | What are application evals and why do they matter if we already have model evals? | **0.800** | 4 / 5 | High |
| **q019** | what is humanity's last exam (hle) and why did the researchers give it that name? | **0.800** | 8 / 10 | High |
| **q015** | what is the rag triad and what does each of its three metrics check? | **0.714** | 5 / 7 | Moderate |
| **q021** | what does faithfulness (groundedness) mean when evaluating a generator... | **0.714** | 5 / 7 | Moderate |
| **Q035** | What is benchmark contamination? | **0.600** | 3 / 5 | Moderate |
| **Q083** | Why do I need custom model evals when benchmarks already exist? How are they different? | **0.667** | 4 / 6 | Moderate |
| **Q014** | What are the different eval methods and who actually executes the evaluation? | **0.667** | 4 / 6 | Moderate |
| **q010** | how are safety issues like toxicity evaluated differently from quality issues? | **0.667** | 4 / 6 | Moderate |
| **Q074** | How do reference-free evals apply to online monitoring and the RAG triad? | **0.000** | 0 / 4 | Failure |

---

## 4. Deep Dive into Performance Patterns

### A. Perfect Relevancy (1.000 — 6 Questions)
In 30% of cases, the cross-encoder reranker isolated chunks where every single factual proposition was relevant:
* **Workflow & Process Queries (`Q044`, `Q059`):** Questions describing end-to-end methodologies (e.g. from benchmarks to custom evals) matched dedicated instructional transcript segments with zero tangential context.
* **Well-Bounded Concepts (`Q034`, `Q037`, `Q060`, `Q066`):** Topics such as eval harness definitions and MMLU scope retrieved tightly focused lecture discussions.

### B. High & Moderate Relevancy (0.600 – 0.875 — 13 Questions)
In 65% of cases, the core answer was fully present, but the context contained 1 to 2 irrelevant claims. Analyzing the reasoning strings provided by the judge reveals two consistent causes:
1. **Conversational Transcript Noise:** The raw video transcripts contain spoken transitions, rhetorical questions, and student chat interruptions that are packaged inside 1000-character chunks along with relevant material (e.g. `Q035` retrieved classroom banter alongside contamination definitions).
2. **Adjacent Curriculum Mentions:** When discussing one topic, the instructor frequently references related future lectures (e.g. `q015` retrieved mentions of offline evaluation cycles alongside RAG triad metrics).

### C. The Outlier Failure Case: `Q074` (Score: 0.000)
* **Question:** *"How do reference-free evals apply to online monitoring and the RAG triad?"*
* **Root Cause Analysis:**
  * This is a **cross-session multi-concept question** requiring synthesis across **Session 2** (reference-based vs. reference-free theory) and **Session 8** (online monitoring and RAG triad production setup).
  * The dense retriever retrieved 2 chunks from Session 8 and 3 chunks from Session 2 based on keyword overlap (*"reference free"*, *"online"*, *"triad"*).
  * However, the individual chunks contained generic framing comments (*"don't worry we will see all these things practically"*, *"golden data sets get more enriched"*) rather than the substantive synthesis connecting the concepts.
  * The judge correctly concluded that none of the 4 extracted claims answered how reference-free evals apply to online monitoring.

---

## 5. Architectural & Pipeline Takeaways

1. **Reranker Efficacy Confirmed:** Fetching $k=16$ candidates and reranking down to top-5 achieves **80.5% context precision** without human intervention or supervised filtering.
2. **Transcript Cleaning is the Highest-ROI Next Step:** Spoken discourse markers ("let's say", "quick quiz", "give me time") account for the majority of the 24 discarded claims. Pre-filtering transcripts prior to chunking will lift average contextual relevancy toward 90%+.
3. **Multi-Hop / Cross-Session Handling:** For cross-session synthesis questions (like `Q074`), increasing initial candidate retrieval from $k=16$ to $k=24$ or performing query decomposition will prevent false-positive keyword trapping.
