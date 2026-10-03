# rag-evals

Component-level evaluation of a RAG system built for an LLM Evals course. Evaluates both the **Retriever** and **Generator** independently using crash-safe pipelines and LLM-as-a-judge.

---

## 🏛️ Core Architectural Principles (Immutable Design Choices)

1. **Direct Official SDKs Over Heavy Framework Wrappers (No LangChain for LLMs or Inference):**
   * **Zero Framework Wrappers for LLMs:** Avoided framework wrappers (e.g., LangChain Chat models) for LLM interactions. All generation and judge evaluations use the **official Google GenAI SDK (`google-genai` / `genai.Client`)** with strict Pydantic schemas and native JSON output modes. This eliminates leaky abstractions, hidden prompt mutations, and breaking framework updates.
   * **Native Model Inference:** Heavy model tasks run through their dedicated libraries directly—using `sentence-transformers` (`CrossEncoder`) for the reranker and Hugging Face/PyTorch for `BAAI/bge-m3` embeddings—bypassing brittle wrapper chains.

2. **Deterministic Arithmetic (Pure Python Scoring, Zero LLM Math):**
   * **No LLM Calculation:** Smaller, high-speed judge models (like `gemini-3.5-flash-lite`) are susceptible to arithmetic mistakes, rounding errors, or inconsistent scale interpretations if asked to produce numerical scores directly.
   * **LLM as a Pure Binary Classifier:** The LLM judge is strictly restricted to atomic boolean verdicts (`is_supported: bool`, `is_relevant: bool`, `is_hallucination: bool`) along with 1-sentence explanations.
   * **Raw Python for All Math:** Every mathematical calculation—such as recall ratios ($\frac{\text{supported claims}}{\text{total claims}}$), precision ratios ($\frac{\text{relevant chunks}}{\text{retrieved chunks}}$), clean pass rates, and dataset-level averages—is computed deterministically in standard Python code across both retriever and generator evaluations.

---

## 1. Retriever Evaluation

Evaluates the vector store, embedding search, and reranker pipeline against a 50-question golden dataset.

* **Stack:** ChromaDB + BGE-M3 embeddings, BAAI/bge-reranker-v2-m3 cross-encoder, evaluated with `gemini-3.5-flash-lite` judge using structured JSON outputs.
* **Metrics (Pure Python Scoring):**
  * **Context Recall:** Did retrieved context as a whole support all golden claims?
  * **Context Precision:** Proportion of retrieved chunks that are strictly relevant to the question.
* **Method:** Multi-stage crash-safe pipeline with isolated subprocesses, intermediate candidate caching, and atomic CSV flushing with resume support.

### Results

| Run | Architecture | Top-$k$ | Judge Calls | Avg Recall | Avg Precision | Perfect (1.0/1.0) | Raw Results |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| `k5_baseline_dualcall` | Dual-call Dense | 5 | 2 / question | 0.659 | 0.648 | 3 / 50 (6%) | [`results/k5_baseline_dualcall_ret_eval.csv`](results/k5_baseline_dualcall_ret_eval.csv) |
| `k8_merged` | Dense Baseline | 8 | 1 / question | 0.768 | 0.728 | 8 / 50 (16%) | [`results/retriever_eval_results.csv`](results/retriever_eval_results.csv) |
| **`reranker_bge_v2`** | **Dense ($k=16$) + Cross-Encoder Reranker** | **5** | **1 / question** | **0.770** | **0.888** | **12 / 50 (24%)** | [`results/retriever_eval_results.csv`](results/retriever_eval_results.csv) |

*Detailed reports: [`docs/k5_baseline_dualcall_eval.md`](docs/k5_baseline_dualcall_eval.md), [`docs/k8_merged_eval_results.md`](docs/k8_merged_eval_results.md), [`docs/retrieval_reranker.md`](docs/retrieval_reranker.md).*

### Key Findings
* **Precision Surge:** Adding `BAAI/bge-reranker-v2-m3` over broad $k=16$ candidates yielded a massive **+16.05% absolute jump in precision** (rising from 72.8% to **88.8%**), with **33 out of 50 questions** scoring a perfect 1.00 precision.
* **Noise Elimination:** In `k8_merged`, 9 questions suffered from high recall but noisy precision ($\le 0.40$). With the reranker, this dropped to just **1 question** (`Q086`).
* **Rescuing Zero-Recall Queries:** Questions with lexical gaps like `Q085` (programmatic vs. human vs. LLM judge) jumped from **0.00 recall to 0.50 recall and 1.00 precision** because cross-encoder query-document attention bridges vocabulary differences that dense cosine distance misses.

---

### ⚙️ System Resource Management: Isolated Subprocess Architecture

Running large embedding models and deep cross-encoders on edge/consumer hardware (e.g., NVIDIA GTX 1650 with 3.6 GB VRAM) presents severe memory limits:
* **BGE-M3 (Embeddings):** Requires ~3.5 GB VRAM.
* **BGE-Reranker-v2-m3 (Cross-Encoder):** Requires ~2.2 GB VRAM.
* **3.5 GB + 2.2 GB = 5.7 GB > 3.6 GB** ➔ Attempting to run both models sequentially inside a single Python process triggers `torch.OutOfMemoryError` because PyTorch and CUDA retain internal memory pools even after `del` and `empty_cache()`.

**Solution: Isolated Process Lifecycles (`evals/retriever_eval.py`)**
1. **Phase 1A Subprocess (`--phase1a`):** Spawns an isolated process that loads BGE-M3, queries ChromaDB for $k=16$ candidates per question, writes intermediate records to `data/contexts/candidates_k16.jsonl`, and **exits completely**. Upon process exit, the OS immediately reclaims 100% of GPU memory.
2. **Phase 1B Subprocess (`--phase1b`):** Starts with an entirely clean GPU (0% allocated VRAM), loads the `CrossEncoder`, reranks all 16 candidate chunks per query down to the top-5, writes `data/contexts/retrieved_context_reranker.jsonl`, and exits cleanly.
3. **Phase 2 (Judging):** Runs purely via the Gemini API without GPU dependency, scoring all 50 questions with rate-limiting and crash-resilient CSV checkpointing.

---

### 🔍 Root Cause Analysis: Why Certain Questions Score Low on Both Metrics

Despite the reranker's precision boost, a small cluster of queries (e.g., `Q052`, `Q065`, `Q074`, `Q091`) continue to score low on recall and precision. Three structural bottlenecks explain why:

1. **Mid-Sentence Slicing (Chunking Boundary Artifacts):**
   * Transcripts are flattened with `" ".join(lines)` without markdown headers or paragraph breaks.
   * `RecursiveCharacterTextSplitter(chunk_size=1000)` has no paragraph separators to split on, so it falls back to splitting at arbitrary space characters near 1,000 characters.
   * Chunks frequently cut off mid-sentence (e.g., starting with `...explanation. Third whatever...` or ending with `...model eval is`). When a premise is in Chunk A and the conclusion is cut into Chunk B, the LLM judge marks the golden claim as unsupported.
2. **Conversational Speech Noise (Data Cleaning Gaps):**
   * Transcripts are raw spoken lectures containing verbal filler (*"so yeah the answer is yes"*, *"Is it 5 or 6? This is your tension"*), digressions, and live audience Q&A (*"Aman is asking..."*).
   * Between 30% and 40% of the character budget in a chunk consists of speech filler, diluting semantic density and pulling down retrieval relevance for technical topics.
3. **The "Cross-Session Trap" (Fixed Top-$k$ Window vs. Multi-Lecture Spread):**
   * Multi-session questions require facts taught across multiple 1-hour lectures (e.g., `Q074` requires Sessions `[2, 3, 8]`; `Q065` requires Sessions `[2, 3, 7]`; `Q053` requires Sessions `[2, 3, 5, 6]`).
   * A global top-5 reranked window gets saturated by 1 or 2 dominant sessions, leaving the remaining required sessions unretrieved. Consequently, precision remains high (~100%) on what was retrieved, but recall is capped at 20%–40% due to missing lectures.

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

*Raw data: [`data/contexts/generated_answers.jsonl`](data/contexts/generated_answers.jsonl), [`results/generator_evals.csv`](results/generator_evals.csv).*
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