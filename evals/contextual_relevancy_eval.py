"""
Contextual Relevancy Evaluation — RAG Triad Metric 1
======================================================
Reference-free evaluation pipeline with decoupled retrieval and judging:

Phase 1 — Context Retrieval:
  1. Randomly samples questions from the golden datasets.
  2. Runs retrieval (CPU embeddings + CUDA cross-encoder reranker).
  3. Generator LLM call is SKIPPED ENTIRELY (no answer generation needed).
  4. Stores {id, question, retrieved_context, context_documents} into:
       data/contexts/ctx_relevancy.jsonl

Phase 2 — Judging & Scoring:
  1. Reads data/contexts/ctx_relevancy.jsonl.
  2. The context file is left COMPLETELY UNTOUCHED (never updated with scores).
  3. Sends (question, retrieved_context) to Judge LLM (Gemini 3.8 Flash).
  4. Extracts atomic factual claims and labels each is_relevant: bool.
  5. Scores deterministically in pure Python: relevant_claims / total_claims.
  6. Persists scores ONLY into:
       results/ctx_relevancy_eval.csv

Architecture principles:
  - Zero framework wrappers: google-genai SDK called directly for judge.
  - LLM judge = pure binary classifier only (zero arithmetic by LLM).
  - All scoring in pure Python.
  - Zero latency tracking.
  - Crash-safe resumable execution.
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types
from google.genai.errors import ServerError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Logging & Environment
# ---------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Silence noisy HTTP request, Hugging Face, and sentence-transformers logging
for _noisy in (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "transformers",
    "sentence_transformers",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
JUDGE_MODEL = "gemini-3.5-flash-lite"
DEFAULT_RUN_ID = "ctx_relevancey_v1"
DEFAULT_SAMPLES = 20
DEFAULT_SEED = 42
INTER_QUESTION_DELAY = 15  # seconds between judge calls (throttles to 4 RPM)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATHS = [
    PROJECT_ROOT / "goldens" / "retriever_golden.jsonl",
    PROJECT_ROOT / "goldens" / "generator_golden.jsonl",
]
CONTEXTS_JSONL = PROJECT_ROOT / "data" / "contexts" / "ctx_relevancy.jsonl"
CSV_OUT = PROJECT_ROOT / "results" / "ctx_relevancy_eval.csv"

CSV_COLUMNS = [
    "id",
    "question",
    "contextual_relevancy_score",
    "relevant_claims",
    "total_claims",
    "run_id",
    "timestamp",
]

# ---------------------------------------------------------------------------
# Judge System Prompt
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are an objective AI evaluation judge for a RAG (Retrieval-Augmented Generation) system.

Your task is to evaluate CONTEXTUAL RELEVANCY: given a user question and a set of retrieved context chunks, determine how much of the context is actually relevant and useful for answering that specific question.

Step 1 — Atomic Claim Extraction:
- Read ALL retrieved context chunks together as a unified body of text.
- Decompose the context into discrete, self-contained atomic factual claims.
- A claim is a single factual statement. Compound sentences must be split.
- IGNORE: speaker filler words, audience chat, repetitions, meta-commentary like "let me explain" or "good question", and off-topic banter.
- Extract only concrete informational facts present in the context.

Step 2 — Relevancy Classification:
- For EACH extracted claim, judge whether it directly helps answer the user's question.
- A claim is relevant (is_relevant: true) if it contains information that contributes to answering the question — either directly or as essential background context for the answer.
- A claim is NOT relevant (is_relevant: false) if it is off-topic, tangential trivia, or generic filler that does not help answer this specific question.
- Judge purely on relevance to the question. Do NOT consider whether the claim is factually correct.

Step 3 — Strict Formatting:
- CRITICAL: Do NOT calculate scores, ratios, or percentages. Return only the list of claims and their boolean verdicts.
- Provide a brief 1-sentence reasoning for each verdict.
- Return a short evaluation_summary (1-2 sentences) describing the overall signal-to-noise ratio of the retrieved context.

Input Format:
User Question: {question}
Retrieved Context: {context}"""


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class ContextClaimVerdict(BaseModel):
    claim: str
    is_relevant: bool
    reasoning: str


class ContextualRelevancyJudgeOutput(BaseModel):
    claims: list[ContextClaimVerdict]
    evaluation_summary: str


# ---------------------------------------------------------------------------
# Retry Helper
# ---------------------------------------------------------------------------
def _is_retryable(exc: BaseException) -> bool:
    """Retry only on 503 Service Unavailable. Never retry 429 quota errors."""
    if isinstance(exc, ServerError):
        return exc.code == 503
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=20, max=120),
    stop=stop_after_attempt(4),
    reraise=True,
)
def call_judge(
    client: genai.Client,
    question: str,
    retrieved_context: str,
) -> ContextualRelevancyJudgeOutput:
    """Calls the Judge LLM with structured output and exponential backoff on 503s."""
    user_content = (
        f"User Question:\n{question}\n\n"
        f"Retrieved Context:\n{retrieved_context}"
    )

    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_PROMPT.strip(),
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ContextualRelevancyJudgeOutput,
        ),
    )

    if isinstance(response.parsed, ContextualRelevancyJudgeOutput):
        return response.parsed
    return ContextualRelevancyJudgeOutput.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# Deterministic Scoring (Pure Python — Zero LLM Math)
# ---------------------------------------------------------------------------
def compute_relevancy_score(
    claims: list[ContextClaimVerdict],
) -> tuple[float, int, int]:
    """Returns (score, relevant_count, total_count) computed in pure Python.

    Score = relevant_claims / total_claims.
    Returns (0.0, 0, 0) if no claims were extracted (edge case protection).
    """
    if not claims:
        return 0.0, 0, 0
    total = len(claims)
    relevant = sum(1 for c in claims if c.is_relevant)
    score = round(relevant / total, 3)
    return score, relevant, total


# ---------------------------------------------------------------------------
# Question Pool Builder
# ---------------------------------------------------------------------------
def build_question_pool() -> list[dict]:
    """Loads and merges questions from all golden datasets into a unified pool.

    Normalises schema to: {id, question, source_file}
    """
    pool: list[dict] = []
    for path in GOLDEN_PATHS:
        if not path.exists():
            log.warning("Golden dataset not found, skipping: %s", path)
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                question = item.get("question") or item.get("query", "")
                pool.append({
                    "id": item["id"],
                    "question": question,
                    "source_file": path.name,
                })
    return pool


# ---------------------------------------------------------------------------
# Phase 1: Retrieve Context and Store in data/contexts/ctx_relevancy.jsonl
# ---------------------------------------------------------------------------
def run_retrieval(
    questions: list[dict],
    output_path: Path,
    top_k: int = 5,
) -> None:
    """Retrieves context chunks for each question and persists to ctx_relevancy.jsonl.

    Does NOT call any generator LLM.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: set[str] = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        existing_ids.add(record["id"])
                    except Exception:
                        pass

    to_retrieve = [q for q in questions if q["id"] not in existing_ids]
    if not to_retrieve:
        log.info("Phase 1: All %d questions already present in %s.", len(questions), output_path)
        return

    log.info("Phase 1: Initialising retriever (CPU embeddings + CUDA reranker)...")
    from rag_evals.pipeline import RAGPipeline
    pipeline = RAGPipeline(use_reranker=True, top_k=top_k)
    log.info("Retriever ready. Retrieving context for %d questions...", len(to_retrieve))

    with output_path.open("a", encoding="utf-8") as fh:
        for idx, item in enumerate(to_retrieve, start=1):
            qid = item["id"]
            question = item["question"]
            log.info("[Phase 1: %d/%d] Retrieving context for %s: %s...", idx, len(to_retrieve), qid, question[:60])

            docs = pipeline.retrieve(question, top_k=top_k)
            context_chunks = [d.page_content for d in docs]
            context_documents = [
                {"content": d.page_content, "metadata": d.metadata}
                for d in docs
            ]

            record = {
                "id": qid,
                "question": question,
                "retrieved_context": context_chunks,
                "context_documents": context_documents,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    log.info("Phase 1 complete. Saved %d contexts to %s.", len(to_retrieve), output_path)


# ---------------------------------------------------------------------------
# Phase 2: Judge & Score -> Save ONLY into results/ctx_relevancy_eval.csv
# ---------------------------------------------------------------------------
def run_evaluation(
    contexts_path: Path,
    csv_path: Path,
    run_id: str,
    delay: float,
) -> None:
    """Reads ctx_relevancy.jsonl, evaluates with Gemini 3.8 Flash, and writes scores to CSV.

    Leaves data/contexts/ctx_relevancy.jsonl completely untouched.
    """
    if not contexts_path.exists():
        log.error("Contexts file not found at %s. Run Phase 1 first.", contexts_path)
        sys.exit(1)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Check already evaluated items for resumability
    completed: set[tuple[str, str]] = set()
    write_csv_header = True
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                completed.add((row.get("run_id", ""), row.get("id", "")))
        write_csv_header = len(completed) == 0
        log.info("Phase 2: Resuming — %d items already evaluated.", len(completed))

    # Read context records from ctx_relevancy.jsonl
    records: list[dict] = []
    with contexts_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    log.info("Phase 2: Loaded %d context records from %s.", len(records), contexts_path)

    client = genai.Client(
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        )
    )

    total = len(records)
    with csv_path.open("a", encoding="utf-8", newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLUMNS)
        if write_csv_header:
            writer.writeheader()
            csv_fh.flush()

        for idx, record in enumerate(records, start=1):
            qid = record["id"]
            question = record["question"]

            if (run_id, qid) in completed:
                log.info("[Phase 2: %d/%d] Skipping already-evaluated: %s", idx, total, qid)
                continue

            print(f"\n[Phase 2: {idx}/{total}] ID: {qid}")
            print(f"  Q: {question[:80]}...")

            context_chunks = record.get("retrieved_context", [])
            retrieved_context_str = "\n\n".join(
                f"[Chunk {j+1}] {chunk}" for j, chunk in enumerate(context_chunks)
            )

            try:
                judge_output = call_judge(client, question, retrieved_context_str)
            except Exception as exc:
                log.error("[%s] Judge error: %s — skipping.", qid, exc)
                continue

            score, relevant_count, total_claims = compute_relevancy_score(judge_output.claims)
            print(f"  Score: {score:.3f}  ({relevant_count}/{total_claims} claims relevant)")
            print(f"  Summary: {judge_output.evaluation_summary}")

            # Persist row ONLY into CSV (data/contexts/ctx_relevancy.jsonl is NOT touched)
            writer.writerow({
                "id": qid,
                "question": question,
                "contextual_relevancy_score": score,
                "relevant_claims": relevant_count,
                "total_claims": total_claims,
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            csv_fh.flush()
            os.fsync(csv_fh.fileno())

            if idx < total:
                log.info("Rate limit: sleeping %.0fs before next question...", delay)
                time.sleep(delay)

    log.info("Phase 2 evaluation complete for run_id=%s.", run_id)
    print_summary(csv_path, run_id)


# ---------------------------------------------------------------------------
# Summary Statistics Printer
# ---------------------------------------------------------------------------
def print_summary(csv_path: Path, run_id: str) -> None:
    if not csv_path.exists():
        log.warning("No results file found at %s.", csv_path)
        return

    scores: list[float] = []
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("run_id") == run_id:
                scores.append(float(row["contextual_relevancy_score"]))

    if not scores:
        log.warning("No rows found for run_id=%s.", run_id)
        return

    avg = sum(scores) / len(scores)
    perfect = sum(1 for s in scores if s == 1.0)

    print("\n" + "=" * 56)
    print(f"  Contextual Relevancy Summary (run_id: {run_id})")
    print("=" * 56)
    print(f"  Questions evaluated : {len(scores)}")
    print(f"  Avg Relevancy Score : {avg:.3f}")
    print(f"  Perfect (1.0) Count : {perfect}/{len(scores)}")
    print("=" * 56 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Contextual Relevancy Evaluation (RAG Triad Metric 1)")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"Number of questions to evaluate (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for reproducible sampling (default: {DEFAULT_SEED})")
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID,
                        help=f"Unique run identifier for this evaluation (default: {DEFAULT_RUN_ID})")
    parser.add_argument("--delay", type=float, default=INTER_QUESTION_DELAY,
                        help=f"Seconds to sleep between judge calls (default: {INTER_QUESTION_DELAY})")
    parser.add_argument("--phase", choices=["all", "retrieve", "judge"], default="all",
                        help="Execution phase: 'retrieve' (Phase 1), 'judge' (Phase 2), or 'all' (both)")
    args = parser.parse_args()

    if args.phase in ("all", "judge") and not os.environ.get("GEMINI_API_KEY"):
        log.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    # Sample questions if retrieval is needed
    if args.phase in ("all", "retrieve"):
        pool = build_question_pool()
        if not pool:
            log.error("No questions found in golden datasets.")
            sys.exit(1)

        rng = random.Random(args.seed)
        sample_size = min(args.num_samples, len(pool))
        sampled_questions = rng.sample(pool, sample_size)
        log.info("Sampled %d questions (seed=%d) from pool of %d.", sample_size, args.seed, len(pool))

        run_retrieval(sampled_questions, CONTEXTS_JSONL, top_k=5)

    # Run evaluation if judging is needed
    if args.phase in ("all", "judge"):
        run_evaluation(CONTEXTS_JSONL, CSV_OUT, args.run_id, args.delay)


if __name__ == "__main__":
    main()
