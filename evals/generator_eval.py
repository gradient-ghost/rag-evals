import csv
import json
import logging
import os
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
from google.genai.errors import APIError, ClientError, ServerError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "gemini-3.5-flash-lite"
RUN_ID = "baseline_21"
SLEEP_SECONDS = 30  # ~2 RPM — conservative given only 2 RPM headroom on gemini-3.7-flash

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "evals/generated_answers.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "evals/generator_evals.csv"

CSV_COLUMNS = [
    "id",
    "question",
    "faithfulness_score",
    "relevance_score",
    "has_hallucination",
    "run_id",
    "timestamp",
]

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
JUDGE_LLM_SYSTEM_PROMPT = """
You are an expert, objective AI evaluation judge evaluating a question-answering assistant in a Retrieval-Augmented Generation (RAG) system.
Your job is to evaluate a Generated Answer against two criteria simultaneously:
1. **Faithfulness (Groundedness / Hallucination Detection):** Is every factual assertion strictly backed by the provided Golden Context?
2. **Answer Relevancy:** Does every factual assertion directly address what the user asked?

Step 1: Atomic Claim Decomposition
- Break the `Generated Answer` down into discrete, self-contained atomic factual claims.
- A claim is a single factual assertion. Compound sentences must be split into separate assertions.
  - *Example:* "LLM-as-a-judge is cheaper than humans and evaluates semantic nuances" becomes:
    - Claim 1: "LLM-as-a-judge is cheaper than human evaluators."
    - Claim 2: "LLM-as-a-judge evaluates semantic nuances."
- **Refusal / No-Answer Handling:** If the generated answer is a standard abstention (e.g., "I cannot answer this question based on the provided context"), extract **zero** claims (an empty list `[]`) and set `is_refusal: true`.

Step 2: Faithfulness Evaluation (`is_supported_by_context`)
- Compare each claim against the `Golden Context`.
- **Semantic Matching (Natural Language Inference):** Do NOT require exact string or verbatim matching. A claim is supported if the context directly states it, paraphrases it, or logically entails it.
- **Strict Grounding:** If a claim relies on external knowledge, unstated assumptions, or contradicts the context, mark it `false`.
- If a claim is supported semantically by the Golden Context, set `is_supported_by_context: true`. Otherwise, set `is_supported_by_context: false`.

Step 3: Answer Relevancy Evaluation (`is_relevant_to_question`)
- Compare each claim against the `User Question`.
- Determine whether the claim is necessary and directly contributes to answering what was asked.
- Disregard whether the claim is factually true or present in the context; evaluate purely whether it addresses the prompt's intent.
- Tangential observations, conversational filler, or lecture trivia that do not directly address the question must be marked `false`.
- If the claim directly helps answer the user's question, set `is_relevant_to_question: true`. Otherwise, set `is_relevant_to_question: false`.

Step 4: Strict Formatting & No Math
- **CRITICAL:** Do NOT calculate averages, fractions, division, or percentage scores. The evaluation pipeline calculates all mathematical metrics in pure Python.
- Your sole job is to return the decomposed claims and their boolean verdicts.

Input Template (Provided in User Turn)

User Question:
{question}

Golden Context:
{golden_context}

Generated Answer:
{generated_answer}
"""

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class ClaimEvaluation(BaseModel):
    claim: str
    is_supported_by_context: bool
    is_relevant_to_question: bool
    reasoning: str


class JudgeOutput(BaseModel):
    is_refusal: bool
    claims: list[ClaimEvaluation]
    evaluation_summary: str


# ---------------------------------------------------------------------------
# Judge Invocation with Retry
# ---------------------------------------------------------------------------
def _is_retryable(exc: BaseException) -> bool:
    """Retry on 503 Service Unavailable (ServerError) — never on 429 quota errors (ClientError)."""
    if isinstance(exc, ServerError):
        return exc.code == 503
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=30, max=120),
    stop=stop_after_attempt(3),
    reraise=True,
)
def call_judge(
    client: genai.Client, question: str, golden_context: str, generated_answer: str
) -> JudgeOutput:
    """Calls gemini-3.8-flash as judge with structured output and exponential backoff.

    Each tenacity attempt makes exactly ONE API call (SDK internal retry is disabled
    via http_options on the client). Only 503s are retried; 429s fail fast.
    """
    user_content = (
        f"User Question:\n{question}\n\n"
        f"Golden Context:\n{golden_context}\n\n"
        f"Generated Answer:\n{generated_answer}"
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_LLM_SYSTEM_PROMPT.strip(),
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=JudgeOutput,
        ),
    )

    if isinstance(response.parsed, JudgeOutput):
        return response.parsed
    return JudgeOutput.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# Deterministic Scoring (Pure Python)
# ---------------------------------------------------------------------------
def compute_scores(judge_output: JudgeOutput) -> tuple[float, float, bool]:
    """Computes faithfulness, relevance, and hallucination flag from judge verdicts."""
    if judge_output.is_refusal or len(judge_output.claims) == 0:
        faithfulness_score = 1.0
        relevance_score = 0.0
        has_hallucination = False
    else:
        total = len(judge_output.claims)
        supported = sum(1 for c in judge_output.claims if c.is_supported_by_context)
        relevant = sum(1 for c in judge_output.claims if c.is_relevant_to_question)

        faithfulness_score = round(supported / total, 3)
        relevance_score = round(relevant / total, 3)
        has_hallucination = supported < total

    return faithfulness_score, relevance_score, has_hallucination


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------
def print_summary_statistics() -> None:
    """Reads OUTPUT_FILE and computes aggregated summary metrics for RUN_ID."""
    if not OUTPUT_FILE.exists():
        log.warning("No results file found at %s — nothing to aggregate.", OUTPUT_FILE)
        return

    faithfulness_scores: list[float] = []
    relevance_scores: list[float] = []
    clean_pass_count = 0

    with OUTPUT_FILE.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("run_id") != RUN_ID:
                continue
            f_score = float(row["faithfulness_score"])
            r_score = float(row["relevance_score"])
            is_hallucinating = row["has_hallucination"].strip().lower() == "true"

            faithfulness_scores.append(f_score)
            relevance_scores.append(r_score)
            if not is_hallucinating:
                clean_pass_count += 1

    total = len(faithfulness_scores)
    if total == 0:
        log.warning("No rows found for run_id=%s.", RUN_ID)
        return

    avg_faithfulness = sum(faithfulness_scores) / total
    avg_relevance = sum(relevance_scores) / total
    clean_pass_pct = (clean_pass_count / total) * 100

    print("\n" + "=" * 54)
    print(f"  Evaluation Summary (run_id: {RUN_ID})")
    print("=" * 54)
    print(f"  Total questions evaluated : {total}")
    print(f"  Average Faithfulness      : {avg_faithfulness:.3f}")
    print(f"  Average Relevancy         : {avg_relevance:.3f}")
    print(f"  Clean Pass Rate           : {clean_pass_count}/{total} ({clean_pass_pct:.1f}%)")
    print("=" * 54 + "\n")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        log.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    if not INPUT_FILE.exists():
        log.error("Input file not found at: %s", INPUT_FILE)
        sys.exit(1)

    log.info("Loading generated answers from %s ...", INPUT_FILE)
    items = [
        json.loads(line)
        for line in INPUT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total = len(items)
    log.info("Loaded %d answers. Output file: %s (run_id: %s)", total, OUTPUT_FILE, RUN_ID)

    # Inspect CSV for completed items to support safe resuming
    completed_pairs: set[tuple[str, str]] = set()
    write_header = True
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                completed_pairs.add((row.get("run_id", ""), row.get("id", "")))
        write_header = len(completed_pairs) == 0
        log.info("Found %d completed items already in %s", len(completed_pairs), OUTPUT_FILE)

    # Disable SDK's built-in retry so each tenacity attempt = exactly 1 API call
    client = genai.Client(
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        )
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
            csv_file.flush()
            os.fsync(csv_file.fileno())

        for i, item in enumerate(items, start=1):
            qid = item["id"]
            question = item["question"]
            golden_context = item["golden_context"]
            generated_answer = item["generated_answer"]

            # Resume logic: skip if (RUN_ID, qid) is already evaluated
            if (RUN_ID, qid) in completed_pairs:
                print(f"[{i}/{total}] {qid}: Skipped (already exists in {RUN_ID})")
                continue

            # Invoke LLM Judge
            judge_output = call_judge(client, question, golden_context, generated_answer)

            # Pure Python deterministic scoring
            faithfulness_score, relevance_score, has_hallucination = compute_scores(judge_output)

            # Atomic, crash-safe CSV record persistence
            writer.writerow({
                "id": qid,
                "question": question,
                "faithfulness_score": faithfulness_score,
                "relevance_score": relevance_score,
                "has_hallucination": has_hallucination,
                "run_id": RUN_ID,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            csv_file.flush()
            os.fsync(csv_file.fileno())

            print(
                f"[{i}/{total}] {qid} evaluated | "
                f"Faithfulness: {faithfulness_score:.1f} | Relevance: {relevance_score:.1f} | "
                f"Sleeping {SLEEP_SECONDS}s..."
            )
            time.sleep(SLEEP_SECONDS)

    log.info("Evaluation complete for run_id=%s.", RUN_ID)
    print_summary_statistics()


if __name__ == "__main__":
    main()
