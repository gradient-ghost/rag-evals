RECALL_SYSTEM_PROMPT = """You are an expert AI evaluator judging the "Contextual Recall" of a Retrieval-Augmented Generation (RAG) system.
Your task is to determine if the provided retrieved context contains enough information to fully support a list of "golden claims" (ground truth facts).

# Instructions:
1. You will be given retrieved text chunks and a numbered list of golden claims.
2. For EACH claim, determine if it can be directly attributed to or logically inferred from the retrieved context, without requiring outside knowledge.
3. Mark `is_supported` true if supported, false if the context is missing the necessary information.
4. Provide a brief, 1-sentence reasoning for your decision.
5. You MUST return exactly one verdict per claim, in the same order as the claims are numbered. Do not merge, skip, or add claims. Use `claim_index` to indicate which numbered claim each verdict corresponds to (1-indexed, matching the input numbering).

# Input Format:
<retrieved_context>
[Chunk 1 text...]
[Chunk 2 text...]
</retrieved_context>

<golden_claims>
1. [Claim 1]
2. [Claim 2]
</golden_claims>
"""

PRECISION_SYSTEM_PROMPT = """You are an expert AI evaluator judging the "Contextual Precision" of a Retrieval-Augmented Generation (RAG) system.
Your task is to determine if specific retrieved text chunks are relevant and useful for answering a given user question.

# Instructions:
1. You will be given a user question and a numbered list of retrieved chunks (with chunk IDs).
2. For EACH chunk, determine if it contains information that directly helps answer the question.
3. A chunk is relevant (`is_relevant`: true) if it contains ANY fact or context that would be used in the final answer, even partially. Mark false only if the chunk is off-topic, generic filler, or contributes nothing to answering this specific question.
4. Provide a brief, 1-sentence reasoning for your decision.
5. You MUST return exactly one verdict per chunk, in the same order as given. Do not merge, skip, or add chunks. Use `chunk_id` exactly as given in the input — do not renumber.

# Input Format:
<user_question>
[The question]
</user_question>

<retrieved_chunks>
[ID: 1] [Chunk text...]
[ID: 2] [Chunk text...]
</retrieved_chunks>
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import csv
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Type, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
JUDGE_MODEL = "gemini-3.1-flash-lite"

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
GOLDEN_PATH    = PROJECT_ROOT / "goldens/claude_golden_dataset.jsonl"
RETRIEVED_PATH = PROJECT_ROOT / "evals/retrieved_context.jsonl"
RESULTS_PATH   = PROJECT_ROOT / "evals/eval_results.csv"

CSV_COLUMNS = [
    "id", "question", "type", "sessions",
    "recall_score", "precision_score",
    "recall_mismatch", "precision_mismatch",
]

# Retry config
_RETRY_INITIAL_BACKOFF = 4   # seconds
_RETRY_MAX_BACKOFF     = 60  # seconds
_RETRY_MAX_ATTEMPTS    = 5

# HTTP status codes that warrant a retry
_RETRYABLE_CODES = {429, 503}

# ---------------------------------------------------------------------------
# Pydantic schemas (index-based, not text-matched)
# ---------------------------------------------------------------------------
class ClaimVerdict(BaseModel):
    claim_index: int   # 1-indexed, matches input numbering
    is_supported: bool
    reasoning: str


class RecallOutput(BaseModel):
    verdicts: list[ClaimVerdict]


class ChunkVerdict(BaseModel):
    chunk_id: int      # matches the [ID: N] given in the input, not re-numbered
    is_relevant: bool
    reasoning: str


class PrecisionOutput(BaseModel):
    verdicts: list[ChunkVerdict]


# ---------------------------------------------------------------------------
# Retry wrapper (hand-rolled, no external dependency)
# ---------------------------------------------------------------------------
T = TypeVar("T", bound=BaseModel)


def _call_judge_with_retry(
    client: genai.Client,
    response_model: Type[T],
    system_prompt: str,
    user_content: str,
) -> Tuple[Optional[T], Optional[str]]:
    """
    Call the Gemini judge and parse the response into *response_model*.

    Returns (result, None) on success, (None, error_message) after all
    retries are exhausted.  Only ResourceExhausted (429) and
    ServiceUnavailable (503) are retried; all other APIErrors are treated
    as hard failures and returned as (None, msg) immediately.
    """
    backoff = _RETRY_INITIAL_BACKOFF
    for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=user_content)],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_json_schema=response_model.model_json_schema(),
                ),
            )
            parsed = response_model.model_validate_json(response.text)
            return parsed, None

        except errors.APIError as exc:
            if exc.code in _RETRYABLE_CODES:
                if attempt == _RETRY_MAX_ATTEMPTS:
                    msg = (
                        f"Judge call failed after {_RETRY_MAX_ATTEMPTS} attempts "
                        f"(HTTP {exc.code}): {exc.message}"
                    )
                    log.error(msg)
                    return None, msg
                log.warning(
                    "HTTP %s on attempt %d/%d — backing off %ds …",
                    exc.code, attempt, _RETRY_MAX_ATTEMPTS, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, _RETRY_MAX_BACKOFF)
            else:
                msg = f"Non-retryable API error (HTTP {exc.code}): {exc.message}"
                log.error(msg)
                return None, msg

        except Exception as exc:  # e.g. JSON parse / validation failure
            msg = f"Unexpected error during judge call: {exc}"
            log.error(msg)
            return None, msg

    # Should not be reached, but satisfies the type-checker
    return None, "Exhausted retry attempts"


# ---------------------------------------------------------------------------
# Phase 1 — Retrieval (no API judge calls, run once)
# ---------------------------------------------------------------------------
def run_phase1() -> None:
    """
    Load the golden dataset, run the retriever for each question (top-5
    chunks), and dump results to RETRIEVED_PATH as JSONL.

    If RETRIEVED_PATH already exists the entire phase is skipped so
    embeddings / the vector store are never re-queried unnecessarily.
    """
    if RETRIEVED_PATH.exists():
        log.info("Phase 1 skipped — %s already exists.", RETRIEVED_PATH)
        return

    log.info("Phase 1 — loading golden dataset from %s …", GOLDEN_PATH)
    goldens = [
        json.loads(line)
        for line in GOLDEN_PATH.read_text().splitlines()
        if line.strip()
    ]
    log.info("Loaded %d golden items.", len(goldens))

    # Import here so the heavy embedding model is only loaded when needed.
    from rag_evals.retriever import build_retriever

    log.info("Building retriever (chroma_store must exist at project root) …")
    retriever = build_retriever()

    RETRIEVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RETRIEVED_PATH.open("w", encoding="utf-8") as fh:
        for i, item in enumerate(goldens, start=1):
            question = item["question"]
            log.info("[%d/%d] Retrieving for: %s …", i, len(goldens), question[:60])
            docs   = retriever.invoke(question)
            chunks = [doc.page_content for doc in docs]
            record = {
                "id":       item["id"],
                "question": question,
                "type":     item.get("type", ""),
                "sessions": item.get("sessions", []),
                "claims":   item.get("claims", []),
                "chunks":   chunks,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Phase 1 complete — wrote %d records to %s.", len(goldens), RETRIEVED_PATH)


# ---------------------------------------------------------------------------
# Phase 2 — Judging (rate-limited, resumable)
# ---------------------------------------------------------------------------
def run_phase2() -> None:
    """
    For each question in RETRIEVED_PATH, call the recall and precision
    judges and write one row to RESULTS_PATH immediately (crash-safe).

    Already-completed question IDs (detected by reading RESULTS_PATH)
    are skipped so the run is safely resumable after a crash or interrupt.
    """
    log.info("Phase 2 — loading retrieved context from %s …", RETRIEVED_PATH)
    items = [
        json.loads(line)
        for line in RETRIEVED_PATH.read_text().splitlines()
        if line.strip()
    ]

    # Build the set of already-completed IDs from any existing CSV.
    completed_ids: set[str] = set()
    write_header = True
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                completed_ids.add(row["id"])
        # Header is already present if there are any rows
        write_header = len(completed_ids) == 0
        log.info(
            "Resuming — %d questions already complete, %d remaining.",
            len(completed_ids),
            len(items) - len(completed_ids),
        )

    client = genai.Client()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = len(items)

    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
            csvfile.flush()

        for i, item in enumerate(items, start=1):
            qid      = item["id"]
            question = item["question"]
            claims   = item["claims"]
            chunks   = item["chunks"]

            if qid in completed_ids:
                log.info("[%d/%d] Skipping already-judged: %s", i, total, qid)
                continue

            print(f"\n[{i}/{total}] {question[:70]} …")

            # ── Recall judge ─────────────────────────────────────────────
            context_text = "\n\n".join(chunks)
            claims_text  = "\n".join(f"{j + 1}. {c}" for j, c in enumerate(claims))
            recall_user  = (
                f"<retrieved_context>\n{context_text}\n</retrieved_context>\n\n"
                f"<golden_claims>\n{claims_text}\n</golden_claims>"
            )

            recall_result, recall_err = _call_judge_with_retry(
                client, RecallOutput, RECALL_SYSTEM_PROMPT, recall_user
            )
            time.sleep(5)

            recall_mismatch = False
            if recall_result is None:
                recall_score    = 0.0
                recall_mismatch = True
                log.warning(
                    "[%s] Recall hard failure: %s — recording 0.0 with MISMATCH.",
                    qid, recall_err,
                )
            else:
                expected_claim_indices = set(range(1, len(claims) + 1))
                returned_claim_indices = {v.claim_index for v in recall_result.verdicts}
                if (
                    len(recall_result.verdicts) != len(claims)
                    or returned_claim_indices != expected_claim_indices
                ):
                    log.warning(
                        "[%s] Recall verdict mismatch — expected indices %s, got %s. "
                        "Recording 0.0.",
                        qid,
                        sorted(expected_claim_indices),
                        sorted(returned_claim_indices),
                    )
                    recall_score    = 0.0
                    recall_mismatch = True
                else:
                    recall_score = (
                        sum(v.is_supported for v in recall_result.verdicts) / len(claims)
                    )

            # ── Precision judge ───────────────────────────────────────────
            chunks_text    = "\n\n".join(f"[ID: {j + 1}] {c}" for j, c in enumerate(chunks))
            precision_user = (
                f"<user_question>\n{question}\n</user_question>\n\n"
                f"<retrieved_chunks>\n{chunks_text}\n</retrieved_chunks>"
            )

            precision_result, precision_err = _call_judge_with_retry(
                client, PrecisionOutput, PRECISION_SYSTEM_PROMPT, precision_user
            )
            time.sleep(5)

            precision_mismatch = False
            if precision_result is None:
                precision_score    = 0.0
                precision_mismatch = True
                log.warning(
                    "[%s] Precision hard failure: %s — recording 0.0 with MISMATCH.",
                    qid, precision_err,
                )
            else:
                expected_chunk_ids = set(range(1, len(chunks) + 1))
                returned_chunk_ids = {v.chunk_id for v in precision_result.verdicts}
                if (
                    len(precision_result.verdicts) != len(chunks)
                    or returned_chunk_ids != expected_chunk_ids
                ):
                    log.warning(
                        "[%s] Precision verdict mismatch — expected IDs %s, got %s. "
                        "Recording 0.0.",
                        qid,
                        sorted(expected_chunk_ids),
                        sorted(returned_chunk_ids),
                    )
                    precision_score    = 0.0
                    precision_mismatch = True
                else:
                    precision_score = (
                        sum(v.is_relevant for v in precision_result.verdicts) / len(chunks)
                    )

            # ── Progress line ─────────────────────────────────────────────
            flags = []
            if recall_mismatch:
                flags.append("recall=MISMATCH")
            if precision_mismatch:
                flags.append("precision=MISMATCH")
            mismatch_tag = f"  [{', '.join(flags)}]" if flags else ""
            print(
                f"  recall={recall_score:.3f}  precision={precision_score:.3f}"
                f"{mismatch_tag}"
            )

            # ── Write row immediately and flush (crash-safe) ──────────────
            writer.writerow({
                "id":                qid,
                "question":          question,
                "type":              item.get("type", ""),
                "sessions":          json.dumps(item.get("sessions", [])),
                "recall_score":      f"{recall_score:.4f}",
                "precision_score":   f"{precision_score:.4f}",
                "recall_mismatch":   recall_mismatch,
                "precision_mismatch": precision_mismatch,
            })
            csvfile.flush()

    log.info("Phase 2 complete.")


# ---------------------------------------------------------------------------
# Aggregation — read back the completed CSV (works on partial/resumed runs)
# ---------------------------------------------------------------------------
def compute_final_stats() -> None:
    """
    Read RESULTS_PATH and print average recall / precision.
    MISMATCH rows are included (their score is 0.0 as recorded).
    Works correctly whether this is a full run or a resumed partial run.
    """
    if not RESULTS_PATH.exists():
        log.warning("No results file found at %s — nothing to aggregate.", RESULTS_PATH)
        return

    recall_scores    = []
    precision_scores = []
    mismatch_count   = 0

    with RESULTS_PATH.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            recall_scores.append(float(row["recall_score"]))
            precision_scores.append(float(row["precision_score"]))
            if row["recall_mismatch"] == "True" or row["precision_mismatch"] == "True":
                mismatch_count += 1

    n = len(recall_scores)
    if n == 0:
        log.warning("Results file is empty — nothing to aggregate.")
        return

    avg_recall    = sum(recall_scores)    / n
    avg_precision = sum(precision_scores) / n

    print("\n" + "=" * 52)
    print(f"  Questions evaluated  : {n}")
    print(f"  Questions w/ MISMATCH: {mismatch_count}")
    print(f"  Avg Recall           : {avg_recall:.4f}")
    print(f"  Avg Precision        : {avg_precision:.4f}")
    print("=" * 52)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_phase1()
    run_phase2()
    compute_final_stats()
