MERGED_SYSTEM_PROMPT = """You are an expert AI evaluator judging a Retrieval-Augmented Generation (RAG) system on two dimensions: Contextual Recall and Contextual Precision.

You will be given a user question, a numbered list of retrieved chunks (with chunk IDs), and a numbered list of golden claims (ground truth facts). You must produce TWO separate sets of verdicts in a single response.

# Part 1 — Recall (claim-level)
For EACH golden claim, determine if it can be directly attributed to or logically inferred from the retrieved context as a whole, without requiring outside knowledge. Mark `is_supported` true if supported, false if the context is missing the necessary information. Provide a brief, 1-sentence reasoning for each.

# Part 2 — Precision (chunk-level)
For EACH retrieved chunk, determine if it contains information that directly helps answer the user question. A chunk is relevant (`is_relevant`: true) if it contains ANY fact or context that would be used in the final answer, even partially. Mark false only if the chunk is off-topic, generic filler, or contributes nothing to answering this specific question. Provide a brief, 1-sentence reasoning for each.

# Rules
1. Judge recall against the FULL set of retrieved chunks combined — claims are supported by the context as a whole, not by any single chunk.
2. Judge precision per INDIVIDUAL chunk, independently of whether that chunk contributed to a supported claim.
3. You MUST return exactly one recall verdict per claim, in the same order as the claims are numbered. Use `claim_index` (1-indexed, matching the input numbering) to indicate which claim each verdict corresponds to. Do not merge, skip, or add claims.
4. You MUST return exactly one precision verdict per chunk, in the same order as given. Use `chunk_id` exactly as given in the input (do not renumber). Do not merge, skip, or add chunks.
5. Recall verdicts and precision verdicts are independent judgments — a chunk being irrelevant does not mean claims aren't supported, and vice versa. Do not let one judgment influence the other.
6. Return both verdict lists even if one of them contains only failures (all false) — never omit a list.

# Input Format:
<user_question>
[The question]
</user_question>

<retrieved_chunks>
[ID: 1] [Chunk 1 text...]
[ID: 2] [Chunk 2 text...]
</retrieved_chunks>

<golden_claims>
1. [Claim 1]
2. [Claim 2]
</golden_claims>

# Output Format:
Return a single JSON object with exactly two keys: `recall_verdicts` (list of {claim_index, is_supported, reasoning}) and `precision_verdicts` (list of {chunk_id, is_relevant, reasoning}).
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import csv
import json
import logging
import time
from datetime import datetime
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
JUDGE_MODEL = "gemini-3.5-flash-lite"   # model for this run

# run_id tags every CSV row — change this string when starting a new run config
RUN_ID = "k8_merged"

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
GOLDEN_PATH    = PROJECT_ROOT / "goldens/claude_golden_dataset.jsonl"
RETRIEVED_PATH = PROJECT_ROOT / "evals/retrieved_context.jsonl"
RESULTS_PATH   = PROJECT_ROOT / "evals/eval_results.csv"

# run_id and timestamp are appended so rows from different runs coexist in one file
CSV_COLUMNS = [
    "id", "question", "type", "sessions",
    "recall_score", "precision_score",
    "recall_mismatch", "precision_mismatch",
    "run_id", "timestamp",
]

# Retry config
_RETRY_INITIAL_BACKOFF = 4   # seconds
_RETRY_MAX_BACKOFF     = 60  # seconds
_RETRY_MAX_ATTEMPTS    = 5

# HTTP status codes that warrant a retry
_RETRYABLE_CODES = {429, 503}

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class ClaimVerdict(BaseModel):
    claim_index: int   # 1-indexed, matches input numbering
    is_supported: bool
    reasoning: str


class ChunkVerdict(BaseModel):
    chunk_id: int      # matches the [ID: N] given in the input, not re-numbered
    is_relevant: bool
    reasoning: str


# Single merged output — both verdict lists in one judge call
class MergedOutput(BaseModel):
    recall_verdicts:    list[ClaimVerdict]
    precision_verdicts: list[ChunkVerdict]


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
            # Log token usage from API response for comparison tracking
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                u = response.usage_metadata
                log.info(
                    "Token usage — prompt: %s, candidates: %s, total: %s",
                    u.prompt_token_count,
                    u.candidates_token_count,
                    u.total_token_count,
                )
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
    Load the golden dataset, run the retriever for each question (top-8
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
    # Override k=8 at call site — retriever.py default (k=5) is not touched
    retriever = build_retriever()
    retriever.search_kwargs = {"k": 8}

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
    For each question in RETRIEVED_PATH, make ONE merged judge call (recall +
    precision together) and write one row to RESULTS_PATH immediately (crash-safe).

    Resume logic: skips a question only if (RUN_ID, id) already exists together
    in the CSV — so re-running with a different run_id re-judges everything.
    """
    log.info("Phase 2 — loading retrieved context from %s …", RETRIEVED_PATH)
    items = [
        json.loads(line)
        for line in RETRIEVED_PATH.read_text().splitlines()
        if line.strip()
    ]

    # Build the set of already-completed (run_id, id) pairs from any existing CSV.
    # Keying on BOTH run_id + id means a new run_id always starts fresh.
    completed_pairs: set[tuple[str, str]] = set()
    write_header = True
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                completed_pairs.add((row.get("run_id", ""), row["id"]))
        write_header = len(completed_pairs) == 0
        log.info(
            "Resuming — %d (run_id, id) pairs already complete.",
            len(completed_pairs),
        )

    client = genai.Client()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = len(items)

    # Accumulate token counts across all questions for a final summary
    total_prompt_tokens    = 0
    total_candidate_tokens = 0
    total_tokens           = 0

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

            # Skip only if this exact (run_id, question id) pair is already done
            if (RUN_ID, qid) in completed_pairs:
                log.info("[%d/%d] Skipping already-judged: %s (run_id=%s)", i, total, qid, RUN_ID)
                continue

            print(f"\n[{i}/{total}] {question[:70]} …")

            # ── Build ONE user_content string — chunks appear once, not duplicated ──
            chunks_text = "\n\n".join(f"[ID: {j + 1}] {c}" for j, c in enumerate(chunks))
            claims_text = "\n".join(f"{j + 1}. {c}" for j, c in enumerate(claims))
            user_content = (
                f"<user_question>\n{question}\n</user_question>\n\n"
                f"<retrieved_chunks>\n{chunks_text}\n</retrieved_chunks>\n\n"
                f"<golden_claims>\n{claims_text}\n</golden_claims>"
            )

            # ── ONE merged judge call — both recall and precision in a single request ──
            merged_result, merged_err = _call_judge_with_retry(
                client, MergedOutput, MERGED_SYSTEM_PROMPT, user_content
            )
            # Single sleep after the single call (no inter-call sleep needed anymore)
            time.sleep(5)

            # ── Validate recall verdicts independently ────────────────────────────
            recall_mismatch = False
            if merged_result is None:
                recall_score    = 0.0
                recall_mismatch = True
                log.warning(
                    "[%s] Merged judge hard failure: %s — recording 0.0 for recall with MISMATCH.",
                    qid, merged_err,
                )
            else:
                expected_claim_indices = set(range(1, len(claims) + 1))
                returned_claim_indices = {v.claim_index for v in merged_result.recall_verdicts}
                if (
                    len(merged_result.recall_verdicts) != len(claims)
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
                        sum(v.is_supported for v in merged_result.recall_verdicts) / len(claims)
                    )

            # ── Validate precision verdicts independently (recall failure doesn't affect this) ──
            precision_mismatch = False
            if merged_result is None:
                precision_score    = 0.0
                precision_mismatch = True
                log.warning(
                    "[%s] Merged judge hard failure: %s — recording 0.0 for precision with MISMATCH.",
                    qid, merged_err,
                )
            else:
                expected_chunk_ids = set(range(1, len(chunks) + 1))
                returned_chunk_ids = {v.chunk_id for v in merged_result.precision_verdicts}
                if (
                    len(merged_result.precision_verdicts) != len(chunks)
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
                        sum(v.is_relevant for v in merged_result.precision_verdicts) / len(chunks)
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
                "id":                 qid,
                "question":           question,
                "type":               item.get("type", ""),
                "sessions":           json.dumps(item.get("sessions", [])),
                "recall_score":       f"{recall_score:.4f}",
                "precision_score":    f"{precision_score:.4f}",
                "recall_mismatch":    recall_mismatch,
                "precision_mismatch": precision_mismatch,
                "run_id":             RUN_ID,
                "timestamp":          datetime.now().isoformat(),  # written at flush time
            })
            csvfile.flush()

    log.info("Phase 2 complete.")


# ---------------------------------------------------------------------------
# Aggregation — read back the completed CSV (works on partial/resumed runs)
# ---------------------------------------------------------------------------
def compute_final_stats() -> None:
    """
    Read RESULTS_PATH and print average recall / precision for the current RUN_ID.
    MISMATCH rows are included (their score is 0.0 as recorded).
    Works correctly whether this is a full run or a resumed partial run.
    """
    if not RESULTS_PATH.exists():
        log.warning("No results file found at %s — nothing to aggregate.", RESULTS_PATH)
        return

    recall_scores    = []
    precision_scores = []
    mismatch_count   = 0
    total_tokens_sum = 0

    with RESULTS_PATH.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            # Only aggregate rows belonging to the current run
            if row.get("run_id", "") != RUN_ID:
                continue
            recall_scores.append(float(row["recall_score"]))
            precision_scores.append(float(row["precision_score"]))
            if row["recall_mismatch"] == "True" or row["precision_mismatch"] == "True":
                mismatch_count += 1

    n = len(recall_scores)
    if n == 0:
        log.warning("No rows found for run_id=%s — nothing to aggregate.", RUN_ID)
        return

    avg_recall    = sum(recall_scores)    / n
    avg_precision = sum(precision_scores) / n

    print("\n" + "=" * 52)
    print(f"  Run ID               : {RUN_ID}")
    print(f"  Questions evaluated  : {n}")
    print(f"  Questions w/ MISMATCH: {mismatch_count}")
    print(f"  Avg Recall           : {avg_recall:.4f}")
    print(f"  Avg Precision        : {avg_precision:.4f}")
    print("=" * 52)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Rename the old k=5 retrieved_context.jsonl so phase 1 runs fresh with k=8.
    # If already renamed (or never existed), this is a no-op.
    old_retrieved = PROJECT_ROOT / "evals/retrieved_context.jsonl"
    new_retrieved = PROJECT_ROOT / "evals/retrieved_context_k5_baseline.jsonl"
    if old_retrieved.exists() and not new_retrieved.exists():
        old_retrieved.rename(new_retrieved)
        log.info("Renamed %s → %s (forcing fresh k=8 retrieval).", old_retrieved.name, new_retrieved.name)

    run_phase1()
    run_phase2()
    compute_final_stats()