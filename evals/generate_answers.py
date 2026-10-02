import json
import logging
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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
MODEL_NAME = "gemini-3.5-flash"
RUN_ID = "baseline_21"
INTER_QUESTION_DELAY = 15  # seconds (throttles to 4 RPM to stay safely within the 5 RPM peak limit)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "goldens/generator_golden_dataset.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data/contexts/generated_answers.jsonl"

SYSTEM_PROMPT = """
You are a factual, strictly grounded question-answering assistant. Your sole task is to answer the user's question using ONLY the facts explicitly provided in the retrieved context chunks.

Operating Rules:
1. Strict Grounding: Rely exclusively on the provided context. Do not incorporate prior training knowledge, unstated assumptions, or external facts. If a claim is not directly stated or logically entailed by the text, treat it as untrue.
2. Noise & Filler Filtering: The context may consist of raw, conversational lecture transcripts containing filler, repetitions, rhetorical tangents, or conversational banter. Disregard all conversational noise and extract only the concrete factual information relevant to the question.
3. Zero Speculation: Do not extrapolate, interpret vague analogies, or make educated guesses. Maintain strict factual fidelity to the source text.
4. Objective Delivery: State the answer directly and clearly. Avoid conversational preambles, meta-commentary, or citing chunk numbers (e.g., avoid "Based on chunk 1" or "The speaker says") unless explicitly asked.
5. Negative Constraint: If the provided context does not contain sufficient factual evidence to answer the question, state exactly: "I cannot answer this question based on the provided context."
"""

# ---------------------------------------------------------------------------
# Generation with Retry
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type(APIError),
    wait=wait_exponential(multiplier=2, min=15, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def call_generator(client: genai.Client, prompt: str) -> str:
    """Invokes the generator LLM with exponential backoff on APIError."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.strip(),
            temperature=0.0,
        ),
    )
    return response.text or ""


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        log.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    if not INPUT_PATH.exists():
        log.error("Input file not found at: %s", INPUT_PATH)
        sys.exit(1)

    log.info("Loading questions from %s ...", INPUT_PATH)
    items = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total = len(items)
    log.info("Loaded %d questions. Output file: %s (run_id: %s)", total, OUTPUT_PATH, RUN_ID)

    # Inspect output file for completed items to support resuming
    completed_pairs: set[tuple[str, str]] = set()
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    completed_pairs.add((record.get("run_id", ""), record["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        log.info("Found %d completed items already in %s", len(completed_pairs), OUTPUT_PATH)

    client = genai.Client()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("a", encoding="utf-8") as fh:
        for i, item in enumerate(items, start=1):
            qid = item["id"]
            question = item.get("question") or item.get("query", "")
            context = item.get("golden_context") or item.get("ideal_context", "")

            # Resume logic: skip if (RUN_ID, qid) is already saved
            if (RUN_ID, qid) in completed_pairs:
                print(f"[{i}/{total}] {qid}: Skipped (already exists in {RUN_ID})")
                continue

            # Format golden context
            if isinstance(context, list):
                context_str = "\n\n".join(context)
            else:
                context_str = str(context)

            # Build user turn prompt
            prompt = (
                f"Context:\n"
                f"---\n"
                f"{context_str}\n"
                f"---\n\n"
                f"Question: {question}"
            )

            # Generate answer
            answer = call_generator(client, prompt)

            # Atomic, crash-safe record persistence
            record = {
                "id": qid,
                "run_id": RUN_ID,
                "question": question,
                "golden_context": context_str,
                "generated_answer": answer,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

            print(
                f"[{i}/{total}] {qid}: Generated response (length: {len(answer)} chars) | Sleeping {INTER_QUESTION_DELAY}s..."
            )
            time.sleep(INTER_QUESTION_DELAY)

    log.info("Generation complete for run_id=%s.", RUN_ID)


if __name__ == "__main__":
    main()
