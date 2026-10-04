GENERATOR_SYSTEM_PROMPT = """You are a factual, strictly grounded question-answering assistant. Your sole task is to answer the user's question using ONLY the facts explicitly provided in the retrieved context chunks.

Operating Rules:
1. Strict Grounding: Rely exclusively on the provided context. Do not incorporate prior training knowledge, unstated assumptions, or external facts. If a claim is not directly stated or logically entailed by the text, treat it as untrue.
2. Noise & Filler Filtering: The context may consist of raw, conversational lecture transcripts containing filler, repetitions, rhetorical tangents, or conversational banter. Disregard all conversational noise and extract only the concrete factual information relevant to the question.
3. Zero Speculation: Do not extrapolate, interpret vague analogies, or make educated guesses. Maintain strict factual fidelity to the source text.
4. Objective Delivery: State the answer directly and clearly. Avoid conversational preambles, meta-commentary, or citing chunk numbers (e.g., avoid "Based on chunk 1" or "The speaker says") unless explicitly asked.
5. Negative Constraint: If the provided context does not contain sufficient factual evidence to answer the question, state exactly: 'I cannot answer this question based on the provided context.'"""

import os
import time
import logging
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Suppress HuggingFace / sentence-transformers output before importing them
# ---------------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = str(PROJECT_ROOT / "chroma_store")
MODEL_ID = os.getenv("GENERATOR_MODEL", "gemini-3.5-flash-lite")
K = 8
INTER_QUESTION_DELAY = 15   # seconds between questions (free tier: 5 req/min)
MAX_RETRIES = 5              # retry attempts on 503
BASE_RETRY_DELAY = 5         # seconds — doubles each attempt (5 → 10 → 20 → 40)

from typing import Optional

# ---------------------------------------------------------------------------
# Lazy embedding function loader (avoids VRAM/RAM overhead when importing generator)
# ---------------------------------------------------------------------------

_EMBEDDING_FN = None


def _get_embedding_fn() -> SentenceTransformerEmbeddingFunction:
    """Lazily loads the embedding model only if legacy standalone generate_answer() is called."""
    global _EMBEDDING_FN
    if _EMBEDDING_FN is None:
        print("Loading embedding model for generator...", end="", flush=True)
        _EMBEDDING_FN = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3",
            normalize_embeddings=True,
        )
        print(" done.")
    return _EMBEDDING_FN


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
) -> types.GenerateContentResponse:
    """Calls generate_content with exponential backoff on 503 overload errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            err = str(e).lower()
            is_503 = "503" in err or "overloaded" in err or "service unavailable" in err
            if is_503 and attempt < MAX_RETRIES - 1:
                wait = BASE_RETRY_DELAY * (2 ** attempt)
                print(f"  [503] Model overloaded — retrying in {wait}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Core Generation Function
# ---------------------------------------------------------------------------

def generate_response(
    question: str,
    context_chunks: list[str],
    client: Optional[genai.Client] = None,
    model: str = MODEL_ID,
    system_prompt: str = GENERATOR_SYSTEM_PROMPT,
) -> str:
    """Formats prompt with context chunks and calls Gemini with the strict system prompt.

    Args:
        question: The user question.
        context_chunks: The retrieved context chunks to ground the answer.
        client: Optional google.genai.Client instance. If None, instantiates one.
        model: Target Gemini model ID (defaults to gemini-3.8-flash).
        system_prompt: System prompt instructing the model (defaults to GENERATOR_SYSTEM_PROMPT).

    Returns:
        The generated answer string.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Export it before running: "
            "export GEMINI_API_KEY=<your-key>"
        )

    # --- Prompt formatting ---
    context_block = "\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks))
    user_prompt = (
        f"Context:\n---\n{context_block}\n---\n\n"
        f"Question: {question}"
    )

    if client is None:
        client = genai.Client()

    response = _generate_with_retry(
        client=client,
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
        ),
    )

    return response.text or ""


# ---------------------------------------------------------------------------
# Legacy Standalone function (kept for backward compatibility)
# ---------------------------------------------------------------------------

def generate_answer(question: str, collection_name: str = "langchain") -> dict:
    """
    1. Queries the Chroma collection with k=8.
    2. Formats the user prompt with retrieved context chunks.
    3. Calls gemini-3.8-flash with the system instruction.
    4. Returns a dict with 'question', 'context_chunks', and 'response'.
    """
    # Guard: API key
    if not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Export it before running: "
            "export GEMINI_API_KEY=<your-key>"
        )

    # --- Retrieval ---
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=_get_embedding_fn(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Chroma collection '{collection_name}' from '{CHROMA_PATH}'. "
            "Make sure the vector store has been initialised by running the retriever first."
        ) from exc

    results = collection.query(query_texts=[question], n_results=K)
    chunks: list[str] = results["documents"][0]

    answer = generate_response(question=question, context_chunks=chunks)

    return {
        "question": question,
        "context_chunks": chunks,
        "response": answer,
    }


# ---------------------------------------------------------------------------
# __main__ — test run with one question of each type
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_QUESTIONS = [
        {
            "type": "single_concept",
            "id": "Q002",
            "question": "How do we define LLM evals, and what are the key characteristics?",
        },
        {
            "type": "cross_session",
            "id": "Q051",
            "question": (
                "How do offline evals, deployment, online evals and the golden dataset "
                "fit together as one loop?"
            ),
        },
        {
            "type": "comparison",
            "id": "Q081",
            "question": "What's the difference between model evals and application evals?",
        },
    ]

    for idx, item in enumerate(TEST_QUESTIONS):
        # Pause between questions to respect free-tier rate limits
        if idx > 0:
            print(f"\n  [rate limit] Waiting {INTER_QUESTION_DELAY}s before next request...")
            time.sleep(INTER_QUESTION_DELAY)

        print(f"\n{'=' * 70}")
        print(f"▶ [{item['type']}] ({item['id']})")
        print(f"  Question : {item['question']}")

        try:
            result = generate_answer(item["question"])
            print(f"  Chunks retrieved: {len(result['context_chunks'])}")
            print(f"\n  Response:\n{result['response']}")
        except EnvironmentError as e:
            print(f"\n  [ERROR] Missing API key — {e}")
            break
        except RuntimeError as e:
            print(f"\n  [ERROR] Chroma not ready — {e}")
            break
        except Exception as e:
            print(f"\n  [ERROR] Unexpected error — {e}")

    print(f"\n{'=' * 70}")
