import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from rag_evals.retriever import build_retriever
from rag_evals.generator import (
    GENERATOR_SYSTEM_PROMPT,
    MODEL_ID,
    generate_response,
)

load_dotenv()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models (RAG Triad Ready)
# ---------------------------------------------------------------------------

class RAGResponse(BaseModel):
    """Structured response from the RAG pipeline containing all RAG Triad constituents.

    Constituents for RAG Triad:
      1. Context Relevance:  (query, context_chunks)
      2. Faithfulness / Groundedness: (context_chunks, answer)
      3. Answer Relevance:   (query, answer)
    """

    query: str
    answer: str
    context_chunks: List[str]
    context_documents: List[Dict[str, Any]] = Field(default_factory=list)
    latency_seconds: float
    model: str


# ---------------------------------------------------------------------------
# Pipeline Class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """End-to-end RAG pipeline connecting Chroma retriever with Gemini generator.

    Architecture principles:
      - Uses official Google GenAI SDK (google-genai / genai.Client) directly.
      - Hybrid retriever: CPU BGE-M3 query embeddings + GPU cross-encoder reranker.
      - Pure deterministic output ready for RAG Triad evaluation.
    """

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        model: str = MODEL_ID,
        client: Optional[genai.Client] = None,
        use_reranker: bool = True,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
    ):
        """Initializes the RAG Pipeline.

        Args:
            retriever: Optional pre-configured retriever. If None, instantiates
                build_retriever(use_reranker=use_reranker, top_n=top_k).
            model: Gemini model ID to use for generation (defaults to gemini-3.8-flash).
            client: Optional google.genai.Client instance. If None, instantiates one.
            use_reranker: Whether to use cross-encoder reranking (defaults to True).
            top_k: Number of context chunks to feed into the generator (defaults to 5).
            system_prompt: System prompt for generator. Defaults to GENERATOR_SYSTEM_PROMPT from generator.py.
        """
        self.model = model
        self.top_k = top_k
        self.system_prompt = system_prompt or GENERATOR_SYSTEM_PROMPT
        self.use_reranker = use_reranker

        # Initialize retriever if not provided
        self.retriever = retriever or build_retriever(
            use_reranker=use_reranker,
            top_n=top_k,
        )

        # Initialize official GenAI client directly
        self.client = client or genai.Client()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Retrieves candidate context documents for the query.

        Args:
            query: The user search query.
            top_k: Optional override for the number of documents to retrieve.

        Returns:
            List of langchain Document objects with page_content and metadata.
        """
        docs = self.retriever.invoke(query)
        k = top_k or self.top_k
        return docs[:k]

    def generate(self, query: str, context_chunks: List[str]) -> str:
        """Generates an answer from context chunks using the configured LLM.

        Args:
            query: The user question.
            context_chunks: List of retrieved text chunks.

        Returns:
            Grounded answer string.
        """
        return generate_response(
            question=query,
            context_chunks=context_chunks,
            client=self.client,
            model=self.model,
            system_prompt=self.system_prompt,
        )

    def query(self, query: str, top_k: Optional[int] = None) -> RAGResponse:
        """Executes full end-to-end RAG pipeline for a given query.

        Args:
            query: The user question.
            top_k: Optional override for number of context chunks.

        Returns:
            RAGResponse containing query, answer, context_chunks, metadata, and latency.
        """
        t0 = time.perf_counter()

        docs = self.retrieve(query, top_k=top_k)
        chunks = [doc.page_content for doc in docs]
        answer = self.generate(query, chunks)

        latency = round(time.perf_counter() - t0, 3)

        context_documents = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]

        return RAGResponse(
            query=query,
            answer=answer,
            context_chunks=chunks,
            context_documents=context_documents,
            latency_seconds=latency,
            model=self.model,
        )

    def run(self, query: str, top_k: Optional[int] = None) -> RAGResponse:
        """Alias for self.query(query, top_k=top_k)."""
        return self.query(query, top_k=top_k)

    def batch_query(
        self,
        queries: List[str],
        delay_seconds: float = 15.0,
        top_k: Optional[int] = None,
    ) -> List[RAGResponse]:
        """Runs the pipeline over a list of queries with inter-request rate limiting.

        Args:
            queries: List of question strings.
            delay_seconds: Delay between questions (15s default for free-tier 5 RPM limit).
            top_k: Optional override for number of retrieved chunks per query.

        Returns:
            List of RAGResponse objects.
        """
        results: List[RAGResponse] = []
        total = len(queries)

        for i, q in enumerate(queries, 1):
            if i > 1 and delay_seconds > 0:
                log.info("Rate limit: sleeping %0.1fs before next query...", delay_seconds)
                time.sleep(delay_seconds)

            log.info("[%d/%d] Querying: %s", i, total, q[:60])
            resp = self.query(q, top_k=top_k)
            results.append(resp)

        return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    default_query = "What is regression testing?"
    test_query = sys.argv[1] if len(sys.argv) > 1 else default_query

    print("=" * 70)
    print(f"Initializing RAG Pipeline (Model: {MODEL_ID}, Reranker: True)...")
    pipeline = RAGPipeline(use_reranker=True)

    print(f"\nRunning Query: '{test_query}'")
    result = pipeline.query(test_query)

    print("\n" + "=" * 70)
    print(f"Latency: {result.latency_seconds}s | Retrieved Chunks: {len(result.context_chunks)}")
    print("=" * 70)
    print("Answer:")
    print(result.answer)
    print("\nRetrieved Context Documents:")
    for idx, doc_meta in enumerate(result.context_documents, 1):
        session = doc_meta["metadata"].get("session", "unknown")
        snippet = doc_meta["content"][:120].replace("\n", " ")
        print(f"  [{idx}] Session {session}: {snippet}...")
    print("=" * 70)
