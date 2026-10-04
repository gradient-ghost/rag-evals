"""rag_evals: End-to-end RAG pipeline and evaluation toolkit."""

from rag_evals.pipeline import RAGPipeline, RAGResponse
from rag_evals.retriever import build_retriever
from rag_evals.generator import (
    GENERATOR_SYSTEM_PROMPT,
    MODEL_ID,
    generate_response,
)

__all__ = [
    "RAGPipeline",
    "RAGResponse",
    "build_retriever",
    "generate_response",
    "GENERATOR_SYSTEM_PROMPT",
    "MODEL_ID",
]
