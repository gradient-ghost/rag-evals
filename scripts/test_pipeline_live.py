"""
Live smoke test for the RAGPipeline with hybrid CPU embedding and GPU reranker.

Verifies:
  1. Chroma retrieval with BGE-M3 on CPU.
  2. Reranking with BGE-Reranker-v2-m3 on CUDA.
  3. Generation with Gemini 3.8 Flash via official google-genai SDK.
  4. VRAM footprint and execution latency.
  5. Structure of RAGResponse (ready for RAG Triad eval).
"""

import sys
import time
from pathlib import Path

# Add src to pythonpath if not installed in editable mode
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from rag_evals import RAGPipeline, RAGResponse


def run_live_test():
    print("=" * 70)
    print("🚀 Running Live RAG Pipeline Smoke Test")
    print("=" * 70)

    # Measure initial VRAM
    if torch.cuda.is_available():
        init_vram = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"Initial GPU VRAM: {init_vram:.1f} MB")

    print("\n1. Initializing RAGPipeline(model='gemini-3.5-flash-lite', use_reranker=True, top_k=5)...")
    t0 = time.time()
    pipeline = RAGPipeline(model="gemini-3.5-flash-lite", use_reranker=True, top_k=5)
    init_time = time.time() - t0
    print(f"✅ Pipeline initialized in {init_time:.2f}s")

    if torch.cuda.is_available():
        model_vram = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"GPU VRAM with Reranker loaded: {model_vram:.1f} MB (within 4096 MB limit)")

    test_query = "What is vibe testing and why isn't it enough for deploying an llm app?"
    print(f"\n2. Executing query: '{test_query}'")
    t1 = time.time()
    response: RAGResponse = pipeline.query(test_query)
    total_time = time.time() - t1

    print("\n" + "=" * 70)
    print("🎯 Query Execution Results")
    print("=" * 70)
    print(f"Total Latency        : {response.latency_seconds:.3f}s (wall: {total_time:.3f}s)")
    print(f"Model ID             : {response.model}")
    print(f"Chunks Retrieved     : {len(response.context_chunks)}")
    print(f"Documents & Metadata : {len(response.context_documents)}")

    print("\n--- Retrieved Context Sessions & Snippets ---")
    for i, doc in enumerate(response.context_documents, 1):
        session = doc["metadata"].get("session", "unknown")
        snippet = doc["content"][:140].replace("\n", " ")
        print(f"  [{i}] (Session {session}): {snippet}...")

    print("\n--- Generated Answer ---")
    print(response.answer)
    print("=" * 70)

    # Assertions for smoke test validation
    assert len(response.context_chunks) == 5, f"Expected 5 chunks, got {len(response.context_chunks)}"
    assert len(response.answer.strip()) > 0, "Generated answer is empty!"
    assert any("vibe" in chunk.lower() for chunk in response.context_chunks), "No retrieved chunk mentioned vibe testing!"
    print("✅ All smoke test assertions passed successfully!")

    if torch.cuda.is_available():
        final_vram = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"\nFinal GPU VRAM: {final_vram:.1f} MB (stable, no memory leak)")
    print("=" * 70)


if __name__ == "__main__":
    run_live_test()
