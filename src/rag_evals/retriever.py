import os
import re
import glob
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

load_dotenv()

# Cross-encoder model used for reranking — referenced by both build_retriever()
# and the two-pass eval helper rerank_docs().
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = str(PROJECT_ROOT / "data" / "raw")
DB_DIR = str(PROJECT_ROOT / "chroma_store")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_transcripts() -> List[Document]:
    """Read each transcript and throw away the VTT timestamps."""
    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)
        docs.append(Document(page_content=text, metadata={"session": session}))
    return docs


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def load_store() -> Chroma:
    """Chunk, embed once, keep it on disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_transcripts()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)
    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


# ---------------------------------------------------------------------------
# Reranker retriever
# ---------------------------------------------------------------------------

class RerankedRetriever(BaseRetriever):
    """Two-stage retriever: wide Chroma search → cross-encoder rerank → top-n.

    Implemented as a BaseRetriever subclass so it exposes the standard
    `.invoke()` interface and is compatible with the eval harness.
    """

    store: Chroma
    cross_encoder: CrossEncoder
    fetch_k: int = 16
    top_n: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # Stage 1 — broad similarity retrieval
        candidates = self.store.similarity_search(query, k=self.fetch_k)

        # Stage 2 — cross-encoder reranking
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Sort descending by score, keep top-n
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: self.top_n]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_retriever(use_reranker: bool = False) -> BaseRetriever:
    """Return a retriever over the Chroma store.

    Args:
        use_reranker: When False (default), returns a plain similarity retriever
            with k=5 — identical to the original behaviour.  When True, fetches
            k=16 candidates from Chroma then reranks with BAAI/bge-reranker-v2-m3
            and returns the top 5, trading latency for higher precision.
    """
    store = load_store()

    if not use_reranker:
        return store.as_retriever(search_kwargs={"k": 5})

    # ── Reranker path ─────────────────────────────────────────────────────────
    # NOTE: bge-reranker-v2-m3 (~2.1 GB) and bge-m3 embeddings (~3.5 GB) cannot
    # coexist on a small GPU. Use rerank_docs() from the eval harness for a
    # two-pass approach. build_retriever(use_reranker=True) is safe only when
    # the embedding model has been freed from VRAM beforehand.
    cross_encoder = CrossEncoder(
        RERANKER_MODEL_NAME,
        device="cuda",
        trust_remote_code=True,
        max_length=512,
    )
    return RerankedRetriever(store=store, cross_encoder=cross_encoder, fetch_k=16, top_n=5)


# ---------------------------------------------------------------------------
# Standalone reranking helper (used by the eval harness for two-pass GPU runs)
# ---------------------------------------------------------------------------

def rerank_docs(
    query: str,
    docs: List[Document],
    cross_encoder: CrossEncoder,
    top_n: int = 5,
) -> List[Document]:
    """Rerank *docs* against *query* using *cross_encoder* and return top-n.

    Intended for the eval harness two-pass pattern:
      1. Retrieve k=16 candidates with the embedding model → free VRAM.
      2. Load CrossEncoder on GPU.
      3. Call rerank_docs() per question.

    This keeps both large models off the GPU at the same time.
    """
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


if __name__ == "__main__":
    retriever = build_retriever()

    results = retriever.invoke("what is regression testing")

    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")