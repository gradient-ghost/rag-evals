"""
Export all chunks from the Chroma store to retrieved_chunks.jsonl.
Each line is a JSON object with: chunk_index, chunk_id, page_content, and metadata.
"""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = str(PROJECT_ROOT / "chroma_store")
OUTPUT_FILE = PROJECT_ROOT / "retrieved_chunks.jsonl"


def main():
    print("Connecting to Chroma store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    print("Fetching all chunks...")
    result = store.get(include=["documents", "metadatas"])

    ids = result["ids"]
    documents = result["documents"]
    metadatas = result["metadatas"]

    # Sort by session number, then by their natural order within session
    records = list(zip(ids, documents, metadatas))
    records.sort(key=lambda x: (int(x[2].get("session", 0)), ids.index(x[0])))

    print(f"Writing {len(records)} chunks to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, (chunk_id, page_content, metadata) in enumerate(records):
            record = {
                "chunk_index": idx,
                "chunk_id": chunk_id,
                "page_content": page_content,
                "metadata": metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done! {len(records)} chunks written to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
