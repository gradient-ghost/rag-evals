"""
Resolves null chunk_indices in generator_golden_dataset.jsonl by matching
each golden_context chunk against retrieved_chunks.jsonl using text overlap.
Updates the file in-place.
"""

import json
import os

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(ROOT, "evals", "retrieved_chunks.jsonl")
GOLDEN_PATH = os.path.join(ROOT, "goldens", "generator_golden_dataset.jsonl")

# --- Load chunks ---
chunks = []
with open(CHUNKS_PATH) as f:
    for line in f:
        chunks.append(json.loads(line))

# --- Load golden dataset ---
golden = []
with open(GOLDEN_PATH) as f:
    for line in f:
        golden.append(json.loads(line))


def best_match_index(ctx_text: str, session: str) -> int | None:
    """Return chunk_index of the chunk with the highest character overlap with ctx_text."""
    snippet = ctx_text[:60].strip()
    best_idx, best_score = None, 0
    for c in chunks:
        if c["metadata"]["session"] != session:
            continue
        page = c["page_content"]
        if snippet in page:
            score = sum(1 for ch in ctx_text if ch in page)
            if score > best_score:
                best_score, best_idx = score, c["chunk_index"]
        elif page[:60] in ctx_text:
            score = sum(1 for ch in page if ch in ctx_text)
            if score > best_score:
                best_score, best_idx = score, c["chunk_index"]
    return best_idx


# --- Resolve nulls ---
updated, failed = 0, []

for r in golden:
    if r["chunk_indices"] is not None:
        continue  # already has indices (g-series), skip

    session = r["source_sessions"][0]
    resolved = []
    for ctx in r["golden_context"]:
        idx = best_match_index(ctx, session)
        if idx is None:
            failed.append((r["id"], ctx[:60]))
        else:
            if idx not in resolved:   # deduplicate (e.g. f005, f007)
                resolved.append(idx)

    r["chunk_indices"] = sorted(resolved)
    updated += 1
    print(f"  {r['id']} → chunk_indices={r['chunk_indices']}")

# --- Write back ---
with open(GOLDEN_PATH, "w") as f:
    for r in golden:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n✅ Updated {updated} records in {GOLDEN_PATH}")

if failed:
    print(f"\n⚠️  {len(failed)} unmatched contexts:")
    for id_, snip in failed:
        print(f"   {id_}: \"{snip}...\"")
else:
    print("✅ All contexts matched successfully — no nulls remaining.")
