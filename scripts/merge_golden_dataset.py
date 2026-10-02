"""
Merges faithfulness_dataset.json and golden_dataset_faithfulness.csv
into a single generator_golden_dataset.jsonl in the goldens/ folder.

Unified schema:
  - id              : str
  - question        : str
  - golden_context  : list[str]
  - source_sessions : list[str]
  - chunk_indices   : list[int] | null  (null for JSON-sourced entries)
"""

import json
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "goldens", "faithfulness_dataset.json")
CSV_PATH  = os.path.join(ROOT, "goldens", "golden_dataset_faithfulness.csv")
OUT_PATH  = os.path.join(ROOT, "goldens", "generator_golden_dataset.jsonl")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

records = []

# --- JSON source (drop f015 — duplicate of f008) ---
with open(JSON_PATH) as f:
    json_data = json.load(f)

for r in json_data:
    if r["id"] == "f015":
        print(f"  [SKIPPED] {r['id']} — near-duplicate of f008")
        continue
    records.append({
        "id":              r["id"],
        "question":        r["query"].strip(),
        "golden_context":  [c.strip() for c in r["ideal_context"]],
        "source_sessions": r["source_sessions"],   # already list[str]
        "chunk_indices":   None
    })

# --- CSV source ---
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        chunks  = [c.strip() for c in row["golden_context"].split("|||") if c.strip()]
        indices = [int(x.strip()) for x in row["chunk_indices"].split(",") if x.strip()]
        records.append({
            "id":              row["id"],
            "question":        row["question"].strip(),
            "golden_context":  chunks,
            "source_sessions": [row["source_sessions"].strip()],
            "chunk_indices":   indices
        })

# --- Validate consistent fields ---
expected_keys = set(records[0].keys())
for r in records:
    assert set(r.keys()) == expected_keys, f"Field mismatch in record {r['id']}"

# --- Write JSONL ---
with open(OUT_PATH, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n✅ Written {len(records)} records to {OUT_PATH}")
print(f"   Fields: {list(records[0].keys())}")
print(f"\n   ID range — JSON: f001–f014 | CSV: g001–g007")
