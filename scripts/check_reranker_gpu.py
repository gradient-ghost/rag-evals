"""
check_reranker_gpu.py
---------------------
Smoke-test that BAAI/bge-reranker-v2-m3 loads and runs on GPU.

Checks:
  1. CUDA availability and device name
  2. Model + tokenizer load onto GPU
  3. Forward pass with a sample (query, passage) pair
  4. Score output sanity

Run from project root:
    python scripts/check_reranker_gpu.py
"""

import sys
import time

# ── 1. CUDA check (no model needed) ─────────────────────────────────────────
try:
    import torch
except ImportError:
    print("❌  torch not installed — run: uv add torch")
    sys.exit(1)

print("=" * 56)
print("  GPU / CUDA status")
print("=" * 56)
print(f"  torch version     : {torch.__version__}")
print(f"  CUDA available    : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n⚠️  CUDA not available — reranker will fall back to CPU.")
    print("   CPU inference is ~10-30x slower but functionally correct.")
    device = "cpu"
else:
    device = "cuda"
    print(f"  CUDA device count : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(f"  GPU [{i}]            : {props.name}  ({total_mem_gb:.1f} GB VRAM)")

print()

# ── 2. Load cross-encoder ────────────────────────────────────────────────────
MODEL = "BAAI/bge-reranker-v2-m3"
print(f"  Loading model     : {MODEL}")
print(f"  Target device     : {device}")
print("  (first run downloads ~1 GB — subsequent runs use cache)\n")

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("❌  sentence-transformers not installed — run: uv add sentence-transformers")
    sys.exit(1)

t0 = time.time()
model = CrossEncoder(
    MODEL,
    device=device,
    trust_remote_code=True,   # bge-reranker-v2-m3 needs this
    max_length=512,
)
load_time = time.time() - t0
print(f"✅  Model loaded in {load_time:.1f}s")

# ── 3. Forward pass ──────────────────────────────────────────────────────────
pairs = [
    ("What is a golden dataset?",
     "A golden dataset is a curated set of input/output pairs used to evaluate LLM applications."),
    ("What is a golden dataset?",
     "Photosynthesis converts sunlight into chemical energy stored in glucose."),
]

print(f"\n  Running inference on {len(pairs)} (query, passage) pairs …")
t1 = time.time()
scores = model.predict(pairs)
infer_time = time.time() - t1

print(f"  Inference time    : {infer_time*1000:.1f} ms")
print(f"\n  Scores:")
for (q, p), s in zip(pairs, scores):
    relevance = "✅ relevant" if s > 0 else "❌ irrelevant"
    print(f"    score={s:+.4f}  [{relevance}]  passage: {p[:60]}…")

# ── 4. Sanity check ──────────────────────────────────────────────────────────
print()
if scores[0] > scores[1]:
    print("✅  Sanity check passed — relevant passage scored higher than irrelevant.")
else:
    print("⚠️  Sanity check FAILED — irrelevant passage scored >= relevant. Check model.")

# ── 5. VRAM usage (GPU only) ─────────────────────────────────────────────────
if device == "cuda":
    alloc_mb  = torch.cuda.memory_allocated()  / (1024 ** 2)
    reserv_mb = torch.cuda.memory_reserved()   / (1024 ** 2)
    print(f"\n  VRAM allocated    : {alloc_mb:.1f} MB")
    print(f"  VRAM reserved     : {reserv_mb:.1f} MB")

print("\n" + "=" * 56)
print(f"  Device to use in retriever.py : '{device}'")
print("=" * 56)
