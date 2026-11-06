"""Unit tests for evals/contextual_relevancy_eval.py"""

import csv
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Add evals dir to path so we can import the module directly
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from contextual_relevancy_eval import (
    ContextClaimVerdict,
    ContextualRelevancyJudgeOutput,
    compute_relevancy_score,
    build_question_pool,
)


# ---------------------------------------------------------------------------
# compute_relevancy_score — pure deterministic arithmetic
# ---------------------------------------------------------------------------

def test_score_all_relevant():
    claims = [
        ContextClaimVerdict(claim="A", is_relevant=True,  reasoning="helps"),
        ContextClaimVerdict(claim="B", is_relevant=True,  reasoning="helps"),
        ContextClaimVerdict(claim="C", is_relevant=True,  reasoning="helps"),
    ]
    score, relevant, total = compute_relevancy_score(claims)
    assert score   == 1.0
    assert relevant == 3
    assert total    == 3


def test_score_all_irrelevant():
    claims = [
        ContextClaimVerdict(claim="X", is_relevant=False, reasoning="off-topic"),
        ContextClaimVerdict(claim="Y", is_relevant=False, reasoning="off-topic"),
    ]
    score, relevant, total = compute_relevancy_score(claims)
    assert score   == 0.0
    assert relevant == 0
    assert total    == 2


def test_score_mixed():
    claims = [
        ContextClaimVerdict(claim="A", is_relevant=True,  reasoning="relevant"),
        ContextClaimVerdict(claim="B", is_relevant=False, reasoning="off-topic"),
        ContextClaimVerdict(claim="C", is_relevant=True,  reasoning="relevant"),
        ContextClaimVerdict(claim="D", is_relevant=False, reasoning="off-topic"),
    ]
    score, relevant, total = compute_relevancy_score(claims)
    assert score   == 0.5
    assert relevant == 2
    assert total    == 4


def test_score_zero_division_protection():
    """Empty claims list must not raise ZeroDivisionError."""
    score, relevant, total = compute_relevancy_score([])
    assert score   == 0.0
    assert relevant == 0
    assert total    == 0


def test_score_rounds_to_three_decimals():
    # 1 of 3 = 0.333...
    claims = [
        ContextClaimVerdict(claim="A", is_relevant=True,  reasoning="r"),
        ContextClaimVerdict(claim="B", is_relevant=False, reasoning="r"),
        ContextClaimVerdict(claim="C", is_relevant=False, reasoning="r"),
    ]
    score, _, _ = compute_relevancy_score(claims)
    assert score == round(1/3, 3)


# ---------------------------------------------------------------------------
# build_question_pool — question loading & normalisation
# ---------------------------------------------------------------------------

FAKE_RETRIEVER_GOLDEN = json.dumps({
    "id": "Q002",
    "type": "single_concept",
    "sessions": [1],
    "question": "How do we define LLM evals?",
    "reference_answer": "...",
    "claims": ["claim A"],
}) + "\n"

FAKE_GENERATOR_GOLDEN = json.dumps({
    "id": "q001",
    "question": "what is vibe testing?",
    "golden_context": ["some context chunk"],
    "source_sessions": ["1"],
    "chunk_indices": [14],
}) + "\n"


def test_build_question_pool_normalises_fields(tmp_path):
    ret_file = tmp_path / "retriever_golden.jsonl"
    gen_file = tmp_path / "generator_golden.jsonl"
    ret_file.write_text(FAKE_RETRIEVER_GOLDEN)
    gen_file.write_text(FAKE_GENERATOR_GOLDEN)

    with patch("contextual_relevancy_eval.GOLDEN_PATHS", [ret_file, gen_file]):
        pool = build_question_pool()

    assert len(pool) == 2
    ids = {item["id"] for item in pool}
    assert "Q002" in ids
    assert "q001" in ids
    for item in pool:
        assert "id"       in item
        assert "question" in item
        assert len(item["question"]) > 0


def test_build_question_pool_skips_missing_files(tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"
    with patch("contextual_relevancy_eval.GOLDEN_PATHS", [missing]):
        pool = build_question_pool()
    assert pool == []


# ---------------------------------------------------------------------------
# Judge output schema validation
# ---------------------------------------------------------------------------

def test_judge_output_schema_validation():
    raw = {
        "claims": [
            {"claim": "LLM evals are systematic.", "is_relevant": True,  "reasoning": "Directly answers the question."},
            {"claim": "The speaker loves coffee.",  "is_relevant": False, "reasoning": "Off-topic filler."},
        ],
        "evaluation_summary": "Mostly on-topic context with some noise.",
    }
    output = ContextualRelevancyJudgeOutput.model_validate(raw)
    assert len(output.claims) == 2
    assert output.claims[0].is_relevant is True
    assert output.claims[1].is_relevant is False
    assert "noise" in output.evaluation_summary


def test_judge_output_empty_claims_is_valid():
    raw = {"claims": [], "evaluation_summary": "No claims extracted."}
    output = ContextualRelevancyJudgeOutput.model_validate(raw)
    assert output.claims == []


# ---------------------------------------------------------------------------
# Resume logic — completed pair detection
# ---------------------------------------------------------------------------

def test_resume_logic_skips_completed(tmp_path):
    """Verify that items already present in CSV are skipped."""
    csv_path = tmp_path / "contextual_relevancy_eval.csv"
    fieldnames = [
        "id", "question", "contextual_relevancy_score",
        "relevant_claims", "total_claims",
        "run_id", "timestamp",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({
            "id": "Q002", "question": "How do we define LLM evals?",
            "contextual_relevancy_score": "0.8",
            "relevant_claims": "4", "total_claims": "5",
            "run_id": "test_run", "timestamp": "2026-10-04T00:00:00+00:00",
        })

    completed: set[tuple[str, str]] = set()
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            completed.add((row.get("run_id", ""), row.get("id", "")))

    assert ("test_run", "Q002") in completed
    assert ("test_run", "Q999") not in completed
