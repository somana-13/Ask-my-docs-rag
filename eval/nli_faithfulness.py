from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from eval.claim_extractor import extract_claims


DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


@lru_cache(maxsize=1)
def get_nli_model(model_name: str = DEFAULT_NLI_MODEL) -> CrossEncoder:
    """
    Load the NLI cross-encoder once.

    The model predicts contradiction, entailment, and neutral scores for
    evidence-claim pairs.
    """
    return CrossEncoder(model_name)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def _get_label_mapping(model: CrossEncoder) -> dict[str, int]:
    """
    Infer label mapping from model config when available.

    Expected labels are usually contradiction, entailment, neutral.
    """
    config = getattr(model.model, "config", None)
    id2label = getattr(config, "id2label", {}) if config else {}

    mapping = {}
    for idx, label in id2label.items():
        mapping[str(label).lower()] = int(idx)

    # Fallback for common cross-encoder NLI models.
    if not mapping:
        mapping = {
            "contradiction": 0,
            "entailment": 1,
            "neutral": 2,
        }

    return mapping


def score_claim_against_evidence(
    claim: str,
    evidence_chunks: list[dict[str, Any]],
    model: CrossEncoder | None = None,
    entailment_threshold: float = 0.60,
    contradiction_threshold: float = 0.60,
) -> dict[str, Any]:
    """
    Score one claim against all retrieved chunks and choose the strongest evidence.
    """
    model = model or get_nli_model()

    if not evidence_chunks:
        return {
            "claim": claim,
            "verdict": "unsupported",
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 1.0,
            "best_source": None,
            "best_section": None,
            "best_evidence": None,
        }

    pairs = []
    for chunk in evidence_chunks:
        evidence_text = chunk.get("raw_text") or chunk.get("document") or ""
        pairs.append((evidence_text, claim))

    raw_scores = model.predict(pairs)

    label_mapping = _get_label_mapping(model)

    entailment_idx = label_mapping.get("entailment", 1)
    contradiction_idx = label_mapping.get("contradiction", 0)
    neutral_idx = label_mapping.get("neutral", 2)

    best_result = None

    for chunk, scores in zip(evidence_chunks, raw_scores):
        probs = _softmax(np.array(scores, dtype=float))

        entailment_score = float(probs[entailment_idx])
        contradiction_score = float(probs[contradiction_idx])
        neutral_score = float(probs[neutral_idx])

        if best_result is None or entailment_score > best_result["entailment_score"]:
            best_result = {
                "claim": claim,
                "entailment_score": entailment_score,
                "contradiction_score": contradiction_score,
                "neutral_score": neutral_score,
                "best_source": chunk.get("source_name"),
                "best_section": chunk.get("section_title"),
                "best_evidence": (chunk.get("raw_text") or chunk.get("document") or "")[:500],
            }

    assert best_result is not None

    if best_result["entailment_score"] >= entailment_threshold:
        verdict = "supported"
    elif best_result["contradiction_score"] >= contradiction_threshold:
        verdict = "contradicted"
    else:
        verdict = "unsupported"

    return {
        **best_result,
        "verdict": verdict,
    }


def evaluate_faithfulness(
    answer: str,
    evidence_chunks: list[dict[str, Any]],
    entailment_threshold: float = 0.60,
    contradiction_threshold: float = 0.60,
) -> dict[str, Any]:
    """
    Evaluate answer faithfulness using claim-level NLI entailment.

    Faithfulness = supported claims / total claims.
    """
    claims = extract_claims(answer)

    if not claims:
        return {
            "faithfulness_score": 0.0,
            "num_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "contradicted_claims": 0,
            "claim_results": [],
        }

    model = get_nli_model()

    claim_results = [
        score_claim_against_evidence(
            claim=claim,
            evidence_chunks=evidence_chunks,
            model=model,
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
        )
        for claim in claims
    ]

    supported = sum(1 for item in claim_results if item["verdict"] == "supported")
    unsupported = sum(1 for item in claim_results if item["verdict"] == "unsupported")
    contradicted = sum(1 for item in claim_results if item["verdict"] == "contradicted")

    faithfulness_score = supported / len(claim_results)

    return {
        "faithfulness_score": faithfulness_score,
        "num_claims": len(claim_results),
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "contradicted_claims": contradicted,
        "claim_results": claim_results,
    }
