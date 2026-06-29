"""Temporal Archetype Bank — bi-modal retrieval via Reciprocal Rank Fusion.

Shape channel : DTW distance between query series tail and each medoid.
Semantic channel : cosine similarity between embedded context and archetype description.
                   (placeholder — uses DTW ranking until embeddings are wired up)

Populate bank.json from TS-RAG medoid data to activate retrieval.
Schema per entry:
  {
    "id": "<str>",
    "medoid": [<float>, ...],
    "description": "<str>",
    "domain": "<str>"
  }
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_BANK_PATH = Path(__file__).parent / "bank.json"


def _load_bank() -> list[dict]:
    if _BANK_PATH.exists():
        return json.loads(_BANK_PATH.read_text())
    return []


def _dtw(a: list[float], b: list[float]) -> float:
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def _rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def retrieve_archetypes(series: list[float], context: str, k: int = 3) -> list[dict]:
    """Return top-k archetypes by RRF-fused shape + semantic similarity."""
    bank = _load_bank()
    if not bank:
        return []

    tail = series[-64:] if len(series) > 64 else series

    # Shape ranking via DTW
    dtw_scores = sorted(
        ((i, _dtw(tail, e["medoid"])) for i, e in enumerate(bank)),
        key=lambda x: x[1],
    )
    dtw_ranks = [i for i, _ in dtw_scores]

    # Semantic ranking — placeholder: mirrors DTW until embeddings are wired
    # TODO: embed context + descriptions, rank by cosine similarity
    semantic_ranks = dtw_ranks

    fused = _rrf([dtw_ranks, semantic_ranks])
    top_k_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
    return [bank[i] for i in top_k_ids]
