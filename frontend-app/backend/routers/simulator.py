"""Deterministic experiment endpoints for the BDH explainer.

All routes hang off this APIRouter, which server.py folds into api_router (/api).
Parameter bounds are enforced by the Pydantic model, so an out-of-range request is
rejected with 422 before any computation happens - no parameter value can be used
to make the simulator expensive.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List

from fastapi import APIRouter, HTTPException, Request

from lib import evidence as ev
from lib.db import db
from lib.simulator import (
    ABSTAIN_COS, BETA_RECENCY, CAPS, DIM, ETA_HEBB, K_SPARSE, MODE_LABELS,
    TASKS, TAU_MATCH, simulate,
)
from models.simulation import (
    HealthResponse, LearnerResponse, LearnerResponseCreate, MetaResponse,
    RunConfig, RunResult,
)

router = APIRouter(tags=["simulator"])

CLAIM = (
    "When a task requires information from earlier context, a persistent learned state "
    "can preserve useful information without replaying the entire context - but it can "
    "also accumulate stale or incorrect state."
)

MODE_CARDS: List[Dict[str, str]] = [
    {
        "id": "attention",
        "label": MODE_LABELS["attention"],
        "kind": "baseline",
        "state": "O(N · d) - grows with every token",
        "summary": "Keeps every key/value pair. Nothing is compressed, so nothing goes stale. "
                   "You pay for that with a state that grows without bound.",
        "honesty": "Single head, hand-set recency bias, no learned projections. A trained "
                   "attention layer would learn this behaviour.",
    },
    {
        "id": "memory",
        "label": MODE_LABELS["memory"],
        "kind": "toy",
        "state": "O(M · d) - constant in N",
        "summary": "M addressable slots written with a convex blend at rate alpha. Constant "
                   "cost in sequence length; the blend is exactly where staleness enters.",
        "honesty": "Addressable slots are an educational abstraction. Real recurrent models "
                   "hold distributed superposed state with no slot addresses.",
    },
    {
        "id": "bdh",
        "label": MODE_LABELS["bdh"],
        "kind": "bdh_inspired",
        "state": "O(d²) - constant in N",
        "summary": "Sparse positive activations write additive outer-product traces into a "
                   "mutable synaptic matrix at inference time. Newer traces dominate without "
                   "needing a large update rate.",
        "honesty": "INSPIRED BY BDH, NOT BDH. No neuron-level graph, no low-rank factorised "
                   "interaction matrices, no layers, no training. Do not cite as a "
                   "reimplementation.",
    },
]

MODEL_MATH: Dict[str, str] = {
    "embedding": "e(s) = normalize(g(sha256(s))) in R^32 - fixed hashed table, never trained.",
    "attention": "score_i = (q·k_i)/sqrt(d) + 0.55·(i/N);  a = softmax(score);  "
                 "readout = Σ a_i · e(value_i)",
    "memory_write": "j = argmax_j cos(k, Kmem_j) if > 0.55 else free slot else argmin_j "
                    "strength_j;  Kmem_j = (1-α)Kmem_j + αk;  S_j = (1-α)S_j + α·e(v)",
    "memory_decay": "S = (1-λ)S and strength = (1-λ)strength every step; age_j = age_j + 1",
    "memory_read": "j* = argmax_j cos(q, Kmem_j);  readout = S_j*",
    "bdh_sparsify": "x = normalize(topk(relu(e[·]), 6)) - sparse and strictly positive",
    "bdh_write": "σ = (1-λ)σ + η·outer(x_v, x_k)    [additive Hebbian fast weight]",
    "bdh_read": "readout = σ · x_q",
    "decode": "nearest neighbour over the value table by cosine; if best cosine < 0.16 the "
              "mode answers UNKNOWN rather than guessing",
    "staleness": "After v1 -> v2 a blended slot holds (1-α)·e(v1) + α·e(v2). For α < 0.5 the "
                 "nearest value is still v1, so the model confidently returns a stale answer.",
}

# --- naive in-process rate limit: 60 runs / 60s / client -------------------------
_HITS: Dict[str, Deque[float]] = defaultdict(deque)
_WINDOW_S = 60.0
_MAX_HITS = 60


def _rate_limit(request: Request) -> None:
    who = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _HITS[who]
    while bucket and now - bucket[0] > _WINDOW_S:
        bucket.popleft()
    if len(bucket) >= _MAX_HITS:
        raise HTTPException(status_code=429, detail="Rate limit: 60 runs per minute per client.")
    bucket.append(now)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    connected, count = True, 0
    try:
        count = await db.experiment_runs.count_documents({})
    except Exception:
        connected = False
    return HealthResponse(status="ok", engine="server-numpy",
                          db_connected=connected, runs_recorded=count)


@router.get("/meta", response_model=MetaResponse)
async def meta() -> MetaResponse:
    return MetaResponse(
        claim=CLAIM,
        caps=CAPS,
        tasks=TASKS,
        modes=MODE_CARDS,
        model_math=MODEL_MATH,
        constants={
            "dim": DIM, "recency_beta": BETA_RECENCY, "match_threshold": TAU_MATCH,
            "sparse_topk": K_SPARSE, "hebbian_gain": ETA_HEBB, "abstain_cosine": ABSTAIN_COS,
            "rate_limit": f"{_MAX_HITS} runs / {int(_WINDOW_S)}s per client",
        },
        papers=ev.PAPERS,
        evidence=ev.CLAIMS,
        glossary=ev.GLOSSARY,
        limitations=ev.LIMITATIONS,
        ai_disclosure=ev.AI_DISCLOSURE,
        presets=ev.FAILURE_PRESETS,
    )


@router.post("/run", response_model=RunResult)
async def run(config: RunConfig, request: Request) -> RunResult:
    _rate_limit(request)
    started = time.perf_counter()
    try:
        out = simulate(config.model_dump())
    except Exception as exc:  # pragma: no cover - defensive
       raise HTTPException(status_code=500, detail=f"simulation failed: {exc}") from exc
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

    result = RunResult(config=config, latency_ms=latency_ms, caps=CAPS, **out)

    try:
        await db.experiment_runs.insert_one({
            "id": result.run_id,
            "config": config.model_dump(),
            "ground_truth": result.ground_truth,
            "outcomes": {m.mode: {"prediction": m.prediction, "correct": m.correct}
                         for m in result.modes},
            "latency_ms": latency_ms,
        })
    except Exception:
        pass  # metadata logging is best-effort; a run must never fail because of it

    return result


@router.get("/runs/{run_id}", response_model=Dict[str, Any])
async def get_run(run_id: str) -> Dict[str, Any]:
    doc = await db.experiment_runs.find_one({"id": run_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="run not found")
    return doc


RUBRIC = [
    "Name the mechanism, not just the outcome: blended write, eviction, decay, noise "
    "accumulation, or reset.",
    "Say what the state COST was: O(N·d) for attention vs a constant state for the "
    "persistent modes.",
    "State whether the ground truth needed the newest binding or the only binding.",
    "Say what you would change to flip the result (raise alpha, add slots, drop decay).",
]


@router.post("/responses", response_model=LearnerResponse)
async def submit_response(payload: LearnerResponseCreate) -> LearnerResponse:
    rid = str(uuid.uuid4())
    stored = False
    try:
        await db.learner_responses.insert_one({
            "id": rid,
            "run_id": payload.run_id,
            "task": payload.task,
            "predicted_failures": payload.predicted_failures,
            "explanation": payload.explanation,
            "self_score": payload.self_score,
        })
        stored = True
    except Exception:
        stored = False
    return LearnerResponse(
        id=rid, accepted=True, rubric=RUBRIC, stored=stored,
        note="Anonymous by design: no account, no cookie, no IP, no personal data is stored "
             "with your answer - only the text and the config you ran.",
    )