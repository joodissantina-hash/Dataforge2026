"""Pydantic v2 request/response models. Mirrored by hand in frontend/src/lib/types.ts."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from lib.simulator import CAPS

_TASK_IDS = ("delayed_recall", "changing_facts", "distractor_recall", "noisy_prediction")


class RunConfig(BaseModel):
    task: str = Field(default="changing_facts", pattern=f"^({'|'.join(_TASK_IDS)})$")
    seq_len: int = Field(default=16, ge=CAPS["seq_len"]["min"], le=CAPS["seq_len"]["max"])
    mem_slots: int = Field(default=4, ge=CAPS["mem_slots"]["min"], le=CAPS["mem_slots"]["max"])
    update_rate: float = Field(default=0.25, ge=CAPS["update_rate"]["min"],
                              le=CAPS["update_rate"]["max"])
    noise: float = Field(default=0.0, ge=CAPS["noise"]["min"], le=CAPS["noise"]["max"])
    decay: float = Field(default=0.01, ge=CAPS["decay"]["min"], le=CAPS["decay"]["max"])
    ood_query: bool = False
    reset_midway: bool = False
    seed: int = Field(default=7, ge=CAPS["seed"]["min"], le=CAPS["seed"]["max"])


class Token(BaseModel):
    index: int
    kind: str
    label: str
    key: Optional[str] = None
    value: Optional[str] = None
    is_stale_fact: bool = False
    is_noisy: bool = False


class SlotState(BaseModel):
    slot: int
    key: Optional[str] = None
    value: Optional[str] = None
    declared_value: Optional[str] = None
    strength: float
    age: int
    occupied: bool


class StepState(BaseModel):
    step: int
    token_index: int
    write_slot: Optional[int] = None
    event: str
    slots: List[SlotState] = Field(default_factory=list)
    energy: Optional[float] = None
    active_units: Optional[int] = None


class ModeResult(BaseModel):
    mode: str
    label: str
    prediction: str
    correct: bool
    confidence: float
    margin: float
    attention: List[float] = Field(default_factory=list)
    top_token_index: Optional[int] = None
    retrieved_key: Optional[str] = None
    retrieved_slot: Optional[int] = None
    retrieved_age: int = 0
    interference: float = 0.0
    state_floats: int
    state_scaling: str
    steps: List[StepState] = Field(default_factory=list)
    events: List[str] = Field(default_factory=list)
    verdict: str
    explanation: str
    energy: Optional[float] = None
    evictions: Optional[int] = None


class RunResult(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: RunConfig
    tokens: List[Token]
    query_key: str
    query_index: int
    ground_truth: str
    reset_at: Optional[int] = None
    notes: List[str] = Field(default_factory=list)
    modes: List[ModeResult]
    latency_ms: float
    computation_status: str = "live"
    engine: str = "server-numpy"
    caps: Dict[str, Any]


class LearnerResponseCreate(BaseModel):
    run_id: Optional[str] = None
    task: str = Field(min_length=1, max_length=64)
    predicted_failures: List[str] = Field(default_factory=list, max_length=16)
    explanation: str = Field(default="", max_length=1000)
    self_score: Optional[int] = Field(default=None, ge=0, le=5)


class LearnerResponse(BaseModel):
    id: bool
    accepted: bool
    rubric: List[str]
    stored: bool
    note: str


class MetaResponse(BaseModel):
    kind: str
    caps: Dict[str, Any]
    tasks: List[Dict[str, str]]
    routes: List[Dict[str, str]]
    model_math: List[Dict[str, str]]
    constants: Dict[str, Any]
    aspects: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    glossary: List[Dict[str, str]]
    limitations: List[str]
    ai_disclosure: str
    presets: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    engine: str
    db_connected: bool
    runs_performed: int