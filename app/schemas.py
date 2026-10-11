"""API contract (guide section 6), as Pydantic v2 models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AnswerType = Literal["decomposition", "recommendation", "calculated", "not_in_data", "clarification_needed", "refused"]


class AnalyzeRequest(BaseModel):
    question: str = Field(..., description="Question or request in plain text")
    period: str | None = Field(None, description="Quarter to explain, for example 2026-Q3")
    compare_to: str | None = Field(None, description="Quarter to compare with, for example 2026-Q2")


class BridgeRow(BaseModel):
    cause: Literal["tariff", "volume", "product_mix", "weather", "waste"]
    pct_points: float
    range: list[float]
    kwh: float | None = None
    money: float | None = None


class Recommendation(BaseModel):
    action: str
    saving_kwh_per_year: list[float]
    saving_money_per_year: list[float]
    saving_co2e_t_per_year: list[float]
    payback_months: list[float] | None
    ease: str
    rationale: str
    evidence_refs: list[str]
    confidence: Literal["low", "medium", "high"]
    estimated_impact: dict[str, Any]
    requires_approval: bool = True
    rank: int | None = None
    meter: str | None = None
    finding: str | None = None
    avoidable_kw: float | None = None
    assumed_cost: list[float] | None = None
    assumed_recoverable_share: list[float] | None = None


class AnalyzeResponse(BaseModel):
    run_id: str
    answer_type: AnswerType
    summary: str
    bridge: list[BridgeRow] = []
    recommendations: list[Recommendation] = []
    assumptions: list[str] = []
    limitations: list[str] = []
    period: str | None = None
    compare_to: str | None = None
    data: dict[str, Any] | None = Field(None, description="Tool results behind the answer (for calculated answers and charts)")


class IngestResponse(BaseModel):
    rows_loaded: int
    rows_rejected: int
    status: str
    table: str | None = None
    detail: str | None = None
    duplicates_collapsed: int | None = None
    validation: dict[str, Any] | None = None
    rerun: dict[str, Any] | None = None
    periods: dict[str, Any] | None = None
