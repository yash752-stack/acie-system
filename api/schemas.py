"""Pydantic request and response schemas for the ACIE API."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class CustomerPredictionRequest(BaseModel):
    """Validated feature contract for churn inference."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 35,
                "country": "UK",
                "acquisition_channel": "Organic",
                "cac": 45.5,
                "initial_plan": "Basic",
                "tenure_months": 12,
                "avg_logins_3m": 25,
                "avg_sessions_3m": 30,
                "avg_session_duration_3m": 24.5,
                "avg_features_used_3m": 6,
                "engagement_score": 0.72,
                "payment_failures": 1,
                "payment_success_rate": 0.96,
                "support_tickets": 2,
            }
        },
    )

    age: float = Field(..., ge=18, le=100)
    country: str = Field(..., min_length=2, max_length=64)
    acquisition_channel: str = Field(..., min_length=2, max_length=64)
    cac: float = Field(..., ge=0, le=10_000)
    initial_plan: str = Field(..., min_length=1, max_length=64)
    tenure_months: float = Field(..., ge=0, le=240)
    avg_logins_3m: float = Field(..., ge=0, le=5_000)
    avg_sessions_3m: float = Field(..., ge=0, le=5_000)
    avg_session_duration_3m: float = Field(..., ge=0, le=1_000)
    avg_features_used_3m: float = Field(..., ge=0, le=500)
    engagement_score: float = Field(
        ...,
        ge=0,
        le=1,
        validation_alias=AliasChoices("engagement_score", "engagement_score_latest"),
    )
    payment_failures: int = Field(..., ge=0, le=100)
    payment_success_rate: float = Field(..., ge=0, le=1)
    support_tickets: int = Field(
        ...,
        ge=0,
        le=100,
        validation_alias=AliasChoices("support_tickets", "total_support_tickets"),
    )

    @field_validator("country", "acquisition_channel", "initial_plan")
    @classmethod
    def strip_text_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    def to_model_features(self) -> dict[str, float | int | str]:
        return {
            "age": self.age,
            "country": self.country,
            "acquisition_channel": self.acquisition_channel,
            "cac": self.cac,
            "initial_plan": self.initial_plan,
            "tenure_months": self.tenure_months,
            "avg_logins_3m": self.avg_logins_3m,
            "avg_sessions_3m": self.avg_sessions_3m,
            "avg_session_duration_3m": self.avg_session_duration_3m,
            "avg_features_used_3m": self.avg_features_used_3m,
            "engagement_score_latest": self.engagement_score,
            "payment_failures": self.payment_failures,
            "payment_success_rate": self.payment_success_rate,
            "total_support_tickets": self.support_tickets,
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    retention_priority: Literal["Monitor", "Intervene", "Save Account"]
    model_loaded: bool
    used_features: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    feature_count: int


class FeaturesResponse(BaseModel):
    features: list[str]
    count: int
    categorical: list[str]
    numerical: list[str]
    allowed_categories: dict[str, list[str]]


class MetricsResponse(BaseModel):
    model: str
    auc_roc: float | None = None
    f1_score: float | None = None
    accuracy: float | None = None
    feature_count: int
