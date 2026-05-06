"""FastAPI application exposing the ACIE churn prediction service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from api.schemas import (
    CustomerPredictionRequest,
    FeaturesResponse,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
)
from api.services.model_service import (
    FeatureValidationError,
    ModelNotLoadedError,
    ModelService,
    ModelServiceError,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("acie.api")
model_service = ModelService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        model_service.load()
    except Exception:
        LOGGER.exception("Failed to load model artifacts during startup")
    yield


app = FastAPI(
    title="ACIE Churn Prediction API",
    description="FastAPI inference service for the Autonomous Customer Intelligence Engine.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    return {
        "message": "ACIE Churn Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_service.loaded else "degraded",
        model_loaded=model_service.loaded,
        feature_count=len(model_service.feature_names),
    )


@app.get("/features", response_model=FeaturesResponse)
def get_features() -> FeaturesResponse:
    try:
        return FeaturesResponse(**model_service.get_feature_contract())
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    try:
        return MetricsResponse(**model_service.get_metrics())
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerPredictionRequest) -> PredictionResponse:
    try:
        result = model_service.predict(customer.to_model_features())
    except FeatureValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ModelServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected prediction failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected prediction failure",
        ) from exc

    return PredictionResponse(
        churn_probability=round(result.probability, 4),
        risk_level=result.risk_level,
        retention_priority=result.retention_priority,
        model_loaded=True,
        used_features=result.used_features,
    )
