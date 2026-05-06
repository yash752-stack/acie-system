"""Model loading and inference utilities for the ACIE API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import torch

from models.churn_model_production import TabTransformer

LOGGER = logging.getLogger("acie.model_service")

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models" / "saved"

CATEGORICAL_COLUMNS = ["country", "acquisition_channel", "initial_plan"]
FRIENDLY_ALIASES = {
    "engagement_score": "engagement_score_latest",
    "support_tickets": "total_support_tickets",
}


class ModelServiceError(Exception):
    """Base exception for model service failures."""


class ModelNotLoadedError(ModelServiceError):
    """Raised when inference is requested before artifacts are available."""


class FeatureValidationError(ModelServiceError):
    """Raised when a request does not satisfy the feature contract."""


@dataclass
class PredictionResult:
    probability: float
    risk_level: str
    retention_priority: str
    used_features: int


class ModelService:
    """Encapsulates model artifact loading, preprocessing, and inference."""

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.feature_names: list[str] = []
        self.numerical_columns: list[str] = []
        self.metadata: dict = {}
        self.loaded = False

    def load(self) -> None:
        metadata_path = MODEL_DIR / "churn_model_metadata.json"
        scaler_path = MODEL_DIR / "scaler.pkl"
        encoders_path = MODEL_DIR / "label_encoders.pkl"
        weights_path = MODEL_DIR / "churn_model.pth"

        LOGGER.info("Loading ACIE model artifacts from %s", MODEL_DIR)

        if not all(path.exists() for path in (metadata_path, scaler_path, encoders_path, weights_path)):
            missing = [str(path.name) for path in (metadata_path, scaler_path, encoders_path, weights_path) if not path.exists()]
            raise ModelNotLoadedError(f"Missing model artifacts: {', '.join(missing)}")

        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        self.feature_names = self.metadata.get("input_features", [])
        self.numerical_columns = [
            feature for feature in self.feature_names if feature not in CATEGORICAL_COLUMNS
        ]
        self.scaler = joblib.load(scaler_path)
        self.encoders = joblib.load(encoders_path)

        self.model = TabTransformer(input_dim=len(self.feature_names))
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.loaded = True
        LOGGER.info("ACIE model loaded successfully with %s features", len(self.feature_names))

    def ensure_loaded(self) -> None:
        if not self.loaded or self.model is None or self.scaler is None:
            raise ModelNotLoadedError("Model artifacts are not loaded")

    def get_allowed_categories(self) -> dict[str, list[str]]:
        self.ensure_loaded()
        return {
            name: [str(value) for value in encoder.classes_.tolist()]
            for name, encoder in self.encoders.items()
        }

    def get_metrics(self) -> dict:
        self.ensure_loaded()
        return {
            "model": self.metadata.get("model_type", "TabTransformer"),
            "auc_roc": self.metadata.get("test_auc"),
            "f1_score": self.metadata.get("test_f1"),
            "accuracy": self.metadata.get("test_accuracy"),
            "feature_count": len(self.feature_names),
        }

    def get_feature_contract(self) -> dict:
        self.ensure_loaded()
        return {
            "features": self.feature_names,
            "count": len(self.feature_names),
            "categorical": CATEGORICAL_COLUMNS,
            "numerical": self.numerical_columns,
            "allowed_categories": self.get_allowed_categories(),
        }

    def _build_frame(self, payload: dict[str, float | int | str]) -> pd.DataFrame:
        self.ensure_loaded()
        normalized_payload = dict(payload)

        for alias, canonical_name in FRIENDLY_ALIASES.items():
            if alias in normalized_payload and canonical_name not in normalized_payload:
                normalized_payload[canonical_name] = normalized_payload[alias]

        missing_features = [feature for feature in self.feature_names if feature not in normalized_payload]
        if missing_features:
            raise FeatureValidationError(
                f"Missing required features: {', '.join(missing_features)}"
            )

        dataframe = pd.DataFrame([normalized_payload], columns=self.feature_names)

        for column in CATEGORICAL_COLUMNS:
            encoder = self.encoders.get(column)
            if encoder is None:
                raise ModelNotLoadedError(f"Encoder not found for column '{column}'")

            raw_value = str(dataframe.at[0, column])
            if raw_value not in encoder.classes_:
                allowed_values = ", ".join(map(str, encoder.classes_.tolist()))
                raise FeatureValidationError(
                    f"Invalid value '{raw_value}' for '{column}'. Allowed values: {allowed_values}"
                )
            dataframe[column] = encoder.transform(dataframe[column].astype(str))

        dataframe[self.numerical_columns] = self.scaler.transform(dataframe[self.numerical_columns])
        return dataframe

    def predict(self, payload: dict[str, float | int | str]) -> PredictionResult:
        dataframe = self._build_frame(payload)
        input_tensor = torch.tensor(dataframe[self.feature_names].values, dtype=torch.float32)

        with torch.no_grad():
            raw_output = self.model(input_tensor)
            probability = float(torch.sigmoid(raw_output).item())

        if probability >= 0.70:
            risk_level = "HIGH"
            retention_priority = "Save Account"
        elif probability >= 0.35:
            risk_level = "MEDIUM"
            retention_priority = "Intervene"
        else:
            risk_level = "LOW"
            retention_priority = "Monitor"

        return PredictionResult(
            probability=probability,
            risk_level=risk_level,
            retention_priority=retention_priority,
            used_features=len(self.feature_names),
        )
