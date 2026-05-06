"""Training utilities and model definition for ACIE churn prediction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import mlflow
    import mlflow.pytorch
except ImportError:  # pragma: no cover - optional during inference-only environments
    mlflow = None

LOGGER = logging.getLogger("acie.training")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "feature_matrix.csv"
ARTIFACT_DIR = ROOT_DIR / "models" / "saved"
CATEGORICAL_COLUMNS = ["country", "acquisition_channel", "initial_plan"]

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class TabTransformer(nn.Module):
    """Transformer-based classifier for tabular churn prediction."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        transformed = self.transformer(embedded.unsqueeze(1)).squeeze(1)
        return self.classifier(transformed)


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    dataframe = pd.read_csv(DATA_PATH)
    features = dataframe.drop(columns=["customer_id", "churned", "ltv"]).copy()
    target = dataframe["churned"].copy()
    return dataframe, features, target


def preprocess_features(features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, dict[str, LabelEncoder]]:
    processed = features.copy()
    encoders: dict[str, LabelEncoder] = {}

    for column in CATEGORICAL_COLUMNS:
        if column in processed.columns:
            encoder = LabelEncoder()
            processed[column] = encoder.fit_transform(processed[column].astype(str))
            encoders[column] = encoder

    numerical_columns = [column for column in processed.columns if column not in CATEGORICAL_COLUMNS]
    scaler = StandardScaler()
    processed[numerical_columns] = scaler.fit_transform(processed[numerical_columns])
    return processed, scaler, encoders


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    features_temp, features_test, target_temp, target_test = train_test_split(
        features,
        target,
        test_size=0.15,
        random_state=42,
        stratify=target,
    )
    features_train, features_val, target_train, target_val = train_test_split(
        features_temp,
        target_temp,
        test_size=0.18,
        random_state=42,
        stratify=target_temp,
    )
    return features_train, features_val, features_test, target_train, target_val, target_test


def train_model(
    features_train: pd.DataFrame,
    features_val: pd.DataFrame,
    target_train: pd.Series,
    target_val: pd.Series,
) -> TabTransformer:
    train_tensor = torch.tensor(features_train.values, dtype=torch.float32)
    val_tensor = torch.tensor(features_val.values, dtype=torch.float32)
    train_labels = torch.tensor(target_train.values, dtype=torch.float32).reshape(-1, 1)

    model = TabTransformer(input_dim=features_train.shape[1])
    positive_weight = torch.tensor(
        [(len(target_train) - target_train.sum()) / max(target_train.sum(), 1)],
        dtype=torch.float32,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    best_val_auc = 0.0
    batch_size = 128
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(50):
        model.train()
        for start in range(0, len(train_tensor), batch_size):
            batch_x = train_tensor[start : start + batch_size]
            batch_y = train_labels[start : start + batch_size]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            val_probs = torch.sigmoid(val_outputs).numpy().flatten()
            val_auc = roc_auc_score(target_val.values, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            torch.save(model.state_dict(), ARTIFACT_DIR / "churn_model.pth")

        if (epoch + 1) % 10 == 0:
            LOGGER.info("Epoch %s/50 | best val AUC %.4f", epoch + 1, best_val_auc)

    model.load_state_dict(torch.load(ARTIFACT_DIR / "churn_model.pth", map_location="cpu", weights_only=True))
    model.eval()
    return model


def evaluate_model(model: TabTransformer, features_test: pd.DataFrame, target_test: pd.Series) -> dict[str, float]:
    test_tensor = torch.tensor(features_test.values, dtype=torch.float32)
    with torch.no_grad():
        raw_outputs = model(test_tensor)
        probabilities = torch.sigmoid(raw_outputs).numpy().flatten()
        predictions = (probabilities > 0.5).astype(int)

    metrics = {
        "test_auc": float(roc_auc_score(target_test.values, probabilities)),
        "test_f1": float(f1_score(target_test.values, predictions)),
        "test_accuracy": float(accuracy_score(target_test.values, predictions)),
    }
    LOGGER.info("Classification report:\n%s", classification_report(target_test.values, predictions))
    return metrics


def save_artifacts(
    feature_columns: list[str],
    scaler: StandardScaler,
    encoders: dict[str, LabelEncoder],
    metrics: dict[str, float],
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, ARTIFACT_DIR / "label_encoders.pkl")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")

    metadata = {
        "model_type": "TabTransformer",
        "input_features": feature_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numerical_columns": [column for column in feature_columns if column not in CATEGORICAL_COLUMNS],
        **metrics,
    }
    with (ARTIFACT_DIR / "churn_model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Starting ACIE churn model training pipeline")
    _, features, target = load_training_data()
    processed_features, scaler, encoders = preprocess_features(features)
    train_x, val_x, test_x, train_y, val_y, test_y = split_data(processed_features, target)

    model = train_model(train_x, val_x, train_y, val_y)
    metrics = evaluate_model(model, test_x, test_y)

    if mlflow is not None:
        with mlflow.start_run(run_name="acie_churn_training"):
            mlflow.log_param("model_type", "TabTransformer")
            mlflow.log_param("input_features", processed_features.shape[1])
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, artifact_path="model")
    else:
        LOGGER.warning("mlflow is not installed; skipping experiment tracking")

    save_artifacts(list(processed_features.columns), scaler, encoders, metrics)
    LOGGER.info("Saved ACIE artifacts with metrics: %s", metrics)


if __name__ == "__main__":
    main()
