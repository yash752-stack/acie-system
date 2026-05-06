"""CLI helper for running a local churn prediction from JSON input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from api.schemas import CustomerPredictionRequest
from api.services.model_service import ModelService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local ACIE churn prediction.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sample_input.json"),
        help="Path to a JSON payload matching the ACIE API contract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    request = CustomerPredictionRequest.model_validate(payload)
    service = ModelService()
    service.load()
    result = service.predict(request.to_model_features())

    response = {
        "churn_probability": round(result.probability, 4),
        "risk_level": result.risk_level,
        "retention_priority": result.retention_priority,
        "used_features": result.used_features,
    }
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
