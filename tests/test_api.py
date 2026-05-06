"""API smoke tests for the ACIE churn prediction service."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.main import app

SAMPLE_PAYLOAD = json.loads((ROOT / "data" / "sample_input.json").read_text(encoding="utf-8"))


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in {"ok", "degraded"}
        assert "feature_count" in payload


def test_predict() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert response.status_code == 200
        payload = response.json()
        assert 0 <= payload["churn_probability"] <= 1
        assert payload["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert payload["used_features"] == 14
