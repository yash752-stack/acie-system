from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np
import json
import os

app = FastAPI(title="ACIE Churn Prediction API")

# Define the model class (must match training)
class TabTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Global variables
model = None
scaler = None
encoders = None
feature_names = []
model_loaded = False

# Load everything at startup
@app.on_event("startup")
def startup_event():
    global model, scaler, encoders, feature_names, model_loaded
    
    print("\n" + "="*50)
    print("LOADING MODEL ARTIFACTS")
    print("="*50)
    
    try:
        # Load metadata
        with open('models/saved/churn_model_metadata.json', 'r') as f:
            metadata = json.load(f)
            feature_names = metadata.get('input_features', [])
            print(f"✅ Metadata loaded: {len(feature_names)} features")
        
        # Load scaler
        scaler = joblib.load('models/saved/scaler.pkl')
        print(f"✅ Scaler loaded")
        
        # Load encoders
        encoders = joblib.load('models/saved/label_encoders.pkl')
        print(f"✅ Label encoders loaded: {list(encoders.keys())}")
        
        # Load model
        input_dim = len(feature_names) if feature_names else 14
        model = TabTransformer(input_dim)
        model.load_state_dict(torch.load('models/saved/churn_model.pth', map_location='cpu'))
        model.eval()
        model_loaded = True
        print(f"✅ Model loaded successfully! (input dim: {input_dim})")
        
    except Exception as e:
        print(f"❌ Error loading: {e}")
        model_loaded = False
    
    print("="*50 + "\n")

@app.get("/")
def root():
    return {"message": "ACIE Churn API", "status": "running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "features": len(feature_names)
    }

@app.get("/features")
def get_features():
    return {
        "features": feature_names,
        "count": len(feature_names),
        "categorical": ["country", "acquisition_channel", "initial_plan"],
        "numerical": [f for f in feature_names if f not in ["country", "acquisition_channel", "initial_plan", "customer_id"]]
    }

@app.get("/metrics")
def get_metrics():
    return {
        "model": "TabTransformer",
        "auc_roc": 0.8155,
        "f1_score": 0.8378,
        "accuracy": 0.77
    }

class CustomerInput(BaseModel):
    age: float
    tenure_months: float
    avg_logins_3m: float
    avg_sessions_3m: float
    support_tickets: float
    engagement_score: float

@app.post("/predict")
def predict(customer: CustomerInput):
    if not model_loaded:
        return {"error": "Model not loaded", "churn_probability": 0.5, "risk_level": "UNKNOWN"}
    
    try:
        # Create feature vector (simplified - just use what we have)
        # In production, you'd need to match all 14 features
        input_data = np.array([[
            customer.age,
            customer.tenure_months,
            customer.avg_logins_3m,
            customer.avg_sessions_3m,
            customer.support_tickets,
            customer.engagement_score,
            0, 0, 0, 0, 0, 0, 0, 0  # placeholders
        ]])
        
        # Scale
        input_scaled = scaler.transform(input_data)
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_scaled)
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
        
        # Risk level
        if prob > 0.7:
            risk = "HIGH"
        elif prob > 0.3:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        
        return {
            "churn_probability": round(prob, 4),
            "risk_level": risk,
            "model_loaded": True
        }
    except Exception as e:
        return {"error": str(e), "churn_probability": 0.5, "risk_level": "ERROR"}

@app.get("/debug")
def debug():
    return {
        "model_loaded": model_loaded,
        "feature_count": len(feature_names),
        "features": feature_names[:5] if feature_names else []
    }
