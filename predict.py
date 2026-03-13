import torch
import pandas as pd
import joblib
from churn_model_production import TabTransformer

# Load model and preprocessing objects
model = TabTransformer(input_dim=14)  # Adjust based on your features
model.load_state_dict(torch.load('models/saved/churn_model.pth'))
model.eval()

scaler = joblib.load('models/saved/scaler.pkl')
label_encoders = joblib.load('models/saved/label_encoders.pkl')

# Example prediction
sample_customer = pd.DataFrame({
    'age': [35],
    'cac': [45.50],
    'tenure_months': [12],
    'avg_logins_3m': [25],
    'avg_sessions_3m': [30],
    # Add all your features here
})

# Preprocess and predict
# ... (add full preprocessing code)

print("✅ Model ready for predictions!")
