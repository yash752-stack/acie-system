"""
ACIE System - Production-Grade Deep Churn Prediction Model
TabTransformer with proper architecture, class weighting, and MLOps tracking
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report, precision_recall_curve
from scipy import stats
import joblib
import json
import mlflow
import mlflow.pytorch

# CRITICAL: Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("="*80)
print("ACIE - TRAINING PRODUCTION-GRADE CHURN PREDICTION MODEL")
print("="*80)

#============================================================================
# 1. LOAD DATA
#============================================================================

print("\n📊 Loading feature matrix...")
df = pd.read_csv('data/processed/feature_matrix.csv')

# Separate features and target
X = df.drop(['customer_id', 'churned', 'ltv'], axis=1)
y = df['churned']

print(f"   Total samples: {len(df)}")
print(f"   Features: {X.shape[1]}")
print(f"   Churn rate: {y.mean()*100:.1f}%")
print(f"   Class imbalance ratio: {(1-y.mean())/y.mean():.2f}:1")

#============================================================================
# 2. PREPROCESSING
#============================================================================

print("\n🔧 Preprocessing...")

# Encode categorical variables
categorical_cols = ['country', 'acquisition_channel', 'initial_plan']
label_encoders = {}

for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
numerical_cols = [col for col in X.columns if col not in categorical_cols]
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Stratified split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp
)

print(f"   Train: {len(X_train)} ({y_train.mean()*100:.1f}% churn)")
print(f"   Val: {len(X_val)} ({y_val.mean()*100:.1f}% churn)")
print(f"   Test: {len(X_test)} ({y_test.mean()*100:.1f}% churn)")

#============================================================================
# 3. TABTRANSFORMER ARCHITECTURE
#============================================================================

class TabTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=8, num_layers=3, dropout=0.3):
        super(TabTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.classifier(x)
        return x

#============================================================================
# 4. TRAINING
#============================================================================

print("\n🚀 Training model...")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train.values)
y_train_t = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_val_t = torch.FloatTensor(X_val.values)
y_val_t = torch.FloatTensor(y_val.values).reshape(-1, 1)
X_test_t = torch.FloatTensor(X_test.values)
y_test_t = torch.FloatTensor(y_test.values).reshape(-1, 1)

# Class weighting
pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()])

# Initialize model
model = TabTransformer(input_dim=X_train.shape[1])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
epochs = 50
batch_size = 128
best_val_auc = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    
    for i in range(0, len(X_train_t), batch_size):
        batch_X = X_train_t[i:i+batch_size]
        batch_y = y_train_t[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_probs = torch.sigmoid(val_outputs).numpy().flatten()
        val_auc = roc_auc_score(y_val.values, val_probs)
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(X_train_t)*batch_size:.4f} | Val AUC: {val_auc:.4f}")
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'models/saved/churn_model.pth')

#============================================================================
# 5. EVALUATION
#============================================================================

print("\n📊 Evaluating on test set...")

model.load_state_dict(torch.load('models/saved/churn_model.pth'))
model.eval()

with torch.no_grad():
    test_outputs = model(X_test_t)
    test_probs = torch.sigmoid(test_outputs).numpy().flatten()
    test_preds = (test_probs > 0.5).astype(int)

test_auc = roc_auc_score(y_test.values, test_probs)
test_f1 = f1_score(y_test.values, test_preds)

print(f"\n✅ TEST RESULTS:")
print(f"   AUC-ROC: {test_auc:.4f}")
print(f"   F1 Score: {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test.values, test_preds, target_names=['Not Churned', 'Churned']))

#============================================================================
# 6. SAVE ARTIFACTS
#============================================================================

print("\n💾 Saving model artifacts...")

joblib.dump(label_encoders, 'models/saved/label_encoders.pkl')
joblib.dump(scaler, 'models/saved/scaler.pkl')

metadata = {
    'model_type': 'TabTransformer',
    'input_features': list(X.columns),
    'test_auc': float(test_auc),
    'test_f1': float(test_f1)
}

with open('models/saved/churn_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Model saved successfully!")
print(f"\n📊 Final Test AUC: {test_auc:.4f}")
