import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Generate 10000 customers
print("Generating 10,000 subscription customers...")
customers = []
for i in range(1, 10001):
    customers.append({
        'customer_id': f'C{i:06d}',
        'age': random.randint(18, 65),
        'plan': random.choice(['Basic', 'Standard', 'Premium']),
        'tenure_months': random.randint(1, 36),
        'engagement': random.uniform(0.1, 0.95),
        'churned': random.choice([0, 1])
    })

pd.DataFrame(customers).to_csv('data/raw/customers.csv', index=False)
print("✅ Data generated!")
