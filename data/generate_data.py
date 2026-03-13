"""
ACIE System - Synthetic Subscription Customer Data Generator
Generates realistic customer behavior data for churn prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json

np.random.seed(42)
random.seed(42)

print("="*80)
print("ACIE SYSTEM - GENERATING SUBSCRIPTION CUSTOMER DATA")
print("="*80)

# Create directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Configuration
N_CUSTOMERS = 10000
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)
MONTHS_TOTAL = 36

#============================================================================
# 1. CUSTOMER ACQUISITION
#============================================================================

print("\n📊 Generating customer profiles...")

channels = ['Organic', 'Paid Search', 'Social Media', 'Referral', 'Email', 'Affiliate']
plans = ['Basic', 'Standard', 'Premium']
plan_prices = {'Basic': 9.99, 'Standard': 19.99, 'Premium': 39.99}

customers = []
for i in range(1, N_CUSTOMERS + 1):
    # Signup date
    signup_date = START_DATE + timedelta(days=random.randint(0, 700))
    
    # Acquisition
    channel = random.choice(channels)
    if channel == 'Paid Search':
        cac = random.uniform(50, 120)
    elif channel == 'Social Media':
        cac = random.uniform(30, 80)
    elif channel == 'Organic':
        cac = random.uniform(10, 30)
    else:
        cac = random.uniform(20, 60)
    
    # Plan selection
    plan = random.choices(plans, weights=[50, 35, 15])[0]
    
    # Demographics
    age = random.randint(18, 65)
    country = random.choice(['US', 'UK', 'Canada', 'Germany', 'France', 'Australia'])
    
    customers.append({
        'customer_id': f'C{i:06d}',
        'signup_date': signup_date,
        'acquisition_channel': channel,
        'cac': round(cac, 2),
        'initial_plan': plan,
        'age': age,
        'country': country
    })

customers_df = pd.DataFrame(customers)
print(f"✅ Created {len(customers_df)} customer profiles")

#============================================================================
# 2. SUBSCRIPTION EVENTS & BEHAVIOR
#============================================================================

print("\n📅 Generating subscription events...")

subscription_events = []
behavioral_features = []
event_id = 1

for idx, customer in customers_df.iterrows():
    customer_id = customer['customer_id']
    signup_date = customer['signup_date']
    current_plan = customer['initial_plan']
    
    # Customer lifecycle behavior patterns
    behavior_type = random.choices(
        ['champion', 'regular', 'early_churn', 'late_churn'],
        weights=[20, 30, 30, 20]
    )[0]
    
    if behavior_type == 'champion':
        active_months = random.randint(24, 36)
        base_engagement = random.uniform(0.7, 0.95)
        churn_prob_base = 0.02
    elif behavior_type == 'regular':
        active_months = random.randint(12, 24)
        base_engagement = random.uniform(0.4, 0.7)
        churn_prob_base = 0.08
    elif behavior_type == 'early_churn':
        active_months = random.randint(1, 6)
        base_engagement = random.uniform(0.1, 0.4)
        churn_prob_base = 0.35
    else:  # late_churn
        active_months = random.randint(6, 12)
        base_engagement = random.uniform(0.3, 0.6)
        churn_prob_base = 0.18
    
    current_date = signup_date
    is_active = True
    month_counter = 0
    
    while month_counter < min(active_months, MONTHS_TOTAL) and is_active:
        month_counter += 1
        
        # Payment event
        payment_amount = plan_prices[current_plan]
        payment_success = random.random() > 0.02
        
        # Engagement metrics
        engagement_score = base_engagement * random.uniform(0.8, 1.2)
        engagement_score = max(0, min(1, engagement_score))
        
        logins_per_month = int(engagement_score * random.randint(15, 45))
        sessions_per_month = int(engagement_score * random.randint(20, 60))
        avg_session_duration = engagement_score * random.uniform(10, 45)
        features_used = int(engagement_score * random.randint(3, 15))
        support_tickets = 1 if random.random() < 0.1 else 0
        
        # Record subscription event
        subscription_events.append({
            'event_id': event_id,
            'customer_id': customer_id,
            'event_date': current_date,
            'event_type': 'payment',
            'plan': current_plan,
            'amount': payment_amount,
            'payment_success': payment_success,
            'month_number': month_counter
        })
        event_id += 1
        
        # Record behavioral features
        behavioral_features.append({
            'customer_id': customer_id,
            'month': current_date.strftime('%Y-%m'),
            'month_number': month_counter,
            'engagement_score': round(engagement_score, 3),
            'logins': logins_per_month,
            'sessions': sessions_per_month,
            'avg_session_duration': round(avg_session_duration, 2),
            'features_used': features_used,
            'support_tickets': support_tickets,
            'plan': current_plan,
            'payment_success': payment_success
        })
        
        # Plan changes
        if random.random() < 0.05:
            if current_plan == 'Basic' and random.random() < 0.7:
                current_plan = 'Standard'
            elif current_plan == 'Premium' and random.random() < 0.3:
                current_plan = 'Standard'
        
        # Churn decision
        churn_this_month = random.random() < churn_prob_base
        if engagement_score < 0.3 or not payment_success:
            churn_this_month = random.random() < 0.4
        
        if churn_this_month:
            subscription_events.append({
                'event_id': event_id,
                'customer_id': customer_id,
                'event_date': current_date,
                'event_type': 'churn',
                'plan': current_plan,
                'amount': 0,
                'payment_success': False,
                'month_number': month_counter
            })
            event_id += 1
            is_active = False
        
        current_date = current_date + timedelta(days=30)
    
    if idx % 1000 == 0 and idx > 0:
        print(f"   Processed {idx}/{len(customers_df)} customers...")

subscription_df = pd.DataFrame(subscription_events)
behavior_df = pd.DataFrame(behavioral_features)

print(f"✅ Generated {len(subscription_df)} subscription events")
print(f"✅ Generated {len(behavior_df)} monthly behavior records")

#============================================================================
# 3. CALCULATE GROUND TRUTH LABELS
#============================================================================

print("\n🎯 Calculating labels...")

churned_customers = subscription_df[subscription_df['event_type'] == 'churn']['customer_id'].unique()
customers_df['churned'] = customers_df['customer_id'].isin(churned_customers).astype(int)

ltv_data = []
for customer_id in customers_df['customer_id']:
    total_revenue = subscription_df[
        (subscription_df['customer_id'] == customer_id) & 
        (subscription_df['event_type'] == 'payment') &
        (subscription_df['payment_success'] == True)
    ]['amount'].sum()
    
    ltv_data.append({
        'customer_id': customer_id,
        'ltv': round(total_revenue, 2)
    })

ltv_df = pd.DataFrame(ltv_data)
customers_df = customers_df.merge(ltv_df, on='customer_id')

print(f"   Churn rate: {customers_df['churned'].mean()*100:.1f}%")
print(f"   Average LTV: ${customers_df['ltv'].mean():.2f}")

#============================================================================
# 4. SAVE DATA
#============================================================================

print("\n💾 Saving datasets...")

customers_df.to_csv('data/raw/customers.csv', index=False)
subscription_df.to_csv('data/raw/subscriptions.csv', index=False)
behavior_df.to_csv('data/raw/behavior.csv', index=False)

print("✅ Saved to data/raw/")

#============================================================================
# 5. CREATE FEATURE MATRIX FOR ML
#============================================================================

print("\n🔧 Creating feature matrix for ML...")

feature_matrix = []

for customer_id in customers_df['customer_id']:
    cust_info = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
    cust_behavior = behavior_df[behavior_df['customer_id'] == customer_id]
    
    if len(cust_behavior) > 0:
        features = {
            'customer_id': customer_id,
            'age': cust_info['age'],
            'country': cust_info['country'],
            'acquisition_channel': cust_info['acquisition_channel'],
            'cac': cust_info['cac'],
            'initial_plan': cust_info['initial_plan'],
            'tenure_months': len(cust_behavior),
            'avg_logins_3m': cust_behavior.tail(3)['logins'].mean(),
            'avg_sessions_3m': cust_behavior.tail(3)['sessions'].mean(),
            'avg_session_duration_3m': cust_behavior.tail(3)['avg_session_duration'].mean(),
            'avg_features_used_3m': cust_behavior.tail(3)['features_used'].mean(),
            'engagement_score_latest': cust_behavior.iloc[-1]['engagement_score'],
            'payment_failures': (cust_behavior['payment_success'] == False).sum(),
            'payment_success_rate': cust_behavior['payment_success'].mean(),
            'total_support_tickets': cust_behavior['support_tickets'].sum(),
            'churned': cust_info['churned'],
            'ltv': cust_info['ltv']
        }
        feature_matrix.append(features)

features_df = pd.DataFrame(feature_matrix)
features_df = features_df.fillna(0)
features_df.to_csv('data/processed/feature_matrix.csv', index=False)

print(f"✅ Created feature matrix with {len(features_df)} samples and {len(features_df.columns)} features")
print("\n" + "="*80)
print("DATA GENERATION COMPLETE!")
print("="*80)
