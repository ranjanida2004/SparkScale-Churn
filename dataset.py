# ============================================================
# SparkScale Churn - Dataset Generation Script (Python Only)
# ============================================================

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# 0. Setup directories
# ============================================================

# Get absolute path of this script
project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_dir, "data")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# ============================================================
# 1. Parameters
# ============================================================

num_rows = 100000  # total logs (adjustable)
num_users = 5000   # approximate unique users
regions = ["North", "South", "West"]
plans = ["Prepaid", "Postpaid"]

# ============================================================
# 2. Generate Raw Telecom Logs
# ============================================================

print("Generating Raw Telecom Dataset...")

np.random.seed(42)

raw_data = pd.DataFrame({
    "row_id": np.arange(num_rows),
    "user_id": np.random.randint(0, num_users, size=num_rows),
    "call_duration": np.round(np.random.uniform(0, 60, size=num_rows), 2),
    "data_usage": np.round(np.random.uniform(0, 5000, size=num_rows), 2),
    "billing_amount": np.round(np.random.uniform(0, 1000, size=num_rows), 2),
    "complaint_count": np.random.randint(0, 6, size=num_rows),
    "tenure_months": np.random.randint(0, 61, size=num_rows),
    "plan_type": np.random.choice(plans, size=num_rows),
    "region": np.random.choice(regions, size=num_rows),
    "activity_date": [datetime.today() - timedelta(days=int(x)) for x in np.random.randint(0, 365, size=num_rows)]
})

# Add churn logic: high complaints or low tenure → higher chance
churn_prob = np.where(
    (raw_data["complaint_count"] > 3) | (raw_data["tenure_months"] < 6),
    0.8,
    0.08
)

raw_data["churn"] = np.random.rand(num_rows) < churn_prob
raw_data["churn"] = raw_data["churn"].astype(int)

# Save Raw Dataset
raw_file = os.path.join(data_dir, "raw_telecom_logs.csv")
raw_data.to_csv(raw_file, index=False)
print(f"Raw Telecom Dataset Saved Successfully → {raw_file}")

# ============================================================
# 3. Aggregated User Features
# ============================================================

print("Generating Aggregated User Features...")

agg_data = raw_data.groupby("user_id").agg(
    avg_data_usage=("data_usage", "mean"),
    avg_call_duration=("call_duration", "mean"),
    total_billing=("billing_amount", "sum"),
    total_complaints=("complaint_count", "sum"),
    avg_tenure=("tenure_months", "mean"),
    label=("churn", "mean")  # average churn as label
).reset_index()

agg_file = os.path.join(data_dir, "aggregated_user_features.csv")
agg_data.to_csv(agg_file, index=False)
print(f"Aggregated Dataset Saved Successfully → {agg_file}")

# ============================================================
# 4. New Month Data (Batch Prediction)
# ============================================================

print("Generating New Month Dataset...")

new_month_data = raw_data.sample(frac=0.2, random_state=42)
new_month_file = os.path.join(data_dir, "new_month_data.csv")
new_month_data.to_csv(new_month_file, index=False)
print(f"New Month Dataset Saved Successfully → {new_month_file}")

# ============================================================
# 5. Summary
# ============================================================

print("--------------------------------------------------")
print("All Three Datasets Generated Successfully:")
print(f"1. Raw Telecom Logs → {raw_file}")
print(f"2. Aggregated User Features → {agg_file}")
print(f"3. New Month Batch Dataset → {new_month_file}")
print("--------------------------------------------------")
