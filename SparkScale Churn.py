# ============================================================
# SparkScale Churn - Full End-to-End Python Project + Visualization
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 0. Setup directories
# ============================================================

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_dir, "data")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print(f"Data will be saved in: {data_dir}")

# ============================================================
# 1. Parameters
# ============================================================

num_rows = 100000
num_users = 5000
regions = ["North", "South", "West"]
plans = ["Prepaid", "Postpaid"]

# ============================================================
# 2. Generate Raw Telecom Logs
# ============================================================

print("\n[STEP 1] Generating Raw Telecom Logs...")

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

churn_prob = np.where(
    (raw_data["complaint_count"] > 3) | (raw_data["tenure_months"] < 6),
    0.8,
    0.08
)

raw_data["churn"] = (np.random.rand(num_rows) < churn_prob).astype(int)

raw_file = os.path.join(data_dir, "raw_telecom_logs.csv")
raw_data.to_csv(raw_file, index=False)
print(f"Raw Telecom Dataset Saved → {raw_file}")

# ============================================================
# 3. Aggregated User Features
# ============================================================

print("\n[STEP 2] Generating Aggregated User Features...")

agg_data = raw_data.groupby("user_id").agg(
    avg_data_usage=("data_usage", "mean"),
    avg_call_duration=("call_duration", "mean"),
    total_billing=("billing_amount", "sum"),
    total_complaints=("complaint_count", "sum"),
    avg_tenure=("tenure_months", "mean"),
    label=("churn", "mean")
).reset_index()

agg_file = os.path.join(data_dir, "aggregated_user_features.csv")
agg_data.to_csv(agg_file, index=False)
print(f"Aggregated User Features Saved → {agg_file}")

# ============================================================
# 4. ML Training
# ============================================================

print("\n[STEP 3] Training Model...")

X = agg_data.drop(columns=["user_id", "label"])
y = agg_data["label"].apply(lambda x: 1 if x > 0 else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Evaluation on Test Data:")
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# ============================================================
# 5. Visualization Section
# ============================================================

print("\n[STEP 4] Generating Visualizations...")

# 1️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_path = os.path.join(data_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print("Confusion Matrix Saved")

# 2️⃣ Feature Importance
importances = rf_model.feature_importances_
feature_names = X.columns

feat_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

plt.figure()
plt.bar(feat_df["feature"], feat_df["importance"])
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

fi_path = os.path.join(data_dir, "feature_importance.png")
plt.savefig(fi_path)
plt.close()
print("Feature Importance Chart Saved")

# 3️⃣ Churn Distribution Pie Chart
plt.figure()
y.value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Churn Distribution")

pie_path = os.path.join(data_dir, "churn_distribution.png")
plt.savefig(pie_path)
plt.close()
print("Churn Distribution Pie Chart Saved")

# ============================================================
# 6. Predict New Month Data
# ============================================================

print("\n[STEP 5] Predicting New Month Data...")

new_month_data = raw_data.sample(frac=0.2, random_state=42)

new_agg = new_month_data.groupby("user_id").agg(
    avg_data_usage=("data_usage", "mean"),
    avg_call_duration=("call_duration", "mean"),
    total_billing=("billing_amount", "sum"),
    total_complaints=("complaint_count", "sum"),
    avg_tenure=("tenure_months", "mean")
).reset_index()

X_new = new_agg.drop(columns=["user_id"])
pred_probs = rf_model.predict_proba(X_new)[:, 1]

new_agg["churn_probability"] = pred_probs

pred_file = os.path.join(data_dir, "new_month_predictions.csv")
new_agg.to_csv(pred_file, index=False)
print("New Month Predictions Saved")

# 4️⃣ Top 10 High-Risk Users
top_users = new_agg.sort_values("churn_probability", ascending=False).head(10)

plt.figure()
plt.bar(top_users["user_id"].astype(str),
        top_users["churn_probability"])
plt.xticks(rotation=45)
plt.title("Top 10 High Risk Users")
plt.xlabel("User ID")
plt.ylabel("Churn Probability")

top_path = os.path.join(data_dir, "top_10_high_risk_users.png")
plt.savefig(top_path)
plt.close()
print("Top 10 High-Risk Users Chart Saved")

# ============================================================
# 7. Project Complete
# ============================================================

print("\n==================================================")
print("SparkScale Churn Project Completed Successfully!")
print("All datasets, predictions, and visualizations are in:", data_dir)
print("==================================================")
