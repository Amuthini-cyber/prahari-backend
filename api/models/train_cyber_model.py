"""
Trains an unsupervised Isolation Forest anomaly detector on railway OT network logs.
The model learns what 'normal' signaling-network traffic looks like and flags
deviations as anomalous (covers DoS, scanning, unauthorized access, insider threats
-- all represented in the underlying NSL-KDD-derived dataset).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("/home/claude/prahari/data/cyber_log_dataset.csv")

cat_cols = ["protocol_type", "service", "flag"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","count","srv_count","serror_rate","srv_serror_rate",
    "same_srv_rate","diff_srv_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_serror_rate"
]

X = df[feature_cols]

# Train mostly on normal traffic so the model learns "normal" behaviour
X_train = X[df["is_attack"] == 0]

model = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train)

# Evaluate: anomaly score on full set
scores = model.decision_function(X)          # higher = more normal
preds = model.predict(X)                      # -1 = anomaly, 1 = normal
df["anomaly_pred"] = (preds == -1).astype(int)

# Quick accuracy check vs known attack labels
detected = ((df["anomaly_pred"] == 1) & (df["is_attack"] == 1)).sum()
total_attacks = (df["is_attack"] == 1).sum()
print(f"Detected {detected}/{total_attacks} known attacks as anomalies "
      f"({detected/total_attacks*100:.1f}% recall)")

false_positives = ((df["anomaly_pred"] == 1) & (df["is_attack"] == 0)).sum()
print(f"False positives on normal traffic: {false_positives}/{(df['is_attack']==0).sum()}")

# Save model + encoders + feature order
joblib.dump({
    "model": model,
    "encoders": encoders,
    "feature_cols": feature_cols
}, "/home/claude/prahari/models/cyber_anomaly_model.pkl")

print("Saved cyber_anomaly_model.pkl")
