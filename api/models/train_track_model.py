"""
Trains a supervised classifier to predict track defect probability from
sensor readings (vibration, stress, wear, age, inspection gap), using the
AI4I-derived track dataset. This is the "predictive maintenance" model.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("/home/claude/prahari/data/track_dataset.csv")

feature_cols = [
    "vibration_level", "track_stress_nm", "wear_index_min",
    "track_age_years", "traffic_load_trains_per_day", "last_inspection_days_ago"
]

X = df[feature_cols]
y = df["defect_detected"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["No Defect", "Defect"]))

joblib.dump({
    "model": model,
    "feature_cols": feature_cols
}, "/home/claude/prahari/models/track_defect_model.pkl")

print("Saved track_defect_model.pkl")
