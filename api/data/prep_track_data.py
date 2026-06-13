"""
Prepares railway track/sensor dataset from the AI4I 2020 Predictive Maintenance dataset.
Re-contextualizes generic machine sensor readings (temperature, rotational speed, torque,
tool wear) as railway track/component health sensors (rail temperature, vibration,
axle load stress, wear level) mapped onto Indian railway routes.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv("/home/claude/ai4i2020.csv", encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]

ROUTES = ["MAS-NZM", "MAS-SBC", "HYB-MAS", "TVC-LTT", "MAS-NZM-2", "ERS-MAS", "MAS-CBE"]
KM_RANGE = (0, 350)

n = len(df)

track_df = pd.DataFrame({
    "segment_id": [f"SEG-{i+1:05d}" for i in range(n)],
    "route": np.random.choice(ROUTES, n),
    "km_marker": np.round(np.random.uniform(*KM_RANGE, n), 1),
    # Rail temperature: derived from "Air temperature [K]" -> Celsius, realistic rail range
    "rail_temperature_c": np.round(df["Air temperature [K]"] - 273.15 + np.random.uniform(5, 25, n), 1),
    # Vibration level: derived from rotational speed, normalized to a 0-10 scale
    "vibration_level": np.round((df["Rotational speed [rpm]"] - df["Rotational speed [rpm]"].min())
                                 / (df["Rotational speed [rpm]"].max() - df["Rotational speed [rpm]"].min()) * 10, 2),
    # Axle/track stress: derived from Torque
    "track_stress_nm": df["Torque [Nm]"],
    # Wear level: directly reuse tool wear (minutes) as "track wear index"
    "wear_index_min": df["Tool wear [min]"],
    "track_age_years": np.random.randint(1, 40, n),
    "traffic_load_trains_per_day": np.random.randint(20, 200, n),
    "last_inspection_days_ago": np.random.randint(1, 365, n),
    # Failure label (re-used directly from AI4I -> track defect indicator)
    "defect_detected": df["Machine failure"],
})

# Composite health score (0-100, higher = worse condition / higher risk)
norm_vib = track_df["vibration_level"] / 10
norm_stress = (track_df["track_stress_nm"] - track_df["track_stress_nm"].min()) / \
               (track_df["track_stress_nm"].max() - track_df["track_stress_nm"].min())
norm_wear = (track_df["wear_index_min"] - track_df["wear_index_min"].min()) / \
             (track_df["wear_index_min"].max() - track_df["wear_index_min"].min())
norm_age = track_df["track_age_years"] / 40
norm_inspection = track_df["last_inspection_days_ago"] / 365

track_df["infra_risk_score"] = np.round(
    (norm_vib * 0.30 + norm_stress * 0.25 + norm_wear * 0.20 +
     norm_age * 0.15 + norm_inspection * 0.10) * 100, 1
)
# Boost score for actual defects
track_df.loc[track_df["defect_detected"] == 1, "infra_risk_score"] = \
    np.clip(track_df.loc[track_df["defect_detected"] == 1, "infra_risk_score"] + 25, 0, 100)

track_df["risk_level"] = pd.cut(
    track_df["infra_risk_score"],
    bins=[-1, 30, 55, 75, 101],
    labels=["Safe", "Warning", "High", "Critical"]
)

track_df.to_csv("/home/claude/prahari/data/track_dataset.csv", index=False)
print(f"Saved {len(track_df)} track segments")
print(track_df["risk_level"].value_counts())
print(track_df.head())
