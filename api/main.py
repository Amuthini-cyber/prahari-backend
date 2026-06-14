"""
PRAHARI Backend API
====================
Serves track health data, accepts citizen reports, runs cyber anomaly detection,
and computes the Convergence Cyber Risk Score (CCRS) per train/route.

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Docs (Swagger UI):
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import os

BASE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="PRAHARI API",
    description="AI-Powered Convergence Risk Intelligence for Railways",
    version="1.0.0"
)

# Allow frontend apps (Citizen App / Command Center) to call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load data + models at startup
# ---------------------------------------------------------------------------
track_df = pd.read_csv(os.path.join(BASE, "data", "track_dataset.csv"))
cyber_bundle = joblib.load(os.path.join(BASE, "models", "cyber_anomaly_model.pkl"))
track_bundle = joblib.load(os.path.join(BASE, "models", "track_defect_model.pkl"))

cyber_model = cyber_bundle["model"]
cyber_encoders = cyber_bundle["encoders"]
cyber_features = cyber_bundle["feature_cols"]

track_model = track_bundle["model"]
track_features = track_bundle["feature_cols"]

# In-memory store for citizen reports (use a real DB in production)
reports_db: List[dict] = []
_report_id_counter = 1


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class CitizenReport(BaseModel):
    train_no: Optional[str] = Field(None, example="12627")
    issue_type: str = Field(..., example="Track Damage")
    location: str = Field(..., example="KM 142/3, Arakkonam")
    description: Optional[str] = Field(None, example="Visible crack near track joint")
    reporter_name: Optional[str] = Field(None, example="Anonymous")


class CyberLogEntry(BaseModel):
    asset_id: str = Field(..., example="SIG-014")
    duration: float = 0
    protocol_type: str = Field(..., example="tcp")
    service: str = Field(..., example="http")
    flag: str = Field(..., example="SF")
    src_bytes: float = 0
    dst_bytes: float = 0
    wrong_fragment: float = 0
    urgent: float = 0
    hot: float = 0
    num_failed_logins: float = 0
    logged_in: float = 1
    num_compromised: float = 0
    count: float = 1
    srv_count: float = 1
    serror_rate: float = 0
    srv_serror_rate: float = 0
    same_srv_rate: float = 1
    diff_srv_rate: float = 0
    dst_host_count: float = 1
    dst_host_srv_count: float = 1
    dst_host_same_srv_rate: float = 1
    dst_host_serror_rate: float = 0


# ---------------------------------------------------------------------------
# Helper: encode + score a single cyber log entry
# ---------------------------------------------------------------------------
def score_cyber_log(entry: CyberLogEntry):
    row = entry.dict()
    row.pop("asset_id")
    df_row = pd.DataFrame([row])

    for col, le in cyber_encoders.items():
        val = df_row.at[0, col]
        if val in le.classes_:
            df_row[col] = le.transform([val])[0]
        else:
            df_row[col] = -1  # unseen category -> treated as anomalous signal

    df_row = df_row[cyber_features]

    raw_score = cyber_model.decision_function(df_row)[0]   # higher = more normal
    is_anomaly = cyber_model.predict(df_row)[0] == -1

    # Convert to a 0-100 risk score (lower decision_function -> higher risk)
    risk_score = float(np.clip((0.5 - raw_score) * 100, 0, 100))

    return {
        "asset_id": entry.asset_id,
        "is_anomaly": bool(is_anomaly),
        "cyber_risk_score": round(risk_score, 1),
        "raw_decision_score": round(float(raw_score), 4),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "PRAHARI API",
        "status": "online",
        "endpoints": ["/tracks", "/tracks/{segment_id}", "/reports",
                      "/cyber-check", "/risk-score/{train_id}", "/docs"]
    }


@app.get("/tracks")
def get_tracks(route: Optional[str] = None, risk_level: Optional[str] = None, limit: int = 100):
    df = track_df
    if route:
        df = df[df["route"] == route]
    if risk_level:
        df = df[df["risk_level"] == risk_level]
    return df.head(limit).to_dict(orient="records")


@app.get("/tracks/{segment_id}")
def get_track_segment(segment_id: str):
    row = track_df[track_df["segment_id"] == segment_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Segment not found")

    record = row.iloc[0].to_dict()

    # Run predictive maintenance model on this segment's sensor data
    X = row[track_features]
    defect_prob = track_model.predict_proba(X)[0][1]
    record["predicted_defect_probability"] = round(float(defect_prob), 3)

    return record


@app.post("/reports")
def create_report(report: CitizenReport):
    global _report_id_counter
    record = report.dict()
    record["report_id"] = f"RPT-{_report_id_counter:05d}"
    record["timestamp"] = datetime.utcnow().isoformat()
    record["status"] = "Pending Verification"
    _report_id_counter += 1
    reports_db.append(record)
    return record


@app.get("/reports")
def list_reports(status: Optional[str] = None, limit: int = 50):
    data = reports_db
    if status:
        data = [r for r in data if r["status"] == status]
    return data[-limit:][::-1]  # most recent first


@app.post("/cyber-check")
def cyber_check(entry: CyberLogEntry):
    """Run the trained Isolation Forest anomaly model on a single OT network log entry."""
    return score_cyber_log(entry)


@app.get("/risk-score/{train_id}")
def get_risk_score(train_id: str):
    """
    Composite CCRS-style score combining:
      - Infra risk (avg of track segments on the train's route)
      - Cyber risk (sample signaling asset check)
      - Public reports (count of unresolved reports for this train)
    """
    return compute_risk_score(train_id)


def compute_risk_score(train_id: str):
    """Internal helper: computes CCRS for a given train_id. Reused by
    /risk-score/{train_id}, /trains, and /dashboard-summary."""
    # crude mapping: use train_id's first 5 chars to pick a "route" deterministically
    routes = track_df["route"].unique()
    route = routes[hash(train_id) % len(routes)]

    route_segments = track_df[track_df["route"] == route]
    infra_risk = float(route_segments["infra_risk_score"].mean())

    # sample a cyber log entry for this "asset" and score it
    sample_log = CyberLogEntry(
        asset_id=f"SIG-{(hash(train_id) % 50) + 1:03d}",
        protocol_type="tcp", service="http", flag="SF",
        duration=0, src_bytes=200, dst_bytes=200, count=5, srv_count=5,
        same_srv_rate=1, dst_host_count=10, dst_host_srv_count=10,
        dst_host_same_srv_rate=1
    )
    cyber_result = score_cyber_log(sample_log)

    train_reports = [r for r in reports_db if r.get("train_no") == train_id
                      and r["status"] != "Resolved"]
    report_risk = min(len(train_reports) * 15, 100)

    ccrs = round(
        infra_risk * 0.45 +
        cyber_result["cyber_risk_score"] * 0.35 +
        report_risk * 0.20,
        1
    )

    if ccrs >= 75:
        level = "Critical"
    elif ccrs >= 55:
        level = "High"
    elif ccrs >= 30:
        level = "Warning"
    else:
        level = "Safe"

    return {
        "train_id": train_id,
        "route": route,
        "ccrs": ccrs,
        "risk_level": level,
        "breakdown": {
            "infra_risk_score": round(infra_risk, 1),
            "cyber_risk_score": cyber_result["cyber_risk_score"],
            "public_report_risk": report_risk,
        },
        "active_reports": len(train_reports)
    }


# ---------------------------------------------------------------------------
# Command Center support endpoints
# ---------------------------------------------------------------------------

# Sample fleet of train numbers for demo purposes (matches PRAHARI mockup style)
DEMO_TRAIN_IDS = ["12627", "12651", "12711", "16317", "12621", "12609", "16723", "12693"]


@app.get("/trains")
def list_trains(limit: int = 20):
    """
    Live Train Risk Status table — returns CCRS + risk level for a fleet of
    trains, for the Command Center dashboard table/heatmap.
    """
    results = [compute_risk_score(tid) for tid in DEMO_TRAIN_IDS[:limit]]
    # sort by CCRS descending so highest-risk trains appear first
    results.sort(key=lambda r: r["ccrs"], reverse=True)
    return results


@app.get("/dashboard-summary")
def dashboard_summary():
    """
    Aggregate stats for the Command Center top bar:
    Active Trains, High Risk Trains, Critical Alerts, Open Reports.
    """
    train_scores = [compute_risk_score(tid) for tid in DEMO_TRAIN_IDS]

    active_trains = len(train_scores)
    high_risk_trains = sum(1 for t in train_scores if t["risk_level"] in ("High", "Critical"))
    critical_alerts = sum(1 for t in train_scores if t["risk_level"] == "Critical")
    open_reports = sum(1 for r in reports_db if r["status"] != "Resolved")

    # Critical track segments (for heatmap context)
    critical_segments = int((track_df["risk_level"] == "Critical").sum())

    return {
        "active_trains": active_trains,
        "high_risk_trains": high_risk_trains,
        "critical_alerts": critical_alerts,
        "open_reports": open_reports,
        "critical_track_segments": critical_segments,
        "trains": train_scores
    }




class SimulateRequest(BaseModel):
    """
    Inputs for the live Convergence Engine demo.
    Each factor is 0-100 (higher = riskier). Defaults provided so the
    endpoint works even if only some sliders are moved.
    """
    infra_risk_score: float = Field(30.0, ge=0, le=100, example=40.0,
                                     description="Track/infrastructure health risk (0-100)")
    cyber_risk_score: float = Field(20.0, ge=0, le=100, example=25.0,
                                     description="Cyber/OT anomaly risk (0-100)")
    signal_delay_score: float = Field(10.0, ge=0, le=100, example=15.0,
                                       description="Signal update delay risk (0-100, e.g. 6s delay vs 10s threshold -> ~60)")
    crew_fatigue_index: float = Field(20.0, ge=0, le=100, example=70.0,
                                       description="Crew alertness/fatigue risk score (0-100), from fatigue model")


@app.post("/simulate")
def simulate_ccrs(req: SimulateRequest):
    """
    PRAHARI Convergence Engine — live what-if simulation.

    Combines four risk signals (infrastructure, cyber, signal delay, crew fatigue)
    into a single Composite Convergence Risk Score (CCRS) with a full breakdown,
    matching the worked example in the PRAHARI proposal (Section 4).

    Designed to power an interactive slider dashboard: send current slider
    values, get back the live CCRS + recommended action.
    """
    weights = {
        "crew_fatigue_index": 0.35,
        "signal_delay_score": 0.30,
        "infra_risk_score": 0.20,
        "cyber_risk_score": 0.15,
    }

    values = {
        "crew_fatigue_index": req.crew_fatigue_index,
        "signal_delay_score": req.signal_delay_score,
        "infra_risk_score": req.infra_risk_score,
        "cyber_risk_score": req.cyber_risk_score,
    }

    ccrs = round(sum(values[k] * weights[k] for k in weights), 1)

    if ccrs >= 75:
        level = "Critical"
        action = ("Impose advisory speed restriction (40 km/h). "
                  "Alert relief crew point at next station. "
                  "Flag section for priority inspection.")
    elif ccrs >= 55:
        level = "High"
        action = ("Increase monitoring frequency. "
                  "Notify loco pilot and station master. "
                  "Schedule inspection within 24 hours.")
    elif ccrs >= 30:
        level = "Warning"
        action = "Continue normal operations with routine monitoring."
    else:
        level = "Safe"
        action = "No action required."

    breakdown_pct = {
        k: round((values[k] * weights[k] / ccrs * 100) if ccrs > 0 else 0, 1)
        for k in weights
    }

    return {
        "ccrs": ccrs,
        "risk_level": level,
        "recommended_action": action,
        "inputs": values,
        "weights": weights,
        "contribution_percent": breakdown_pct,
    }
