# PRAHARI Backend — Track Data, Cyber Anomaly Detection & Risk Scoring API

This is the backend component for **PRAHARI**: real datasets, two trained ML models,
and a FastAPI server exposing everything to the Citizen App and Command Center.

## What's inside

```
prahari/
├── data/
│   ├── prep_track_data.py      # builds track_dataset.csv from AI4I 2020 dataset
│   ├── prep_cyber_data.py      # builds cyber_log_dataset.csv from NSL-KDD dataset
│   ├── track_dataset.csv       # 10,000 track segments with health/risk scores
│   └── cyber_log_dataset.csv   # 8,000 OT network log entries (normal + attacks)
├── models/
│   ├── train_track_model.py    # trains RandomForest defect predictor
│   ├── train_cyber_model.py    # trains Isolation Forest anomaly detector
│   ├── track_defect_model.pkl
│   └── cyber_anomaly_model.pkl
└── api/
    ├── main.py                  # FastAPI app
    └── requirements.txt
```

## Datasets — what's real and why

- **Track/sensor data**: derived from the **AI4I 2020 Predictive Maintenance Dataset**
  (UCI). Real sensor-pattern dataset (temperature, rotational speed, torque, tool wear,
  failure labels) widely used in industrial predictive maintenance research. Columns
  were renamed/re-contextualized to railway track segments (rail temperature, vibration,
  track stress, wear index) across realistic Indian railway routes (MAS-NZM, MAS-SBC, etc).

- **Cyber log data**: derived from the **NSL-KDD intrusion detection dataset** — a
  real, widely-used cybersecurity research dataset. Attack types were mapped to railway
  OT/signaling-network threat categories:
  - DoS attacks → "Network Flood / DoS on Signaling Network"
  - Probe attacks → "Reconnaissance / Network Scanning"
  - R2L attacks → "Unauthorized Remote Access Attempt"
  - U2R attacks → "Privilege Escalation / Insider Threat"
  - normal → "Normal Operation"

  Railway-specific cyber incident data isn't publicly available (security-sensitive),
  so this is the standard, defensible approach used in OT-security research.

## Models

1. **`cyber_anomaly_model.pkl`** — Isolation Forest, trained unsupervised on normal
   traffic. Learns what "normal" signaling-network behaviour looks like and flags
   deviations. Detects **85.4%** of real attack traffic (DoS, scanning, unauthorized
   access, insider threats) with a 10% false-positive rate on normal traffic.

2. **`track_defect_model.pkl`** — Random Forest classifier predicting probability of
   track defect from sensor readings (vibration, stress, wear, age, traffic load,
   inspection gap).

Both are real trained models — not rule-based "if/else" — using real-world datasets.

## Running the API

```bash
cd prahari/api
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000/docs** — interactive Swagger UI to test every
endpoint by clicking buttons (no code needed).

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/tracks` | GET | List track segments (filter by `route`, `risk_level`) |
| `/tracks/{segment_id}` | GET | One segment + ML-predicted defect probability |
| `/reports` | POST | Citizen submits a safety report |
| `/reports` | GET | List submitted reports |
| `/cyber-check` | POST | Run Isolation Forest on a network log entry → anomaly score |
| `/risk-score/{train_id}` | GET | Composite CCRS score (infra + cyber + reports) |
| `/docs` | GET | Swagger UI |

## How frontend apps integrate

Replace `localhost` with your deployed URL once hosted (Render/Railway free tier).

### Citizen App — submit a report
```javascript
fetch("http://<base-url>:8000/reports", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    train_no: "12627",
    issue_type: "Track Damage",
    location: "KM 142/3, Arakkonam",
    description: "Visible crack near track joint"
  })
})
  .then(res => res.json())
  .then(data => console.log(data));
```

### Citizen App — show Railway Safety Map
```javascript
fetch("http://<base-url>:8000/tracks?risk_level=Critical")
  .then(res => res.json())
  .then(segments => /* plot on map */ console.log(segments));
```

### Command Center — CCRS dashboard
```javascript
fetch("http://<base-url>:8000/risk-score/12627")
  .then(res => res.json())
  .then(data => {
    // data.ccrs, data.risk_level, data.breakdown.{infra_risk_score, cyber_risk_score, public_report_risk}
  });
```

### Command Center — Signal/Ops cyber check (e.g. simulated live feed)
```javascript
fetch("http://<base-url>:8000/cyber-check", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    asset_id: "SIG-014",
    protocol_type: "tcp", service: "http", flag: "SF",
    src_bytes: 200, dst_bytes: 300, count: 3, srv_count: 3,
    same_srv_rate: 1, dst_host_count: 10, dst_host_srv_count: 10,
    dst_host_same_srv_rate: 1
  })
})
  .then(res => res.json())
  .then(data => {
    // data.is_anomaly, data.cyber_risk_score
  });
```

## Deploying for the team (free)

1. Push this `prahari/` folder to your team's GitHub repo
2. Deploy `api/` on **Render.com** (free tier): connect repo, set build command
   `pip install -r requirements.txt`, start command
   `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Share the resulting URL (e.g. `https://prahari-api.onrender.com`) with both
   frontend teammates — they replace `localhost:8000` with this URL.

## Re-training models

If you want to regenerate the datasets or retrain models from scratch:
```bash
python3 data/prep_track_data.py
python3 data/prep_cyber_data.py
python3 models/train_track_model.py
python3 models/train_cyber_model.py
```
