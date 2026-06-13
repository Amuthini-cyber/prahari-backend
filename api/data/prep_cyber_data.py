"""
Prepares a railway OT/network cyber-log dataset from the NSL-KDD intrusion detection dataset.
NSL-KDD attack categories map onto railway signaling/control-network threat types:

- DoS attacks (neptune, smurf, back, etc.)      -> "Network Flood / DoS on signaling network"
- Probe attacks (satan, ipsweep, portsweep...)  -> "Reconnaissance / Network Scanning"
- R2L attacks (guess_passwd, ftp_write...)      -> "Unauthorized Remote Access Attempt"
- U2R attacks (buffer_overflow, rootkit...)     -> "Privilege Escalation / Insider Threat"
- normal                                        -> "Normal Operation"
"""
import pandas as pd
import numpy as np

np.random.seed(42)

COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

df = pd.read_csv("/home/claude/kdd_train.txt", names=COLS)

DOS = {"neptune","smurf","back","teardrop","pod","land","apache2","udpstorm","processtable","worm","mailbomb"}
PROBE = {"satan","ipsweep","nmap","portsweep","mscan","saint"}
R2L = {"guess_passwd","ftp_write","imap","phf","multihop","warezmaster","warezclient",
       "spy","xlock","xsnoop","snmpguess","snmpgetattack","httptunnel","sendmail","named"}
U2R = {"buffer_overflow","loadmodule","rootkit","perl","sqlattack","xterm","ps"}

def map_category(label):
    if label == "normal":
        return "Normal Operation"
    if label in DOS:
        return "Network Flood / DoS on Signaling Network"
    if label in PROBE:
        return "Reconnaissance / Network Scanning"
    if label in R2L:
        return "Unauthorized Remote Access Attempt"
    if label in U2R:
        return "Privilege Escalation / Insider Threat"
    return "Unknown Anomaly"

df["threat_category"] = df["label"].apply(map_category)
df["is_attack"] = (df["label"] != "normal").astype(int)

# Sample down to a manageable size, keeping class balance reasonable
normal = df[df["is_attack"] == 0].sample(n=4000, random_state=42)
attacks = df[df["is_attack"] == 1].sample(n=4000, random_state=42)
log_df = pd.concat([normal, attacks]).sample(frac=1, random_state=42).reset_index(drop=True)

# Add railway-context fields
ASSET_IDS = [f"SIG-{i:03d}" for i in range(1, 51)]  # signaling boxes / control nodes
log_df["asset_id"] = np.random.choice(ASSET_IDS, len(log_df))
log_df["timestamp"] = pd.date_range("2026-01-01", periods=len(log_df), freq="3min")

# Select compact, model-friendly feature set + railway-relevant columns
feature_cols = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","count","srv_count","serror_rate","srv_serror_rate",
    "same_srv_rate","diff_srv_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_serror_rate"
]

out_cols = ["timestamp","asset_id"] + feature_cols + ["threat_category","is_attack"]
log_df = log_df[out_cols]

log_df.to_csv("/home/claude/prahari/data/cyber_log_dataset.csv", index=False)
print(f"Saved {len(log_df)} cyber log entries")
print(log_df["threat_category"].value_counts())
print(log_df.head())
