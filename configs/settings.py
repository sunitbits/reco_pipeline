"""Central configuration (paths, K, split, quality-gate thresholds). Every stage reads/writes through these paths, so tasks
communicate only via files (re-runnable individually, Airflow-friendly)."""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("RECO_HOME", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
RAW, ERR = DATA / "raw", DATA / "errors"
PROC, FEAT = DATA / "processed", DATA / "features"
VALID = PROC / "validated"          # validated (type-cast) tables, before preparation
REP, MODELS = ROOT / "reports", ROOT / "models"

SEED = int(os.environ.get("RECO_SEED", 42))
K = 10                      # evaluation cutoff
REL_THRESHOLD = 4           # rating >= 4 counts as "relevant"
TEST_FRACTION = 0.2         # most recent 20% of time -> test
REFERENCE_DATE = pd.Timestamp("2025-01-01")   # fixed "now" => reproducible validation
DATA_START, DATA_END = "2023-01-01", "2024-12-31"
MAX_REJECT_RATE = 0.10      # quality gate: fail pipeline if >10% rows rejected

def ensure_dirs():
    for p in (RAW, VALID, ERR, PROC, FEAT, REP, MODELS):
        p.mkdir(parents=True, exist_ok=True)

def save_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load_csv(path, parse_dates=()):
    return pd.read_csv(path, parse_dates=list(parse_dates))
