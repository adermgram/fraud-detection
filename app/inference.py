import json
from pathlib import Path

import joblib
import pandas as pd

from features import build_feature_frame

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

_model = joblib.load(MODEL_DIR / "fraud_model.pkl")
_encoders = joblib.load(MODEL_DIR / "encoders.pkl")
_threshold = json.loads((MODEL_DIR / "threshold.json").read_text())
OPERATING_THRESHOLD = _threshold["operating_threshold"]


def score_transaction(txn: dict) -> dict:
    """Run the trained model on a single transaction dict and return risk info."""
    df = pd.DataFrame([txn])
    X = build_feature_frame(df, _encoders, fit=False)
    risk_score = float(_model.predict_proba(X)[0, 1])
    return {
        "risk_score": risk_score,
        "is_flagged": risk_score >= OPERATING_THRESHOLD,
        "threshold_used": OPERATING_THRESHOLD,
    }
