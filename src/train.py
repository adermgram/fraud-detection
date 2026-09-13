"""Train and evaluate fraud detection models, then save the best one for the Flask app."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import TARGET, build_feature_frame, feature_columns, nigerianize_raw

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "credit_card_fraud_2026.csv"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 42

# Minimum recall the deployed threshold must hit. Missing fraud costs the bank
# real money; a false alarm just costs a staff member a few minutes of review
# (see app workflow: Approve / Investigate / Reject), so we deliberately bias
# toward recall over precision when picking the operating threshold.
TARGET_RECALL = 0.75


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["transaction_id"])


def pick_threshold(y_true, y_proba, target_recall: float) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns thresholds of length n-1
    precision, recall = precision[:-1], recall[:-1]

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    f1_idx = int(np.argmax(f1))
    f1_optimal = {
        "threshold": float(thresholds[f1_idx]),
        "precision": float(precision[f1_idx]),
        "recall": float(recall[f1_idx]),
        "f1": float(f1[f1_idx]),
    }

    # Among thresholds that reach the target recall, pick the highest
    # (= best precision available) one; fall back to the lowest threshold
    # (max achievable recall) if the target is never reached.
    eligible = np.where(recall >= target_recall)[0]
    if len(eligible) > 0:
        best_idx = eligible[np.argmax(thresholds[eligible])]
    else:
        best_idx = int(np.argmax(recall))
    operating = {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1[best_idx]),
        "target_recall": target_recall,
    }
    return {"f1_optimal": f1_optimal, "operating": operating}


def evaluate(name, model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    preds_default = (proba >= 0.5).astype(int)
    return {
        "name": name,
        "model": model,
        "proba": proba,
        "auc": auc,
        "report_default": classification_report(y_test, preds_default, digits=3),
        "confusion_default": confusion_matrix(y_test, preds_default).tolist(),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_data()
    encoders: dict = {}
    X = build_feature_frame(df, encoders, fit=True)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train_res, y_train_res)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train_res, y_train_res)

    results = [
        evaluate("RandomForest", rf, X_test, y_test),
        evaluate("XGBoost", xgb, X_test, y_test),
    ]
    best = max(results, key=lambda r: r["auc"])

    thresholds = pick_threshold(y_test, best["proba"], TARGET_RECALL)

    importances = sorted(
        zip(feature_columns(), best["model"].feature_importances_),
        key=lambda t: t[1],
        reverse=True,
    )

    # Per-category average merchant_risk_score, so the app can auto-fill this
    # field from the chosen merchant category instead of asking staff to type
    # an arbitrary 0-100 number.
    nigerian_df = nigerianize_raw(df)
    merchant_risk_lookup = (
        nigerian_df.groupby("merchant_category")["merchant_risk_score"].mean().round(1).to_dict()
    )

    # --- Save artifacts ---
    joblib.dump(best["model"], MODEL_DIR / "fraud_model.pkl")
    joblib.dump(encoders, MODEL_DIR / "encoders.pkl")
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(feature_columns(), indent=2))
    (MODEL_DIR / "merchant_risk_lookup.json").write_text(json.dumps(merchant_risk_lookup, indent=2))
    (MODEL_DIR / "threshold.json").write_text(json.dumps({
        "model_used": best["name"],
        "operating_threshold": thresholds["operating"]["threshold"],
        "f1_optimal_threshold": thresholds["f1_optimal"]["threshold"],
        "rationale": (
            f"Operating threshold chosen to guarantee >= {TARGET_RECALL:.0%} recall "
            "on the held-out test set. Missed fraud costs the bank money; a false "
            "positive only costs a staff review (Approve/Investigate/Reject), so "
            "the system is deliberately biased toward catching fraud over raw precision."
        ),
        "operating_metrics": thresholds["operating"],
        "f1_optimal_metrics": thresholds["f1_optimal"],
    }, indent=2))

    report_lines = []
    report_lines.append("FRAUD DETECTION MODEL — TRAINING REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Dataset: {DATA_PATH.name}")
    report_lines.append(f"Rows: {len(df)}  Fraud rate: {y.mean():.4%}")
    report_lines.append(f"Train rows (post-SMOTE): {len(X_train_res)}  Test rows: {len(X_test)}")
    report_lines.append("")
    for r in results:
        report_lines.append(f"--- {r['name']} ---")
        report_lines.append(f"ROC-AUC: {r['auc']:.4f}")
        report_lines.append(f"Confusion matrix @0.5: {r['confusion_default']}")
        report_lines.append(r["report_default"])
        report_lines.append("")
    report_lines.append(f"Selected model: {best['name']} (highest ROC-AUC = {best['auc']:.4f})")
    report_lines.append("")
    report_lines.append("Threshold tuning on selected model:")
    report_lines.append(f"  F1-optimal   -> {thresholds['f1_optimal']}")
    report_lines.append(f"  Operating    -> {thresholds['operating']}")
    report_lines.append("")
    report_lines.append("Feature importances:")
    for name, imp in importances:
        report_lines.append(f"  {name:28s} {imp:.4f}")

    report_text = "\n".join(report_lines)
    (REPORT_DIR / "metrics_report.txt").write_text(report_text)

    print(report_text)


if __name__ == "__main__":
    main()
