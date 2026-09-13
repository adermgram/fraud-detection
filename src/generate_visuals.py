"""Generate report-ready charts and diagrams from the already-trained model.

Reuses the saved model/encoders (no retraining) and reconstructs the exact
same train/test split used in train.py (same random_state), so every number
plotted here matches reports/metrics_report.txt exactly.

Run: python generate_visuals.py   (from the src/ directory)
Output: ../reports/figures/*.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from features import TARGET, build_feature_frame, feature_columns

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "credit_card_fraud_2026.csv"
MODEL_DIR = ROOT / "model"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# From the actual training run (reports/metrics_report.txt) — XGBoost isn't
# saved to disk since RandomForest won the ROC-AUC comparison, so its number
# is carried over here just for the comparison chart.
XGBOOST_TEST_AUC = 0.6849

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
})

FRAUD_COLOR = "#d64550"
LEGIT_COLOR = "#3f7cac"
ACCENT = "#2e2e38"


def load_test_predictions():
    df = pd.read_csv(DATA_PATH).drop(columns=["transaction_id"])
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
    model = joblib.load(MODEL_DIR / "fraud_model.pkl")
    threshold = json.loads((MODEL_DIR / "threshold.json").read_text())

    X = build_feature_frame(df, encoders, fit=False)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    proba = model.predict_proba(X_test)[:, 1]
    return df, y, model, X_test, y_test, proba, threshold


def chart_class_distribution(y):
    counts = y.value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=[LEGIT_COLOR, FRAUD_COLOR])
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c + 200, f"{c:,}\n({c/len(y):.1%})",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of transactions")
    ax.set_title("Class Distribution — Fraud vs Legitimate\n(20,000 transactions total)")
    ax.set_ylim(0, max(counts.values) * 1.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_distribution.png", dpi=180)
    plt.close(fig)


def chart_roc(y_test, proba):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=FRAUD_COLOR, linewidth=2.5, label=f"RandomForest (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set")
    ax.legend(loc="lower right", fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_roc_curve.png", dpi=180)
    plt.close(fig)


def chart_precision_recall(y_test, proba, operating_threshold):
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=ACCENT, linewidth=2.5)

    # mark the operating point actually used by the app
    idx = np.argmin(np.abs(thresholds - operating_threshold))
    ax.scatter([recall[idx]], [precision[idx]], color=FRAUD_COLOR, s=90, zorder=5,
               label=f"Operating threshold ({operating_threshold:.1%})\n"
                     f"Recall {recall[idx]:.1%}, Precision {precision[idx]:.1%}")
    ax.set_xlabel("Recall (fraud caught)")
    ax.set_ylabel("Precision (alerts that are real fraud)")
    ax.set_title("Precision–Recall Curve — Test Set")
    ax.legend(loc="upper right", fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_precision_recall_curve.png", dpi=180)
    plt.close(fig)


def chart_confusion_matrix(y_test, proba, operating_threshold):
    preds = (proba >= operating_threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"])
    disp.plot(ax=ax, cmap="Reds", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix\nOperating Threshold ({operating_threshold:.1%})", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_feature_importance(model):
    importances = sorted(zip(feature_columns(), model.feature_importances_),
                          key=lambda t: t[1], reverse=True)[:15]
    names = [n for n, _ in importances][::-1]
    vals = [v for _, v in importances][::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(names, vals, color=ACCENT)
    ax.set_xlabel("Feature importance (Gini)")
    ax.set_title("Top 15 Feature Importances — RandomForest")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_feature_importance.png", dpi=180)
    plt.close(fig)


def chart_model_comparison(rf_auc):
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["RandomForest\n(selected)", "XGBoost"]
    vals = [rf_auc, XGBOOST_TEST_AUC]
    colors = [FRAUD_COLOR, "grey"]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("ROC-AUC (test set)")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_model_comparison.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Workflow / architecture diagrams
# ---------------------------------------------------------------------------

def draw_flow(stages, title, filename, max_per_row=4, box_w=2.6, box_h=1.1,
              gap_x=0.5, gap_y=1.3, fontsize=9.5):
    """Draw a left-to-right flowchart, wrapping into extra rows as needed.
    `stages` is a list of (label, note) tuples; note is optional extra text.
    Every row reads left-to-right (no snaking) so stage index == reading order,
    which keeps arrow routing simple and correct."""
    n_rows = -(-len(stages) // max_per_row)  # ceil
    fig_w = max_per_row * (box_w + gap_x) + 1
    fig_h = n_rows * (box_h + gap_y) + 1
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    positions = {}
    for i, (label, note) in enumerate(stages):
        row_idx, col_idx = divmod(i, max_per_row)
        x = 0.7 + col_idx * (box_w + gap_x)
        y_top = fig_h - 1 - row_idx * (box_h + gap_y)
        is_last = i == len(stages) - 1
        box = FancyBboxPatch(
            (x, y_top - box_h), box_w, box_h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.4, edgecolor=ACCENT,
            facecolor=FRAUD_COLOR if is_last else "#eef1f5",
        )
        ax.add_patch(box)
        label_text = label if not note else f"{label}\n{note}"
        ax.text(x + box_w / 2, y_top - box_h / 2, label_text, ha="center", va="center",
                fontsize=fontsize, color="white" if is_last else ACCENT)
        positions[i] = (x, y_top, box_w, box_h, row_idx, col_idx)

    for i in range(len(stages) - 1):
        x1, y1, w1, h1, r1, c1 = positions[i]
        x2, y2, w2, h2, r2, c2 = positions[i + 1]
        if r1 == r2:
            # same row: short straight arrow, right edge -> left edge
            start = (x1 + w1, y1 - h1 / 2)
            end = (x2, y2 - h2 / 2)
            arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     color=ACCENT, linewidth=1.3)
            ax.add_patch(arrow)
        else:
            # wrap to next row: right-angle elbow (down, across, down) through
            # the empty gap between rows — unambiguous "continues below",
            # unlike a single diagonal line spanning the whole row width.
            xa = x1 + w1 / 2   # bottom-center of the last box in this row
            ya = y1 - h1
            xb = x2 + w2 / 2   # top-center of the first box in the next row
            yb = y2
            y_mid = (ya + yb) / 2
            ax.plot([xa, xa], [ya, y_mid], color=ACCENT, linewidth=1.3)
            ax.plot([xa, xb], [y_mid, y_mid], color=ACCENT, linewidth=1.3)
            arrow = FancyArrowPatch((xb, y_mid), (xb, yb), arrowstyle="-|>",
                                     mutation_scale=14, color=ACCENT, linewidth=1.3)
            ax.add_patch(arrow)

    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def diagram_training_pipeline():
    stages = [
        ("Raw Dataset", "Kaggle CSV, 20,000 rows"),
        ("Nigerian Vocabulary\nMapping", "channels, cards,\nmerchants, NGN"),
        ("Feature Engineering", "log(amount), night flag,\nlabel encoding"),
        ("Train/Test Split", "80/20, stratified"),
        ("SMOTE Oversampling", "training split only"),
        ("Train Models", "RandomForest\n& XGBoost"),
        ("Evaluate", "ROC-AUC on\nreal test set"),
        ("Select Best Model", "highest ROC-AUC"),
        ("Threshold Tuning", "target ~75% recall"),
        ("Save Artifacts", "model.pkl, encoders.pkl,\nthreshold.json"),
    ]
    draw_flow(stages, "Model Training Pipeline", "07_training_pipeline.png", max_per_row=5)


def diagram_app_workflow():
    stages = [
        ("Staff selects/creates\nCustomer", "profile: card, age,\nbalance, disputes"),
        ("New Transaction Form", "amount, merchant, channel,\ndevice, auth, location"),
        ("Auto-Derive Features", "server clock +\ncustomer's txn history"),
        ("Assemble Feature\nVector", "26 features, same\nencoding as training"),
        ("Trained Model Scores\nTransaction", "predict_proba()"),
        ("Risk Score +\nFlag/Clear", "compared to\noperating threshold"),
        ("Staff Decision", "Approve / Investigate\n/ Reject"),
        ("Dashboard & History", "counts, audit trail"),
    ]
    draw_flow(stages, "Fraud Review Application Workflow", "08_app_workflow.png", max_per_row=4)


def diagram_erd():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("Database Schema (Entity Relationships)", fontsize=13, fontweight="bold", pad=10)

    box_w, box_h = 2.2, 1.1
    raw_entities = {
        "user": ("User\n(staff login)", 0.6, 4.0),
        "customer": ("Customer\n(account profile)", 0.6, 1.2),
        "transaction": ("Transaction\n(one submitted txn)", 3.6, 1.2),
        "prediction": ("Prediction\n(risk score, flag)", 6.4, 1.2),
        "decision": ("Decision\n(Approve/Investigate/Reject)", 6.4, 4.0),
    }
    # each entry: edge midpoints, computed once, so arrows always land exactly
    # on a box boundary instead of an arbitrary hand-picked coordinate
    entities = {}
    for key, (name, x, y) in raw_entities.items():
        box = FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.08,rounding_size=0.12",
                              linewidth=1.4, edgecolor=ACCENT, facecolor="#eef1f5")
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + box_h / 2, name, ha="center", va="center",
                fontsize=9.5, color=ACCENT)
        entities[key] = {
            "left": (x, y + box_h / 2), "right": (x + box_w, y + box_h / 2),
            "top": (x + box_w / 2, y + box_h), "bottom": (x + box_w / 2, y),
        }

    def arrow(start, end, label):
        ar = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                              color=ACCENT, linewidth=1.3)
        ax.add_patch(ar)
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.15, label, fontsize=8.5, ha="center", color=ACCENT)

    arrow(entities["customer"]["right"], entities["transaction"]["left"], "1 -> many")
    arrow(entities["transaction"]["right"], entities["prediction"]["left"], "1 -> 1")
    arrow(entities["prediction"]["top"], entities["decision"]["bottom"], "1 -> 1")
    arrow(entities["user"]["right"], entities["decision"]["left"], "1 -> many")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_database_erd.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    df, y, model, X_test, y_test, proba, threshold = load_test_predictions()
    rf_auc = roc_auc_score(y_test, proba)
    op_threshold = threshold["operating_threshold"]

    chart_class_distribution(y)
    chart_roc(y_test, proba)
    chart_precision_recall(y_test, proba, op_threshold)
    chart_confusion_matrix(y_test, proba, op_threshold)
    chart_feature_importance(model)
    chart_model_comparison(rf_auc)

    diagram_training_pipeline()
    diagram_app_workflow()
    diagram_erd()

    print(f"Saved {len(list(FIG_DIR.glob('*.png')))} images to {FIG_DIR}")


if __name__ == "__main__":
    main()
