"""Build a plain-language project report PDF covering the dataset, model
choices, features, and evaluation results — meant to be handed to someone
who will write the full project report from it.

Run: python generate_report_pdf.py   (from the src/ directory)
Output: ../reports/Fraud_Detection_Project_Report.pdf
"""
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
OUT_PATH = ROOT / "reports" / "Fraud_Detection_Project_Report.pdf"

NAVY = colors.HexColor("#101a34")
ACCENT = colors.HexColor("#3f7cac")
DANGER = colors.HexColor("#d64550")
MUTED = colors.HexColor("#5b6270")
LIGHT_BG = colors.HexColor("#f2f4f8")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("CoverTitle", fontSize=26, leading=32, textColor=NAVY,
                           fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle("CoverSubtitle", fontSize=13, leading=18, textColor=MUTED,
                           alignment=TA_CENTER, spaceAfter=6))
styles.add(ParagraphStyle("H1", fontSize=17, leading=22, textColor=NAVY,
                           fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=10))
styles.add(ParagraphStyle("H2", fontSize=13, leading=17, textColor=ACCENT,
                           fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle("Body", fontSize=10.3, leading=15.5, textColor=colors.HexColor("#1f2430"),
                           alignment=TA_LEFT, spaceAfter=8))
styles.add(ParagraphStyle("Caption", fontSize=9, leading=12, textColor=MUTED,
                           alignment=TA_CENTER, spaceBefore=4, spaceAfter=14, fontName="Helvetica-Oblique"))
styles.add(ParagraphStyle("Callout", fontSize=10.3, leading=15.5, textColor=colors.HexColor("#1f2430"),
                           backColor=LIGHT_BG, borderPadding=10, spaceAfter=12, spaceBefore=4))
styles.add(ParagraphStyle("TableCell", fontSize=9, leading=12.5, textColor=colors.HexColor("#1f2430")))
styles.add(ParagraphStyle("TableHead", fontSize=9.3, leading=12.5, textColor=colors.white,
                           fontName="Helvetica-Bold"))

story = []


def h1(text):
    story.append(Paragraph(text, styles["H1"]))


def h2(text, append=True):
    p = Paragraph(text, styles["H2"])
    if append:
        story.append(p)
    return p


def body(text):
    story.append(Paragraph(text, styles["Body"]))


def callout(text):
    story.append(Paragraph(text, styles["Callout"]))


def figure(filename, caption, width=5.4 * inch, append=True):
    path = FIG_DIR / filename
    with PILImage.open(path) as im:
        w, h = im.size
    height = width * h / w
    flowables = [Spacer(1, 6), Image(str(path), width=width, height=height),
                 Paragraph(caption, styles["Caption"])]
    if append:
        story.extend(flowables)
    return flowables


def column_label(text):
    """Long snake_case names have no spaces, so a narrow table column would
    force reportlab to break them mid-word (e.g. "ho/me_km"). The audience
    for this report is explicitly non-technical, so showing the underlying
    column name with spaces instead of underscores both fixes the wrapping
    and reads more naturally (e.g. "distance from home km")."""
    return text.replace("_", " ")


def feature_table(rows):
    header = [Paragraph("Column", styles["TableHead"]), Paragraph("What it represents", styles["TableHead"]),
              Paragraph("Why it can signal fraud", styles["TableHead"])]
    data = [header] + [
        [Paragraph(f"<b>{column_label(r[0])}</b>", styles["TableCell"]), Paragraph(r[1], styles["TableCell"]),
         Paragraph(r[2], styles["TableCell"])]
        for r in rows
    ]
    t = Table(data, colWidths=[1.5 * inch, 2.3 * inch, 2.5 * inch], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde2ea")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))


def metric_table(rows, col_widths=None):
    header = [Paragraph(c, styles["TableHead"]) for c in rows[0]]
    data = [header] + [[Paragraph(str(c), styles["TableCell"]) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde2ea")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))


# =====================================================================
# COVER
# =====================================================================
story.append(Spacer(1, 2.2 * inch))
story.append(Paragraph("Fraud Detection System", styles["CoverTitle"]))
story.append(Paragraph("Model Workflow, Feature Guide & Evaluation Results", styles["CoverSubtitle"]))
story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph(
    "A plain-language walkthrough of the dataset, the modeling decisions, and what the "
    "results mean — written so it can be turned directly into a project report.",
    ParagraphStyle("CoverNote", fontSize=10, leading=15, textColor=MUTED, alignment=TA_CENTER)
))
story.append(PageBreak())

# =====================================================================
# 1. INTRODUCTION
# =====================================================================
h1("1. What This System Does")
body(
    "This project is a fraud detection system for bank card transactions. Given the details of a "
    "transaction (amount, merchant, channel, device, and a set of behavioural signals), a trained "
    "machine learning model produces a <b>risk score</b> — a percentage estimate of how likely that "
    "transaction is to be fraudulent. Transactions above a chosen risk threshold are flagged for a "
    "staff member to review and decide on (Approve, Investigate, or Reject)."
)
body(
    "The system does not attempt to auto-block transactions. This is a deliberate design choice, "
    "explained in Section 6: the cost of missing real fraud is far higher than the cost of a staff "
    "member spending a few minutes reviewing a false alarm, so the model is intentionally tuned to "
    "catch more fraud even if that means more transactions get flagged for review."
)

h2("Why This Is a Hard Problem")
body(
    "Fraud is rare. In this dataset, only 1.7% of transactions are fraudulent — the other 98.3% are "
    "completely normal. A model that just guessed “legitimate” every single time would be "
    "98.3% accurate and completely useless, since it would catch zero fraud. This is why the project "
    "does not use plain accuracy to judge the model (Section 5 explains this in detail) and why "
    "specific techniques (SMOTE, threshold tuning) are used to handle this imbalance."
)
figure("01_class_distribution.png",
       "Figure 1 — Class balance in the dataset: fraud is a small minority, which is realistic "
       "(real-world card fraud is often even rarer) but makes the model harder to train and evaluate fairly.")

# =====================================================================
# 2. FEATURES
# =====================================================================
story.append(PageBreak())
h1("2. Understanding the Data: What Each Column Means")
body(
    "Every transaction is described by a set of columns (features). Some describe the transaction "
    "itself, some describe the customer's account, and some are calculated automatically from recent "
    "activity. The table below explains each one in plain terms and why it plausibly relates to fraud. "
    "<i>Note: in the actual dataset and code, these column names use underscores instead of spaces "
    "(e.g. “distance from home km” appears in code as distance_from_home_km) — shown with spaces "
    "here for readability.</i>"
)

h2("Transaction details")
feature_table([
    ("amount_ngn", "The Naira value of the transaction.",
     "Unusually large amounts (relative to what's normal) are a classic fraud signal."),
    ("merchant_category", "The type of business the payment was made to (e.g. Restaurants, Crypto Exchange, Gaming/Betting).",
     "Some merchant types are inherently higher-risk for fraud/money-laundering than others (e.g. crypto exchanges)."),
    ("channel", "How the transaction was made: Internet Banking, POS, Mobile App, USSD, or ATM.",
     "Some channels have weaker identity checks than others, making them easier to exploit."),
    ("device_type", "The device used (Android Phone, iPhone, Windows PC, POS Terminal, ATM Machine, etc.).",
     "An unfamiliar or unusual device for a given channel can indicate account takeover."),
    ("auth_method", "How the transaction was authenticated: OTP, 3D Secure, PIN, Biometric, or No Authentication.",
     "Weaker authentication (especially 'No Authentication') removes a key barrier fraudsters must bypass."),
    ("cvv_retry_count", "How many times the card's security code was mistyped before this transaction.",
     "Repeated CVV failures often mean someone is guessing the code, a sign of card-not-present fraud."),
])

h2("Location & timing")
feature_table([
    ("distance_from_home_km", "How far the transaction location is from the customer's home.",
     "A transaction far from where the customer normally is can indicate a stolen card."),
    ("is_foreign_transaction", "Whether the transaction happened outside the customer's home country.",
     "Cross-border transactions are statistically more likely to be fraudulent."),
    ("time_of_day_hour / day_of_week", "When the transaction happened.",
     "Fraud disproportionately happens at unusual hours (e.g. late at night) when a legitimate card owner is unlikely to be transacting."),
])

h2("Account & customer profile")
feature_table([
    ("card_type", "Visa, Mastercard, or Verve.",
     "Card networks differ in adoption and fraud-prevention tooling, giving each a slightly different baseline risk."),
    ("card_age_months", "How long the card has existed.",
     "Newly issued cards are more often targeted or used in fraud rings than long-established ones."),
    ("customer_age", "The account holder's age.",
     "Certain age groups are statistically more or less likely to be fraud targets or victims."),
    ("account_balance_ngn", "The customer's current account balance.",
     "Balance relative to transaction size can indicate whether a purchase is plausible for that account."),
    ("prior_disputes", "How many times this customer has previously disputed a transaction.",
     "A history of disputes can mean the account is a repeat fraud target or otherwise higher-risk."),
])

h2("Behavioural signals (calculated automatically)")
feature_table([
    ("hours_since_last_txn", "Time gap since this customer's previous transaction.",
     "Fraud often comes in rapid bursts once a card is compromised — a very short gap is suspicious."),
    ("txn_count_last_24h", "How many transactions this customer has made in the last day.",
     "An unusually high transaction count in a short time is a hallmark of card testing/fraud rings."),
    ("velocity_score", "A single number summarizing how fast this account has been transacting recently.",
     "Combines the two signals above into one strong, easy-to-use fraud indicator (see Section 6 — it's one of the two most important features)."),
    ("merchant_risk_score", "The average historical fraud rate for this merchant category.",
     "Some categories (e.g. Crypto Exchange) are consistently riskier — this bakes that pattern directly into the model (the single most important feature)."),
])

h2("Explicit risk flags")
feature_table([
    ("is_new_merchant", "Whether this is the customer's first payment to this merchant.", "New payees are a common early sign of a compromised account being drained."),
    ("used_vpn", "Whether a VPN was detected.", "Fraudsters use VPNs to mask their real location."),
    ("ip_country_mismatch", "Whether the connection's IP address country differs from the card's country.", "A strong signal of remote/unauthorized access."),
    ("billing_shipping_mismatch", "Whether billing and shipping addresses differ.", "A classic online-fraud pattern — goods shipped somewhere other than the cardholder's address."),
    ("is_ai_generated_scam_attempt", "Whether the transaction shows signs of an AI-generated scam (e.g. deepfake-assisted social engineering).", "A modern, emerging fraud vector worth tracking explicitly."),
])

# =====================================================================
# 3. DATA PREPARATION
# =====================================================================
story.append(PageBreak())
h1("3. Preparing the Data")

h2("Adapting the dataset")
body(
    "The starting dataset is a Kaggle credit-card-fraud dataset using US terms (USD amounts, "
    "generic channel names, international card schemes). Before training, every category is "
    "remapped to a Nigerian banking context — amounts converted to Naira, channels renamed to "
    "Internet Banking/USSD/Mobile App/POS/ATM, and cards consolidated to Visa/Mastercard/Verve. "
    "This remapping happens <i>before</i> the model is trained, so the model genuinely learns on "
    "these categories rather than just having them relabeled afterwards for display."
)

h2("Turning raw data into model-ready numbers")
body(
    "Machine learning models need numbers, not text. Categories like “merchant_category” "
    "or “auth_method” are converted to numbers through a process called <b>label encoding</b> "
    "(each distinct category gets assigned an integer code, e.g. “Visa” = 0, "
    "“Mastercard” = 1). True/False flags are converted to 1/0. The transaction amount is "
    "also log-transformed (a standard technique that stops a few very large transactions from "
    "dominating the model's view of what's “normal”), and a simple “is it night-time?” "
    "flag is derived from the hour of day."
)

h2("Splitting the data fairly")
body(
    "The data is split 80% for training and 20% for testing, using a <b>stratified split</b> — meaning "
    "both the training set and the test set keep the same 1.7% fraud ratio. This matters because a "
    "careless random split could accidentally put almost no fraud cases in the test set, making the "
    "evaluation meaningless."
)

h2("Fixing the class imbalance: SMOTE")
callout(
    "<b>What is SMOTE?</b> Short for Synthetic Minority Oversampling Technique. Since fraud cases are "
    "rare, a model trained directly on the raw data would barely notice them and mostly learn to "
    "predict “legitimate”. SMOTE creates additional, synthetic (but realistic) fraud examples "
    "by interpolating between existing real fraud cases, until the training set has a roughly equal "
    "number of fraud and non-fraud examples. Crucially, <b>SMOTE is applied only to the training data</b> "
    "— the test set is left at its real, untouched 1.7% fraud rate, so the reported results reflect "
    "real-world conditions rather than an artificially easy, balanced test."
)

# =====================================================================
# 4. MODEL SELECTION
# =====================================================================
story.append(PageBreak())
h1("4. Choosing the Model")
body(
    "Two models were trained and compared: <b>Random Forest</b> and <b>XGBoost</b>. Both are "
    "“ensemble” methods that work by building many decision trees (a decision tree is a "
    "series of simple yes/no questions, like “Is the merchant risk score above 50? → Is the "
    "authentication method 'No Authentication'? → ...”) and combining their votes into a "
    "final answer. This family of model was chosen over deep learning because on structured, "
    "spreadsheet-style data like this, tree ensembles consistently perform as well or better, train "
    "in seconds rather than hours, and are far easier to interpret and explain — a real advantage "
    "when the results need to be defended and understood by people who aren't machine learning "
    "specialists."
)
figure("06_model_comparison.png",
       "Figure 2 — Random Forest scored a meaningfully higher ROC-AUC than XGBoost on the same "
       "held-out test set (0.753 vs 0.685), so it was selected as the deployed model.")
body(
    "<b>Why ROC-AUC and not accuracy to compare them?</b> As explained in Section 1, accuracy is "
    "misleading on this dataset. ROC-AUC (explained fully in Section 5) measures how well a model "
    "ranks fraud above legitimate transactions, regardless of any specific cutoff point, which makes "
    "it a fair way to compare two models."
)

# =====================================================================
# 5. EVALUATION METRICS EXPLAINED
# =====================================================================
story.append(PageBreak())
h1("5. Evaluation Metrics, Explained Simply")
body(
    "This section explains each evaluation term used in this project in plain language, before "
    "Section 6 shows what these numbers actually turned out to be for this model."
)

h2("The Confusion Matrix")
body(
    "Every prediction the model makes falls into one of four buckets, usually drawn as a 2×2 grid:"
)
metric_table([
    ["", "Model said: Legitimate", "Model said: Fraud"],
    ["Actually Legitimate", "True Negative (correct)", "False Positive (false alarm)"],
    ["Actually Fraud", "False Negative (missed fraud)", "True Positive (correctly caught)"],
], col_widths=[1.7 * inch, 1.9 * inch, 1.9 * inch])
body(
    "Every other metric below is just a different way of summarizing these four numbers."
)

h2("Accuracy — and why it's misleading here")
body(
    "Accuracy = (correct predictions) ÷ (all predictions). It sounds like the obvious choice, but "
    "on a dataset that's 98.3% legitimate, a model can score 98.3% accuracy by simply never flagging "
    "anything as fraud — a completely useless model. This is why accuracy is not used to judge this "
    "model's performance."
)

h2("Precision — “of everything we flagged, how much was real fraud?”")
body(
    "Precision = True Positives ÷ (True Positives + False Positives). A high precision means "
    "that when the system raises an alarm, it's usually right. Low precision means lots of false "
    "alarms — costly in staff time, but not dangerous, since nothing gets auto-blocked."
)

h2("Recall — “of all the real fraud, how much did we actually catch?”")
body(
    "Recall = True Positives ÷ (True Positives + False Negatives). A high recall means the model "
    "misses very little real fraud. Low recall means a lot of fraud slips through undetected — the "
    "costly, dangerous kind of mistake in this context."
)

h2("F1-score — a single balance of precision and recall")
body(
    "F1 is the harmonic mean of precision and recall — a single number that's only high when both "
    "precision and recall are reasonably good. It's useful for comparing models generally, but it "
    "treats “missed fraud” and “false alarm” as equally bad, which (as Section 6 "
    "explains) is not true for this system."
)

h2("ROC-AUC — how well the model ranks risk overall")
body(
    "ROC-AUC (“Area Under the ROC Curve”) measures how well the model separates fraud from "
    "legitimate transactions across every possible cutoff point, not just one. A score of 0.5 means "
    "the model is no better than random guessing; 1.0 would mean perfect separation. Because it "
    "doesn't depend on picking one specific threshold, it's the fairest single number for comparing "
    "two models, which is why it was used to choose between Random Forest and XGBoost in Section 4."
)

# =====================================================================
# 6. RESULTS
# =====================================================================
story.append(PageBreak())
h1("6. Our Results, Explained")

h2("How well the model separates fraud from legitimate transactions")
figure("02_roc_curve.png",
       "Figure 3 — ROC curve for the selected Random Forest model. AUC = 0.753, meaning the model "
       "does a genuinely meaningful (though imperfect) job of ranking fraud above legitimate "
       "transactions — a random model would trace the diagonal dashed line (AUC = 0.5).")

h2("Choosing the operating threshold")
body(
    "A model outputs a risk score from 0% to 100% for every transaction — it doesn't decide “fraud” "
    "or “not fraud” on its own. A cutoff (threshold) has to be chosen: any transaction scored "
    "above it gets flagged. Picking this cutoff is a genuine business decision, not a purely "
    "technical one, because of the trade-off shown below."
)
figure("03_precision_recall_curve.png",
       "Figure 4 — Precision vs. recall at every possible threshold. The marked point is the "
       "threshold actually used: 9.6%, chosen to guarantee catching about 75% of fraud, "
       "accepting a low precision (3.6%) as the cost of that."
       )
callout(
    "<b>Why prioritize recall over precision here?</b> In this system, a flagged transaction is not "
    "auto-blocked — it goes to a staff member to review (Approve / Investigate / Reject). A false "
    "alarm therefore costs a few minutes of staff time. A missed fraud case costs the bank real "
    "money. Given that imbalance in consequences, the threshold was deliberately chosen to catch "
    "more fraud (higher recall) even though it means far more transactions get flagged for review "
    "(lower precision) than a “balanced” threshold would produce. This is a considered "
    "trade-off, not an oversight — it is worth stating explicitly in a project report as evidence of "
    "understanding the real-world cost asymmetry in fraud detection."
)

heading = h2("The result, laid out plainly", append=False)
fig_flowables = figure("04_confusion_matrix.png",
                        "Figure 5 — Confusion matrix on the test set at the chosen 9.6% threshold.",
                        append=False)
story.append(KeepTogether([heading] + fig_flowables))
metric_table([
    ["Outcome", "Count", "What it means"],
    ["True Positives", "51", "Real fraud cases correctly flagged for review."],
    ["False Negatives", "17", "Real fraud cases the model missed (25% of all fraud in the test set)."],
    ["False Positives", "1,383", "Legitimate transactions incorrectly flagged (cost: staff review time)."],
    ["True Negatives", "2,549", "Legitimate transactions correctly left alone."],
], col_widths=[1.4 * inch, 0.8 * inch, 3.9 * inch])
body(
    "In plain terms: out of 68 real fraud cases in the test set, the system caught 51 of them "
    "(75% recall). To achieve that, it also flagged 1,383 legitimate transactions that turned out "
    "to be fine — meaning only about 1 in 28 flagged transactions is genuinely fraudulent (3.6% "
    "precision). That sounds low in isolation, but is expected and considered acceptable in fraud "
    "detection specifically because of the review workflow described above — this is a widely "
    "recognized, realistic trade-off, not a flaw unique to this project."
)

heading = h2("What the model actually learned to pay attention to", append=False)
fig_flowables = figure(
    "05_feature_importance.png",
    "Figure 6 — The features the model relied on most. merchant_risk_score and velocity_score "
    "dominate, followed by auth_method — all of which line up with well-known fraud indicators, "
    "supporting that the model learned genuine patterns rather than noise.",
    append=False,
)
story.append(KeepTogether([heading] + fig_flowables))

# =====================================================================
# 7. SYSTEM WORKFLOW
# =====================================================================
story.append(PageBreak())
h1("7. How It All Fits Together")
body(
    "The diagrams below summarize the two halves of the project: the offline process that produces "
    "the trained model, and the live application that uses it to score real transactions."
)
figure("07_training_pipeline.png", "Figure 7 — The model training pipeline, start to finish.")
figure("08_app_workflow.png", "Figure 8 — What happens when a staff member submits a transaction in the live application.")
figure("09_database_erd.png", "Figure 9 — How the application's data is structured.")

# =====================================================================
# 8. LIMITATIONS
# =====================================================================
story.append(PageBreak())
h1("8. Limitations & Honest Caveats")
body(
    "A good project report states its limitations clearly rather than hiding them — examiners and "
    "readers generally respond better to honesty here than to an implied claim of perfection."
)
feature_table([
    ("Synthetic data", "The dataset is Kaggle-generated, not real bank transaction logs (real data is not accessible for a project like this).",
     "Real-world patterns may differ; results should be read as a demonstration of method, not a production benchmark."),
    ("Fraud rate", "1.7% fraud is higher than real card fraud rates (often well under 0.5%).",
     "A more imbalanced real dataset would likely be even harder to achieve high precision on."),
    ("velocity_score formula", "In the live app, this is computed with a simple hand-written formula from recent transaction count and recency.",
     "A production system would derive this from much richer historical behavioural data."),
    ("Low precision in absolute terms", "Only ~3.6% of flagged transactions are truly fraudulent.",
     "Acceptable given the staff-review workflow, but a larger dataset and richer features (e.g. network/graph-based signals between accounts) would be the natural next step to improve this."),
])

# =====================================================================
# 9. CONCLUSION
# =====================================================================
h1("9. Conclusion")
body(
    "The trained Random Forest model shows genuine, meaningful ability to separate fraudulent from "
    "legitimate transactions (ROC-AUC 0.753), built on a dataset deliberately adapted to reflect a "
    "realistic class imbalance and a Nigerian banking context. Rather than using a default decision "
    "threshold, the operating point was deliberately chosen to catch about 75% of fraud, reflecting "
    "a considered judgment about the relative costs of missed fraud versus false alarms in a system "
    "where every flag goes to human review. The features the model relies on most — merchant "
    "risk, transaction velocity, and authentication strength — align with recognized fraud "
    "indicators, supporting that the model has learned real, defensible patterns."
)

doc = SimpleDocTemplate(
    str(OUT_PATH), pagesize=A4,
    topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    title="Fraud Detection System - Model Workflow & Evaluation Report",
)
doc.build(story)
print(f"Wrote {OUT_PATH}")
