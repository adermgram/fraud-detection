"""Build a screenshot-driven user manual PDF for the SafeGuard web app.

Requires reports/screenshots/*.png to already exist — generate them by
running capture_screenshots.py against a locally running instance of the
app (python run.py) first.

Run: python generate_user_manual.py   (from the src/ directory)
Output: ../reports/SafeGuard_User_Manual.pdf
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
SHOT_DIR = ROOT / "reports" / "screenshots"
OUT_PATH = ROOT / "reports" / "SafeGuard_User_Manual.pdf"

NAVY = colors.HexColor("#101a34")
ACCENT = colors.HexColor("#3f7cac")
MUTED = colors.HexColor("#5b6270")
LIGHT_BG = colors.HexColor("#f2f4f8")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("CoverTitle", fontSize=26, leading=32, textColor=NAVY,
                           fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle("CoverSubtitle", fontSize=13, leading=18, textColor=MUTED,
                           alignment=TA_CENTER, spaceAfter=6))
styles.add(ParagraphStyle("H1", fontSize=17, leading=22, textColor=NAVY,
                           fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=10))
styles.add(ParagraphStyle("H2", fontSize=12.5, leading=16, textColor=ACCENT,
                           fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle("Body", fontSize=10.3, leading=15.5, textColor=colors.HexColor("#1f2430"),
                           alignment=TA_LEFT, spaceAfter=8))
styles.add(ParagraphStyle("Caption", fontSize=9, leading=12, textColor=MUTED,
                           alignment=TA_CENTER, spaceBefore=4, spaceAfter=14, fontName="Helvetica-Oblique"))
styles.add(ParagraphStyle("Callout", fontSize=10.3, leading=15.5, textColor=colors.HexColor("#1f2430"),
                           backColor=LIGHT_BG, borderPadding=10, spaceAfter=12, spaceBefore=4))
styles.add(ParagraphStyle("Step", fontSize=10.3, leading=15.5, textColor=colors.HexColor("#1f2430"),
                           leftIndent=14, spaceAfter=4))
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


def step(n, text):
    story.append(Paragraph(f"<b>{n}.</b> {text}", styles["Step"]))


def callout(text):
    story.append(Paragraph(text, styles["Callout"]))


def screenshot(filename, caption, width=5.6 * inch, append=True):
    path = SHOT_DIR / filename
    with PILImage.open(path) as im:
        w, h = im.size
    height = width * h / w
    box = TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d7dce5")),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ])
    framed = Table([[Image(str(path), width=width, height=height)]])
    framed.setStyle(box)
    flowables = [Spacer(1, 6), framed, Paragraph(caption, styles["Caption"])]
    if append:
        story.extend(flowables)
    return flowables


def glossary_table(rows):
    header = [Paragraph("Term / Field", styles["TableHead"]), Paragraph("What it means", styles["TableHead"])]
    data = [header] + [
        [Paragraph(f"<b>{r[0]}</b>", styles["TableCell"]), Paragraph(r[1], styles["TableCell"])]
        for r in rows
    ]
    t = Table(data, colWidths=[1.7 * inch, 4.6 * inch], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde2ea")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))


# =====================================================================
# COVER
# =====================================================================
story.append(Spacer(1, 2.2 * inch))
story.append(Paragraph("SafeGuard", styles["CoverTitle"]))
story.append(Paragraph("User Manual — Fraud Detection & Review System", styles["CoverSubtitle"]))
story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph(
    "A walkthrough of every screen, what to enter, and what the results mean.",
    ParagraphStyle("CoverNote", fontSize=10, leading=15, textColor=MUTED, alignment=TA_CENTER)
))
story.append(PageBreak())

# =====================================================================
# 1. GETTING STARTED
# =====================================================================
h1("1. Getting Started")
body(
    "SafeGuard is a staff tool for reviewing bank transactions for fraud risk. Every transaction "
    "submitted is scored automatically by a trained model, and anything risky enough is flagged for "
    "a staff member to review and decide on."
)
h2("Logging in")
step(1, "Open the app in a browser.")
step(2, "Enter your staff username and password and click <b>Log in</b>.")
screenshot("01_login.png", "The login screen.")

# =====================================================================
# 2. DASHBOARD
# =====================================================================
story.append(PageBreak())
h1("2. The Dashboard")
body("This is the home screen after logging in — a quick overview of activity.")
screenshot("09_dashboard_with_data.png", "The dashboard, showing two submitted transactions.")
glossary_table([
    ("Total Transactions", "How many transactions have been submitted in total."),
    ("Flagged as Risky", "How many transactions scored above the risk threshold and were flagged for review."),
    ("Pending Review", "How many <i>flagged</i> transactions still don't have a staff decision recorded."),
    ("Recent Transactions table", "The latest submissions, each with its risk score and status. Click "
     "<b>View</b> to open the full detail page for any of them."),
])

# =====================================================================
# 3. ADDING A CUSTOMER
# =====================================================================
story.append(PageBreak())
h1("3. Adding a Customer")
body(
    "Before checking a transaction, the customer making it must exist in the system. A customer's "
    "profile is entered once and reused for every transaction they make afterwards — you don't "
    "re-type their card details or balance each time."
)
screenshot("03_customers.png", "The Customers page: add a new profile on the left, browse existing ones on the right.")
glossary_table([
    ("Card Type", "Visa, Mastercard, or Verve — the card scheme on the customer's account."),
    ("Card Age (months)", "How long the customer has had this card. Newer cards are statistically higher-risk."),
    ("Account Balance", "The customer's current account balance, in Naira."),
    ("Prior Disputes", "How many times this customer has previously disputed a transaction. Leave at 0 for a new customer."),
    ("Home State", "Where the customer is based. Shown for reference only — it doesn't feed the risk "
     "model directly (the companion model report covers exactly which fields the model does use)."),
])

# =====================================================================
# 4. SUBMITTING A TRANSACTION
# =====================================================================
story.append(PageBreak())
h1("4. Submitting a New Transaction")
body(
    "This is the core screen: enter the details of a transaction and the system will score it for "
    "fraud risk immediately."
)
screenshot("04_new_transaction_blank.png", "The New Transaction form.", width=5.9 * inch)
callout(
    "<b>Every field on this form is meant to be filled in by hand for a real transaction.</b> The "
    "“Quick-fill a demo scenario” dropdown in the top right is only there as an optional shortcut "
    "for testing/demos — it pre-fills sample values, but nothing requires you to use it."
)
screenshot("05_new_transaction_dropdown_open.png", "The optional demo-scenario dropdown — Likely Legit, Borderline, or Likely Fraud.")
glossary_table([
    ("Customer Account", "Who this transaction belongs to. Must be added first (Section 3) if new."),
    ("Amount (NGN)", "The transaction value in Naira."),
    ("Merchant Category", "What kind of business the payment went to."),
    ("Channel", "How the transaction was made — Internet Banking, POS, Mobile App, USSD, or ATM."),
    ("Device Type", "The device used for the transaction."),
    ("Auth Method", "How the transaction was authenticated (OTP, PIN, Biometric, 3D Secure, or No Authentication)."),
    ("Transaction Location", "How far this transaction is from the customer's usual location — Same city, "
     "Same state, a different state, or outside Nigeria. This is a simplified stand-in for an exact "
     "distance, and also sets whether the transaction counts as “foreign.”"),
    ("CVV Retry Count", "How many times the card security code was mistyped before this transaction went through."),
    ("Risk Indicators", "A set of yes/no flags for things staff or the system may already know about this "
     "specific transaction — e.g. a VPN was detected, or the billing and shipping addresses don't match."),
])

# =====================================================================
# 5. READING THE RESULT
# =====================================================================
story.append(PageBreak())
h1("5. Reading the Risk Result")
body("After submitting, you land on the transaction's detail page with an immediate verdict.")
screenshot("06_transaction_detail_clear.png", "A low-risk transaction: scored CLEAR.", width=5.9 * inch)
screenshot("07_transaction_detail_flagged.png", "A high-risk transaction: scored FLAGGED FOR REVIEW.", width=5.9 * inch)
glossary_table([
    ("Risk score", "A percentage from the model: how likely this transaction is to be fraudulent. "
     "Higher = riskier."),
    ("Operating threshold", "The cutoff risk score above which a transaction gets flagged. Currently "
     "9.6% — deliberately set low so more fraud gets caught, at the cost of more false alarms (see "
     "the companion model report for why)."),
    ("CLEAR", "The risk score was below the threshold — no action needed."),
    ("FLAGGED FOR REVIEW", "The risk score was at or above the threshold — a staff member should look "
     "at it and record a decision (Section 6)."),
    ("“Auto-derived from customer history”", "Hours Since Last Txn, Txns in Last 24h, Velocity Score, "
     "and Merchant Risk Score are <i>not</i> typed by staff — the system calculates them automatically "
     "from the customer's recent activity and the current time. They're shown here for transparency, "
     "not as fields you fill in."),
    ("The small tags at the bottom", "(Foreign, New Merchant, VPN, IP Mismatch, etc.) — these just restate "
     "which risk-indicator checkboxes were ticked when the transaction was submitted."),
])

# =====================================================================
# 6. RECORDING A DECISION
# =====================================================================
story.append(PageBreak())
h1("6. Recording a Staff Decision")
body(
    "A flagged transaction isn't blocked automatically — a staff member reviews it and records what "
    "should happen next."
)
screenshot("08_decision_recorded.png", "A decision recorded on a flagged transaction.", width=5.9 * inch)
glossary_table([
    ("Approve", "Reviewed and judged legitimate despite being flagged — let it stand."),
    ("Investigate", "Not clear-cut — needs further follow-up (e.g. contacting the customer) before a final call."),
    ("Reject", "Judged fraudulent — the transaction should be treated as unauthorized."),
    ("Notes", "Optional free text to record why this decision was made — useful for an audit trail."),
])
body(
    "Once submitted, a decision cannot be changed from this screen — the form is replaced with the "
    "recorded outcome, as shown above."
)

# =====================================================================
# 7. HISTORY
# =====================================================================
story.append(PageBreak())
h1("7. Transaction History")
body("Every transaction ever submitted, in one place.")
screenshot("10_history.png", "The History page.", width=5.9 * inch)
callout(
    "<b>One easily-missed detail:</b> the Decision column shows “Pending” for "
    "<i>any</i> transaction without a recorded decision — including ones marked CLEAR. This doesn't "
    "mean a clear transaction needs action; it simply hasn't had a decision entered. Only <i>flagged</i> "
    "transactions are what the Dashboard's “Pending Review” count actually tracks (Section 2)."
)

# =====================================================================
# 8. MOBILE USE
# =====================================================================
story.append(PageBreak())
h1("8. Using SafeGuard on a Phone")
body(
    "The app is fully responsive — every screen works on a phone, with the navigation menu collapsing "
    "into a menu button."
)
screenshot("11_mobile_dashboard.png", "The dashboard on a phone-sized screen.", width=2.6 * inch)
screenshot("12_mobile_nav_open.png", "Tapping the menu button opens the navigation.", width=2.6 * inch)

# =====================================================================
# 9. QUICK REFERENCE
# =====================================================================
story.append(PageBreak())
h1("9. Quick Reference: Things That Aren't Obvious at First Glance")
glossary_table([
    ("Why a small transaction can still get flagged", "The risk threshold is deliberately set low "
     "(9.6%) so the system catches more real fraud, accepting more false alarms as the trade-off — a "
     "flagged transaction just means “worth a look,” not “definitely fraud.”"),
    ("Why some fields aren't on the transaction form", "Customer age, card age, account balance, and "
     "prior disputes live on the <b>Customer</b> profile (Section 3), not the transaction form — "
     "they're entered once, not re-typed every time."),
    ("Why Time of Day / Day of Week aren't fields either", "They're taken automatically from the "
     "server clock at the moment you submit — not something staff enter."),
    ("What “Verve” is", "A Nigerian debit card scheme, alongside Visa and Mastercard."),
    ("What USSD means", "A basic-phone banking channel (dialing a short code like *737#) with no "
     "internet connection required — offered as a Channel option since it's common in Nigeria."),
    ("If you see a “Page Not Found” or error page", "You've hit a broken/old link, or something went "
     "wrong server-side. Use the button on that page to get back to the Dashboard."),
])

doc = SimpleDocTemplate(
    str(OUT_PATH), pagesize=A4,
    topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    title="SafeGuard - User Manual",
)
doc.build(story)
print(f"Wrote {OUT_PATH}")
