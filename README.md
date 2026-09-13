# Fraud Detection System

A Flask app that scores bank transactions for fraud risk using a trained
RandomForest model, with a staff review workflow (Approve / Investigate /
Reject). Adapted to a Nigerian banking context: NGN amounts, local payment
channels (Internet Banking, USSD, POS, ATM), and local card schemes
(Visa/Mastercard/Verve).

## Stack

Flask · SQLAlchemy (Postgres in production, SQLite locally) · scikit-learn ·
Bootstrap

## Project layout

```
dataset/          raw source data (20,000 transactions, 1.7% fraud)
src/
  features.py     shared preprocessing (used by both training and the app)
  train.py        trains RandomForest + XGBoost, picks the best, tunes threshold
  generate_visuals.py   report charts + workflow diagrams
model/            trained model, encoders, threshold, merchant-risk lookup
reports/          evaluation report + chart images
app/
  __init__.py     app factory, DB + login setup, error handlers, demo seed data
  models.py       User, Customer, Transaction, Prediction, Decision
  routes.py       login, dashboard, customers, new transaction, history
  inference.py    loads the model and scores a single transaction
  templates/      Jinja2 + Bootstrap UI
run.py            entry point
```

## Setup

```
pip install -r requirements.txt
python run.py
```

Visit http://127.0.0.1:5000 — log in with `admin` / `admin123` (seeded on
first run).

## Retrain the model

```
pip install -r requirements-dev.txt   # adds xgboost, imbalanced-learn, matplotlib
cd src
python train.py
python generate_visuals.py
```

## Deploying for free (Render + Supabase)

The app reads `DATABASE_URL` from the environment and falls back to local
SQLite if unset. This matters in production: free hosts wipe local disk on
every restart, so a real Postgres database is required once deployed.

1. **Database** — create a free project on [Supabase](https://supabase.com).
   Click **Connect** → copy the **Session pooler** connection string (not
   "Direct connection" — that requires IPv6, which Render doesn't support
   outbound).
2. **Hosting** — on [Render](https://render.com): New → Blueprint → connect
   this repo (it reads `render.yaml` automatically). Paste the Supabase
   connection string as `DATABASE_URL` when prompted.
3. Every push to `main` auto-redeploys.

Free-tier notes: Render spins down after 15 min idle (~1 min to wake back
up); Supabase pauses after 7 days idle (dashboard → **Restore project**, no
data lost).

## How it works

1. Staff add a **Customer** profile once (name, card, balance, etc.).
2. On **New Transaction**, pick that customer and fill in the transaction
   specifics — amount, merchant, channel, device, auth method, location, risk
   flags. A "Quick-fill" dropdown offers demo scenarios, but manual entry is
   the default path.
3. Fields the model also needs — time of day, transaction velocity, merchant
   risk score, etc. — are computed automatically from the server clock and
   the customer's transaction history, not typed by staff.
4. The model returns a risk score and a Flagged/Clear verdict immediately.
5. If flagged, staff record a decision (Approve / Investigate / Reject).
6. **Dashboard** and **History** show outcomes across all transactions.

## Modeling notes

- **Imbalance**: fraud is 1.7% of transactions. SMOTE is applied to the
  training split only — evaluation uses the real class balance.
- **Model choice**: RandomForest over XGBoost, purely on ROC-AUC (~0.75 vs
  ~0.68) on the held-out test set.
- **Threshold**: tuned for ~75% recall rather than the default 0.5 cutoff.
  A flagged transaction goes to staff review, not an auto-block — so a missed
  fraud case costs more than an extra false alarm.
- **Top predictors**: merchant risk score, transaction velocity, and auth
  method dominate — consistent with known fraud indicators in the literature.

Full numbers and charts: `reports/metrics_report.txt` and `reports/figures/`.
