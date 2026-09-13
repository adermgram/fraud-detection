from datetime import datetime, timezone

from flask_login import UserMixin

from app import db


def now_utc():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=now_utc)


class Customer(db.Model):
    """A bank account holder. Profile fields are entered once and reused for
    every transaction, instead of being re-typed each time."""
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    account_number = db.Column(db.String(20), unique=True, nullable=False)
    card_type = db.Column(db.String(50), nullable=False)
    card_age_months = db.Column(db.Integer, nullable=False)
    customer_age = db.Column(db.Integer, nullable=False)
    account_balance_ngn = db.Column(db.Float, nullable=False)
    prior_disputes = db.Column(db.Integer, default=0)
    home_state = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=now_utc)

    transactions = db.relationship("Transaction", backref="customer", order_by="Transaction.created_at")


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey("customer.id"), nullable=False)

    # --- entered by staff for this specific transaction ---
    amount_ngn = db.Column(db.Float, nullable=False)
    merchant_category = db.Column(db.String(50), nullable=False)
    auth_method = db.Column(db.String(50), nullable=False)
    channel = db.Column(db.String(50), nullable=False)
    device_type = db.Column(db.String(50), nullable=False)
    cvv_retry_count = db.Column(db.Integer, default=0)
    location_bucket = db.Column(db.String(20), nullable=False)
    is_new_merchant = db.Column(db.Boolean, default=False)
    used_vpn = db.Column(db.Boolean, default=False)
    ip_country_mismatch = db.Column(db.Boolean, default=False)
    billing_shipping_mismatch = db.Column(db.Boolean, default=False)
    is_ai_generated_scam_attempt = db.Column(db.Boolean, default=False)

    # --- auto-derived at submission time (location bucket / customer history / clock) ---
    is_foreign_transaction = db.Column(db.Boolean, default=False)
    distance_from_home_km = db.Column(db.Float, nullable=False)
    hours_since_last_txn = db.Column(db.Float, nullable=False)
    txn_count_last_24h = db.Column(db.Integer, nullable=False)
    velocity_score = db.Column(db.Float, nullable=False)
    merchant_risk_score = db.Column(db.Float, nullable=False)
    time_of_day_hour = db.Column(db.Integer, nullable=False)
    day_of_week = db.Column(db.Integer, nullable=False)

    created_at = db.Column(db.DateTime, default=now_utc)

    prediction = db.relationship("Prediction", backref="transaction", uselist=False)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.Integer, db.ForeignKey("transaction.id"), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    is_flagged = db.Column(db.Boolean, nullable=False)
    threshold_used = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=now_utc)

    decision = db.relationship("Decision", backref="prediction", uselist=False)


class Decision(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("prediction.id"), nullable=False)
    staff_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    action = db.Column(db.String(20), nullable=False)  # Approve / Investigate / Reject
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=now_utc)

    staff = db.relationship("User")
