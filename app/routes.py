import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from app import db
from app.inference import score_transaction
from app.models import Customer, Decision, Prediction, Transaction, User

auth_bp = Blueprint("auth", __name__)
main_bp = Blueprint("main", __name__)

MERCHANT_RISK_LOOKUP = json.loads(
    (Path(__file__).resolve().parent.parent / "model" / "merchant_risk_lookup.json").read_text()
)
DEFAULT_MERCHANT_RISK = round(sum(MERCHANT_RISK_LOOKUP.values()) / len(MERCHANT_RISK_LOOKUP), 1)

MERCHANT_CATEGORIES = list(MERCHANT_RISK_LOOKUP.keys())
CARD_TYPES = ["Visa", "Mastercard", "Verve"]
AUTH_METHODS = ["OTP", "3D Secure", "No Authentication", "Biometric", "PIN"]
CHANNELS = ["Internet Banking", "POS", "Mobile App", "USSD", "ATM"]
DEVICE_TYPES = ["Android Phone", "Mac", "iPhone", "POS Terminal", "ATM Machine",
                "Tablet", "Windows PC", "Smart Watch"]
NIGERIAN_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue",
    "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu",
    "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi",
    "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo",
    "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara",
    "Abuja (FCT)",
]
LOCATION_BUCKETS = {
    "same_city": {"label": "Same city", "distance_km": 5, "is_foreign": False},
    "same_state": {"label": "Same state, different city", "distance_km": 60, "is_foreign": False},
    "other_state": {"label": "Different state in Nigeria", "distance_km": 350, "is_foreign": False},
    "abroad": {"label": "Outside Nigeria", "distance_km": 2500, "is_foreign": True},
}


def now_utc():
    return datetime.now(timezone.utc)


def as_aware_utc(dt):
    """Postgres' plain DateTime column returns naive datetimes on read even
    though everything this app writes is UTC (see models.py) — SQLite
    happens not to hit this, which is why it only ever showed up in
    production. Every value here is already UTC, so just attach the tzinfo
    back on if it's missing, rather than changing the column type and
    needing to reset the live database."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password_hash, request.form["password"]):
            login_user(user)
            return redirect(url_for("main.dashboard"))
        flash("Invalid username or password", "error")
    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))


@main_bp.route("/")
@login_required
def dashboard():
    total = Transaction.query.count()
    flagged = Prediction.query.filter_by(is_flagged=True).count()
    pending = (
        Prediction.query.filter_by(is_flagged=True)
        .outerjoin(Decision)
        .filter(Decision.id.is_(None))
        .count()
    )
    recent = Transaction.query.order_by(Transaction.created_at.desc()).limit(15).all()
    return render_template(
        "dashboard.html", total=total, flagged=flagged, pending=pending, recent=recent
    )


@main_bp.route("/customers", methods=["GET", "POST"])
@login_required
def customers():
    if request.method == "POST":
        account_number = request.form["account_number"].strip()
        existing = Customer.query.filter_by(account_number=account_number).first()
        if existing:
            flash(f"Account number {account_number} is already in use by {existing.full_name}.", "error")
            return redirect(url_for("main.customers"))

        customer = Customer(
            full_name=request.form["full_name"],
            account_number=account_number,
            card_type=request.form["card_type"],
            card_age_months=int(request.form["card_age_months"]),
            customer_age=int(request.form["customer_age"]),
            account_balance_ngn=float(request.form["account_balance_ngn"]),
            prior_disputes=int(request.form.get("prior_disputes") or 0),
            home_state=request.form["home_state"],
        )
        db.session.add(customer)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            flash(f"Account number {account_number} is already in use.", "error")
            return redirect(url_for("main.customers"))
        flash(f"Customer {customer.full_name} added.", "success")
        return redirect(url_for("main.new_transaction", customer_id=customer.id))

    all_customers = Customer.query.order_by(Customer.full_name).all()
    return render_template(
        "customers.html",
        customers=all_customers,
        card_types=CARD_TYPES,
        home_states=NIGERIAN_STATES,
    )


@main_bp.route("/transaction/new", methods=["GET", "POST"])
@login_required
def new_transaction():
    if request.method == "POST":
        customer = Customer.query.get_or_404(int(request.form["customer_id"]))
        bucket_key = request.form["location_bucket"]
        bucket = LOCATION_BUCKETS[bucket_key]
        merchant_category = request.form["merchant_category"]

        now = now_utc()
        last_txn = (
            Transaction.query.filter_by(customer_id=customer.id)
            .order_by(Transaction.created_at.desc())
            .first()
        )
        if last_txn:
            hours_since_last_txn = round((now - as_aware_utc(last_txn.created_at)).total_seconds() / 3600, 2)
        else:
            hours_since_last_txn = 72.0  # no history yet: treat as a normal gap

        txn_count_last_24h = Transaction.query.filter(
            Transaction.customer_id == customer.id,
            Transaction.created_at >= now - timedelta(hours=24),
        ).count()

        velocity_score = min(100.0, txn_count_last_24h * 8 + max(0.0, 15 - hours_since_last_txn * 3))
        merchant_risk_score = MERCHANT_RISK_LOOKUP.get(merchant_category, DEFAULT_MERCHANT_RISK)

        txn = Transaction(
            customer_id=customer.id,
            amount_ngn=float(request.form["amount_ngn"]),
            merchant_category=merchant_category,
            auth_method=request.form["auth_method"],
            channel=request.form["channel"],
            device_type=request.form["device_type"],
            cvv_retry_count=int(request.form.get("cvv_retry_count") or 0),
            location_bucket=bucket_key,
            is_new_merchant="is_new_merchant" in request.form,
            used_vpn="used_vpn" in request.form,
            ip_country_mismatch="ip_country_mismatch" in request.form,
            billing_shipping_mismatch="billing_shipping_mismatch" in request.form,
            is_ai_generated_scam_attempt="is_ai_generated_scam_attempt" in request.form,
            is_foreign_transaction=bucket["is_foreign"],
            distance_from_home_km=bucket["distance_km"],
            hours_since_last_txn=hours_since_last_txn,
            txn_count_last_24h=txn_count_last_24h,
            velocity_score=velocity_score,
            merchant_risk_score=merchant_risk_score,
            time_of_day_hour=now.hour,
            day_of_week=now.weekday(),
        )
        db.session.add(txn)
        db.session.flush()

        feature_input = {
            "amount_ngn": txn.amount_ngn,
            "merchant_category": txn.merchant_category,
            "card_type": customer.card_type,
            "auth_method": txn.auth_method,
            "channel": txn.channel,
            "device_type": txn.device_type,
            "is_foreign_transaction": txn.is_foreign_transaction,
            "hours_since_last_txn": txn.hours_since_last_txn,
            "txn_count_last_24h": txn.txn_count_last_24h,
            "distance_from_home_km": txn.distance_from_home_km,
            "card_age_months": customer.card_age_months,
            "customer_age": customer.customer_age,
            "account_balance_ngn": customer.account_balance_ngn,
            "is_new_merchant": txn.is_new_merchant,
            "used_vpn": txn.used_vpn,
            "ip_country_mismatch": txn.ip_country_mismatch,
            "billing_shipping_mismatch": txn.billing_shipping_mismatch,
            "cvv_retry_count": txn.cvv_retry_count,
            "velocity_score": txn.velocity_score,
            "time_of_day_hour": txn.time_of_day_hour,
            "day_of_week": txn.day_of_week,
            "is_ai_generated_scam_attempt": txn.is_ai_generated_scam_attempt,
            "merchant_risk_score": txn.merchant_risk_score,
            "prior_disputes": customer.prior_disputes,
        }
        result = score_transaction(feature_input)
        prediction = Prediction(
            transaction_id=txn.id,
            risk_score=result["risk_score"],
            is_flagged=result["is_flagged"],
            threshold_used=result["threshold_used"],
        )
        db.session.add(prediction)
        db.session.commit()

        return redirect(url_for("main.transaction_detail", txn_id=txn.id))

    preselected_customer_id = request.args.get("customer_id", type=int)
    return render_template(
        "new_transaction.html",
        all_customers=Customer.query.order_by(Customer.full_name).all(),
        preselected_customer_id=preselected_customer_id,
        merchant_categories=MERCHANT_CATEGORIES,
        auth_methods=AUTH_METHODS,
        channels=CHANNELS,
        device_types=DEVICE_TYPES,
        location_buckets=LOCATION_BUCKETS,
    )


@main_bp.route("/transaction/<int:txn_id>")
@login_required
def transaction_detail(txn_id):
    txn = Transaction.query.get_or_404(txn_id)
    return render_template("transaction_detail.html", txn=txn, prediction=txn.prediction,
                            location_buckets=LOCATION_BUCKETS)


@main_bp.route("/transaction/<int:txn_id>/decide", methods=["POST"])
@login_required
def decide(txn_id):
    txn = Transaction.query.get_or_404(txn_id)
    prediction = txn.prediction
    if prediction.decision is None:
        db.session.add(Decision(
            prediction_id=prediction.id,
            staff_id=current_user.id,
            action=request.form["action"],
            notes=request.form.get("notes", ""),
        ))
        db.session.commit()
        flash(f"Decision recorded: {request.form['action']}", "success")
    return redirect(url_for("main.transaction_detail", txn_id=txn_id))


@main_bp.route("/history")
@login_required
def history():
    transactions = Transaction.query.order_by(Transaction.created_at.desc()).all()
    return render_template("history.html", transactions=transactions)
