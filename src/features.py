"""Shared preprocessing so training and the Flask app transform transactions identically.

The raw Kaggle CSV uses US vocabulary (USD amounts, "Online"/"Contactless"
channels, RuPay/Discover cards). This module remaps that vocabulary to a
Nigerian banking context *before* encoding, so the trained model and the
Flask app agree on the same category names end to end.
"""
import numpy as np
import pandas as pd

USD_TO_NGN = 1600

CHANNEL_MAP = {
    "Online": "Internet Banking",
    "POS": "POS",
    "In-App": "Mobile App",
    "Contactless": "USSD",
    "ATM": "ATM",
}
CARD_TYPE_MAP = {
    "Visa": "Visa",
    "Mastercard": "Mastercard",
    "Amex": "Verve",
    "RuPay": "Verve",
    "Discover": "Verve",
}
MERCHANT_CATEGORY_MAP = {
    "Restaurants": "Restaurants/Eateries",
    "Online Retail": "Online Retail (Jumia/Konga)",
    "Groceries": "Supermarket",
    "Streaming": "Streaming (Netflix/DSTV)",
    "Travel": "Travel/Airline",
    "Gift Cards": "Gift Cards/Vouchers",
    "Electronics": "Electronics",
    "Fuel": "Fuel Station",
    "Gaming": "Gaming/Betting",
    "Utilities": "Utility Bills",
    "Crypto Exchange": "Crypto Exchange",
    "Healthcare": "Hospital/Pharmacy",
}

# Location bucket -> representative distance from home (km), used by the app
# so staff pick a bucket instead of typing an exact distance.
LOCATION_BUCKETS = {
    "same_city": {"label": "Same city", "distance_km": 5, "is_foreign": False},
    "same_state": {"label": "Same state, different city", "distance_km": 60, "is_foreign": False},
    "other_state": {"label": "Different state in Nigeria", "distance_km": 350, "is_foreign": False},
    "abroad": {"label": "Outside Nigeria", "distance_km": 2500, "is_foreign": True},
}

CATEGORICAL_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOLEAN_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
NUMERIC_COLS = [
    "amount_ngn",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_ngn",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]
ENGINEERED_COLS = ["amount_log", "is_night"]

TARGET = "is_fraud"

NIGHT_HOURS = {0, 1, 2, 3, 4, 5}


def nigerianize_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw Kaggle CSV's US vocabulary onto Nigerian banking terms.

    Idempotent: rows already using Nigerian terms (e.g. from the Flask app)
    pass straight through the .map(...).fillna(original) below.
    """
    df = df.copy()
    df["channel"] = df["channel"].map(CHANNEL_MAP).fillna(df["channel"])
    df["card_type"] = df["card_type"].map(CARD_TYPE_MAP).fillna(df["card_type"])
    df["merchant_category"] = df["merchant_category"].map(MERCHANT_CATEGORY_MAP).fillna(df["merchant_category"])
    if "amount_usd" in df.columns:
        df["amount_ngn"] = df["amount_usd"] * USD_TO_NGN
    if "account_balance_usd" in df.columns:
        df["account_balance_ngn"] = df["account_balance_usd"] * USD_TO_NGN
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = nigerianize_raw(df)
    df["amount_log"] = np.log1p(df["amount_ngn"])
    df["is_night"] = df["time_of_day_hour"].isin(NIGHT_HOURS).astype(int)
    for col in BOOLEAN_COLS:
        df[col] = df[col].astype(bool).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict, fit: bool) -> pd.DataFrame:
    """Label-encode categorical columns in place using (or building) a dict of LabelEncoders."""
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    for col in CATEGORICAL_COLS:
        if fit:
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col].astype(str))
            encoders[col] = enc
        else:
            enc = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v, enc=enc: enc.transform([v])[0] if v in enc.classes_ else -1
            )
    return df


def feature_columns() -> list:
    return NUMERIC_COLS + ENGINEERED_COLS + BOOLEAN_COLS + CATEGORICAL_COLS


def build_feature_frame(df: pd.DataFrame, encoders: dict, fit: bool) -> pd.DataFrame:
    df = engineer_features(df)
    df = encode_categoricals(df, encoders, fit=fit)
    return df[feature_columns()]
