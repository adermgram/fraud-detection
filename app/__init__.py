import os
import sys
from pathlib import Path

from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"


def _database_uri() -> str:
    """Use a hosted Postgres in production (DATABASE_URL env var), local
    SQLite file otherwise. Free hosts (Render, Railway, etc.) wipe the local
    filesystem on every restart/redeploy, so SQLite can't be trusted there —
    a real DATABASE_URL is required once this is deployed."""
    url = os.environ.get("DATABASE_URL")
    if url:
        # Some providers (Render, Heroku-style) hand out "postgres://", but
        # SQLAlchemy 2.x only accepts the "postgresql://" scheme.
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url
    return f"sqlite:///{ROOT / 'app' / 'fraud_detection.db'}"


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
    app.config["SQLALCHEMY_DATABASE_URI"] = _database_uri()
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    login_manager.init_app(app)

    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    from app.routes import auth_bp, main_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()
        _seed_default_user()
        _seed_demo_customers()

    return app


def _seed_default_user():
    from werkzeug.security import generate_password_hash
    from app.models import User

    if User.query.count() == 0:
        db.session.add(User(
            username="admin",
            password_hash=generate_password_hash("admin123"),
        ))
        db.session.commit()


def _seed_demo_customers():
    from app.models import Customer

    if Customer.query.count() > 0:
        return
    demo_customers = [
        dict(full_name="Abubakar Ibrahim", account_number="2001112233", card_type="Verve",
             card_age_months=48, customer_age=34, account_balance_ngn=650_000,
             prior_disputes=0, home_state="Jigawa"),
        dict(full_name="Chiamaka Okafor", account_number="2004445566", card_type="Mastercard",
             card_age_months=20, customer_age=27, account_balance_ngn=1_250_000,
             prior_disputes=0, home_state="Lagos"),
        dict(full_name="Fatima Sule", account_number="2007778899", card_type="Visa",
             card_age_months=6, customer_age=22, account_balance_ngn=120_000,
             prior_disputes=1, home_state="Kano"),
    ]
    for data in demo_customers:
        db.session.add(Customer(**data))
    db.session.commit()
