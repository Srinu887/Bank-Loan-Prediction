import os
import re
import sqlite3
from pathlib import Path
from datetime import datetime
from functools import wraps

import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"
MODEL_PATH = BASE_DIR / "loan_model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"

app = Flask(__name__)
app.secret_key = "CHANGE_THIS_TO_A_RANDOM_SECRET_KEY"

# -----------------------------
# Validation rules
# -----------------------------
FIELD_RULES = {
    "age": {
        "label": "Age",
        "required": True,
        "type": "float",
        "min": 18,
        "max": 65,
        "placeholder": "",
        "range_text": "18 to 65",
    },
    "experience": {
        "label": "Experience",
        "required": True,
        "type": "float",
        "min": 0,
        "max": 50,
        "placeholder": "",
        "range_text": "0 to 50 years",
    },
    "income": {
        "label": "Income",
        "required": True,
        "type": "float",
        "min": 0,
        "max": 1000,
        "placeholder": "",
        "range_text": "0 to 1000",
    },
    "family": {
        "label": "Family",
        "required": True,
        "type": "int",
        "min": 1,
        "max": 10,
        "placeholder": "",
        "range_text": "1 to 10",
    },
    "ccavg": {
        "label": "CCAvg",
        "required": True,
        "type": "float",
        "min": 0,
        "max": 20,
        "placeholder": "",
        "range_text": "0 to 20",
    },
    "education": {
        "label": "Education",
        "required": True,
        "type": "int",
        "choices": [1, 2, 3],
        "placeholder": "",
        "range_text": "1, 2, or 3",
    },
    "mortgage": {
        "label": "Mortgage",
        "required": True,
        "type": "float",
        "min": 0,
        "max": 10000,
        "placeholder": "",
        "range_text": "0 to 10000",
    },
    "securities": {
        "label": "Securities Account",
        "required": True,
        "type": "int",
        "choices": [0, 1],
        "placeholder": "",
        "range_text": "0 or 1",
    },
    "cd": {
        "label": "CD Account",
        "required": True,
        "type": "int",
        "choices": [0, 1],
        "placeholder": "",
        "range_text": "0 or 1",
    },
    "online": {
        "label": "Online",
        "required": True,
        "type": "int",
        "choices": [0, 1],
        "placeholder": "",
        "range_text": "0 or 1",
    },
    "creditcard": {
        "label": "CreditCard",
        "required": True,
        "type": "int",
        "choices": [0, 1],
        "placeholder": "",
        "range_text": "0 or 1",
    },
}

REGISTER_RULES = {
    "name": {"label": "Name", "required": True, "min_len": 2, "max_len": 100},
    "email": {"label": "Email", "required": True},
    "password": {"label": "Password", "required": True, "min_len": 6, "max_len": 128},
    "confirm": {"label": "Confirm Password", "required": True},
}

# -----------------------------
# Load ML artifacts
# -----------------------------
model = None
scaler = None

def load_artifacts():
    global model, scaler

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler file: {SCALER_PATH}")

    model = tf.keras.models.load_model(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))

load_artifacts()

# -----------------------------
# DB helpers
# -----------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        age REAL,
        experience REAL,
        income REAL,
        family REAL,
        ccavg REAL,
        education INTEGER,
        mortgage REAL,
        securities_account INTEGER,
        cd_account INTEGER,
        online INTEGER,
        creditcard INTEGER,
        probability REAL,
        decision TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Auth helpers
# -----------------------------
def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper

def current_user():
    if "user_id" not in session:
        return None
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    conn.close()
    return user

# -----------------------------
# Utility helpers
# -----------------------------
def empty_prediction_form():
    return {
        "age": "",
        "experience": "",
        "income": "",
        "family": "",
        "ccavg": "",
        "education": "",
        "mortgage": "",
        "securities": "",
        "cd": "",
        "online": "",
        "creditcard": "",
    }

def is_valid_email(email: str) -> bool:
    pattern = r"^[^\s@]+@[^\s@]+\.[^\s@]+$"
    return bool(re.match(pattern, email or ""))

def convert_value(raw_value, expected_type):
    if expected_type == "int":
        if raw_value == "":
            raise ValueError("This field is required.")
        if "." in str(raw_value).strip():
            f = float(raw_value)
            if not f.is_integer():
                raise ValueError("Must be a whole number.")
            return int(f)
        return int(raw_value)
    elif expected_type == "float":
        if raw_value == "":
            raise ValueError("This field is required.")
        return float(raw_value)
    return raw_value

def validate_single_field(field_name, raw_value):
    if field_name not in FIELD_RULES:
        return False, "Unknown field.", None

    rule = FIELD_RULES[field_name]
    value_text = (raw_value or "").strip()

    if rule.get("required") and value_text == "":
        return False, f"{rule['label']} is required.", None

    try:
        value = convert_value(value_text, rule["type"])
    except ValueError as e:
        return False, f"{rule['label']}: {str(e)}", None

    if "choices" in rule and value not in rule["choices"]:
        choices_txt = ", ".join(map(str, rule["choices"]))
        return False, f"{rule['label']} must be one of: {choices_txt}.", None

    if "min" in rule and value < rule["min"]:
        return False, f"{rule['label']} must be at least {rule['min']}.", None

    if "max" in rule and value > rule["max"]:
        return False, f"{rule['label']} must be at most {rule['max']}.", None

    if field_name == "age" and not (18 <= value <= 65):
        return False, "Age must be between 18 and 65.", None

    return True, "", value

def validate_prediction_form(form_data):
    errors = {}
    cleaned = {}

    for field_name in FIELD_RULES.keys():
        raw_value = (form_data.get(field_name) or "").strip()
        ok, msg, parsed_value = validate_single_field(field_name, raw_value)

        if not ok:
            errors[field_name] = msg
            cleaned[field_name] = raw_value
        else:
            cleaned[field_name] = parsed_value

    return cleaned, errors

def validate_register_form(form_data):
    errors = {}

    name = (form_data.get("name") or "").strip()
    email = (form_data.get("email") or "").strip().lower()
    password = form_data.get("password") or ""
    confirm = form_data.get("confirm") or ""

    if not name:
        errors["name"] = "Name is required."
    elif len(name) < 2:
        errors["name"] = "Name must be at least 2 characters."
    elif len(name) > 100:
        errors["name"] = "Name must be at most 100 characters."

    if not email:
        errors["email"] = "Email is required."
    elif not is_valid_email(email):
        errors["email"] = "Enter a valid email address."

    if not password:
        errors["password"] = "Password is required."
    elif len(password) < 6:
        errors["password"] = "Password must be at least 6 characters."
    elif len(password) > 128:
        errors["password"] = "Password is too long."

    if not confirm:
        errors["confirm"] = "Confirm password is required."
    elif password != confirm:
        errors["confirm"] = "Passwords do not match."

    cleaned = {
        "name": name,
        "email": email,
        "password": password,
        "confirm": confirm,
    }
    return cleaned, errors

# -----------------------------
# ML predict
# -----------------------------
def predict_loan_internal(age, exp, income, family, ccavg, education,
                          mortgage, securities, cd, online, creditcard):
    x = np.array([[
        float(age), float(exp), float(income), float(family), float(ccavg), float(education),
        float(mortgage), float(securities), float(cd), float(online), float(creditcard)
    ]], dtype=float)

    x_scaled = scaler.transform(x)
    prob = float(model.predict(x_scaled, verbose=0)[0][0])
    decision = "✅ Loan Approved" if prob > 0.5 else "❌ Loan Not Approved"
    return prob, decision

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    form = {"name": "", "email": "", "password": "", "confirm": ""}
    errors = {}

    if request.method == "POST":
        form = {
            "name": (request.form.get("name") or "").strip(),
            "email": (request.form.get("email") or "").strip().lower(),
            "password": request.form.get("password") or "",
            "confirm": request.form.get("confirm") or "",
        }

        cleaned, errors = validate_register_form(request.form)

        if errors:
            for msg in errors.values():
                flash(msg, "danger")
            return render_template(
                "register.html",
                form=form,
                errors=errors,
                register_rules=REGISTER_RULES
            )

        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (
                    cleaned["name"],
                    cleaned["email"],
                    generate_password_hash(cleaned["password"]),
                    datetime.now().isoformat()
                )
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            flash("Email already registered. Please login.", "warning")
            return redirect(url_for("login"))

        conn.close()
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template(
        "register.html",
        form=form,
        errors=errors,
        register_rules=REGISTER_RULES
    )

@app.route("/login", methods=["GET", "POST"])
def login():
    form = {"email": ""}
    errors = {}

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        form["email"] = email

        if not email:
            errors["email"] = "Email is required."
        elif not is_valid_email(email):
            errors["email"] = "Enter a valid email address."

        if not password:
            errors["password"] = "Password is required."

        if errors:
            for msg in errors.values():
                flash(msg, "danger")
            return render_template("login.html", form=form, errors=errors)

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "danger")
            return render_template("login.html", form=form, errors={})

        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        flash(f"Welcome, {user['name']}!", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html", form=form, errors=errors)

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    user = current_user()
    result = None
    errors = {}
    form = empty_prediction_form()

    if request.method == "POST":
        form = {k: (request.form.get(k) or "").strip() for k in FIELD_RULES.keys()}
        cleaned, errors = validate_prediction_form(request.form)

        if errors:
            for msg in errors.values():
                flash(msg, "danger")
            return render_template(
                "dashboard.html",
                user=user,
                form=form,
                result=result,
                errors=errors,
                field_rules=FIELD_RULES
            )

        try:
            prob, decision = predict_loan_internal(
                cleaned["age"],
                cleaned["experience"],
                cleaned["income"],
                cleaned["family"],
                cleaned["ccavg"],
                cleaned["education"],
                cleaned["mortgage"],
                cleaned["securities"],
                cleaned["cd"],
                cleaned["online"],
                cleaned["creditcard"]
            )

            conn = get_db()
            conn.execute("""
                INSERT INTO predictions (
                    user_id, age, experience, income, family, ccavg, education, mortgage,
                    securities_account, cd_account, online, creditcard,
                    probability, decision, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user["id"],
                cleaned["age"],
                cleaned["experience"],
                cleaned["income"],
                cleaned["family"],
                cleaned["ccavg"],
                cleaned["education"],
                cleaned["mortgage"],
                cleaned["securities"],
                cleaned["cd"],
                cleaned["online"],
                cleaned["creditcard"],
                prob,
                decision,
                datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()

            result = {"prob": prob, "decision": decision}
            form = empty_prediction_form()
            flash("Prediction completed successfully.", "success")

        except Exception as e:
            flash(f"Prediction failed: {e}", "danger")

    return render_template(
        "dashboard.html",
        user=user,
        form=form,
        result=result,
        errors=errors,
        field_rules=FIELD_RULES
    )

@app.route("/validate_field", methods=["POST"])
@login_required
def validate_field():
    data = request.get_json(silent=True) or {}
    field_name = data.get("field")
    raw_value = data.get("value", "")

    ok, msg, parsed = validate_single_field(field_name, str(raw_value))
    return jsonify({
        "valid": ok,
        "message": msg,
        "field": field_name,
        "value": raw_value,
        "parsed_value": parsed if ok else None
    })

@app.route("/history")
@login_required
def history():
    user = current_user()
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM predictions
        WHERE user_id = ?
        ORDER BY id ASC
        LIMIT 200
    """, (user["id"],)).fetchall()
    conn.close()
    return render_template("history.html", user=user, rows=rows)

@app.route("/clear_history", methods=["POST"])
@login_required
def clear_history():
    user = current_user()
    conn = get_db()
    conn.execute("DELETE FROM predictions WHERE user_id = ?", (user["id"],))
    conn.commit()
    conn.close()
    flash("History cleared.", "info")
    return redirect(url_for("history"))

if __name__ == "__main__":
    print("✅ Loan Flask App running: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)