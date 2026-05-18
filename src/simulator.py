"""Synthetic user and portfolio simulation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

RISK_TIER_WEIGHTS: dict[str, float] = {
    "Prime": 0.40,
    "Near-Prime": 0.30,
    "Subprime": 0.20,
    "Deep-Subprime": 0.10,
}

RISK_TIER_ORDER = list(RISK_TIER_WEIGHTS.keys())

RISK_PROFILES: dict[str, dict[str, object]] = {
    "Prime": {
        "cibil_range": (750, 900),
        "income_probs": [0.08, 0.42, 0.32, 0.18],
        "employment_probs": [0.70, 0.18, 0.07, 0.05],
        "age_probs": [0.12, 0.33, 0.37, 0.18],
        "payment_streak_range": (18, 36),
        "utilization_beta": (3.0, 5.5),
        "account_age_range": (18, 120),
        "delinquency_probs": [0.78, 0.16, 0.04, 0.015, 0.004, 0.001],
        "transaction_range": (10, 40),
        "limit_range": (100000.0, 500000.0),
    },
    "Near-Prime": {
        "cibil_range": (680, 749),
        "income_probs": [0.22, 0.46, 0.24, 0.08],
        "employment_probs": [0.60, 0.20, 0.15, 0.05],
        "age_probs": [0.18, 0.37, 0.31, 0.14],
        "payment_streak_range": (9, 30),
        "utilization_beta": (2.5, 3.5),
        "account_age_range": (8, 96),
        "delinquency_probs": [0.55, 0.24, 0.11, 0.06, 0.03, 0.01],
        "transaction_range": (8, 32),
        "limit_range": (50000.0, 150000.0),
    },
    "Subprime": {
        "cibil_range": (580, 679),
        "income_probs": [0.42, 0.37, 0.16, 0.05],
        "employment_probs": [0.45, 0.18, 0.28, 0.09],
        "age_probs": [0.26, 0.34, 0.26, 0.14],
        "payment_streak_range": (2, 20),
        "utilization_beta": (2.1, 2.2),
        "account_age_range": (3, 72),
        "delinquency_probs": [0.28, 0.24, 0.19, 0.14, 0.09, 0.06],
        "transaction_range": (4, 24),
        "limit_range": (20000.0, 75000.0),
    },
    "Deep-Subprime": {
        "cibil_range": (300, 579),
        "income_probs": [0.58, 0.26, 0.12, 0.04],
        "employment_probs": [0.30, 0.12, 0.32, 0.26],
        "age_probs": [0.31, 0.30, 0.23, 0.16],
        "payment_streak_range": (0, 12),
        "utilization_beta": (1.8, 1.7),
        "account_age_range": (1, 48),
        "delinquency_probs": [0.14, 0.17, 0.19, 0.18, 0.17, 0.15],
        "transaction_range": (1, 18),
        "limit_range": (10000.0, 40000.0),
    },
}

AGE_BUCKETS = ["18-25", "26-35", "36-50", "51+"]
INCOME_BUCKETS = ["low", "mid", "high", "very_high"]
EMPLOYMENT_TYPES = ["salaried", "self_employed", "gig", "student"]
SPENDING_CATEGORIES = ["essentials", "lifestyle", "mixed"]

AGE_BUCKET_MIDPOINTS: dict[str, int] = {
    "18-25": 22,
    "26-35": 30,
    "36-50": 43,
    "51+": 57,
}

INCOME_SPEND_MULTIPLIER: dict[str, float] = {
    "low": 0.82,
    "mid": 0.96,
    "high": 1.08,
    "very_high": 1.18,
}

RISK_DEFAULT_BASE_RATE: dict[str, float] = {
    "Prime": 0.005,
    "Near-Prime": 0.02,
    "Subprime": 0.06,
    "Deep-Subprime": 0.15,
}

RISK_STRESS_SENSITIVITY: dict[str, float] = {
    "Prime": 0.35,
    "Near-Prime": 0.55,
    "Subprime": 0.70,
    "Deep-Subprime": 0.90,
}


@dataclass(slots=True)
class UserSchema:
    user_id: str
    age_bucket: str
    income_bucket: str
    cibil_score: int
    employment_type: str
    payment_streak: int
    utilization_ratio: float
    spending_category: str
    account_age_months: int
    delinquency_count: int
    transaction_frequency: int
    risk_tier: str
    initial_credit_limit: float


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_credit_limit(
    generator: np.random.Generator,
    risk_tier: str,
    income_bucket: str,
) -> float:
    low, high = RISK_PROFILES[risk_tier]["limit_range"]
    income_position = INCOME_BUCKETS.index(income_bucket) / (len(INCOME_BUCKETS) - 1)
    anchor = low + (high - low) * (0.25 + 0.65 * income_position)
    limit = generator.normal(loc=anchor, scale=(high - low) * 0.12)
    return round(float(np.clip(limit, low, high)), 2)


def _sample_user(
    generator: np.random.Generator,
    user_index: int,
    risk_tier: str,
) -> UserSchema:
    profile = RISK_PROFILES[risk_tier]

    age_bucket = generator.choice(AGE_BUCKETS, p=profile["age_probs"])
    employment_type = generator.choice(EMPLOYMENT_TYPES, p=profile["employment_probs"])
    income_bucket = generator.choice(INCOME_BUCKETS, p=profile["income_probs"])

    if employment_type == "student":
        income_bucket = generator.choice(["low", "mid"], p=[0.8, 0.2])
    elif employment_type == "gig" and income_bucket == "very_high":
        income_bucket = generator.choice(["mid", "high"], p=[0.65, 0.35])
    elif age_bucket == "18-25" and income_bucket == "very_high":
        income_bucket = generator.choice(["mid", "high"], p=[0.7, 0.3])
    elif age_bucket == "51+" and employment_type == "student":
        employment_type = generator.choice(
            ["salaried", "self_employed", "gig"], p=[0.45, 0.35, 0.20]
        )

    cibil_min, cibil_max = profile["cibil_range"]
    score_drift = {
        "low": -18,
        "mid": 0,
        "high": 14,
        "very_high": 24,
    }[income_bucket]
    cibil_score = int(
        np.clip(generator.integers(cibil_min, cibil_max + 1) + score_drift, 300, 900)
    )

    streak_min, streak_max = profile["payment_streak_range"]
    payment_streak = int(generator.integers(streak_min, streak_max + 1))

    age_midpoint = AGE_BUCKET_MIDPOINTS[age_bucket]
    if employment_type == "student":
        payment_streak = min(payment_streak, int(generator.integers(0, 12)))
    if age_midpoint >= 43:
        payment_streak = min(36, payment_streak + int(generator.integers(0, 4)))

    util_alpha, util_beta = profile["utilization_beta"]
    utilization_ratio = float(
        np.clip(generator.beta(util_alpha, util_beta), 0.02, 0.98)
    )
    if income_bucket == "low":
        utilization_ratio = float(np.clip(utilization_ratio + 0.06, 0.02, 0.98))
    elif income_bucket == "very_high":
        utilization_ratio = float(np.clip(utilization_ratio - 0.05, 0.02, 0.98))

    spending_category = generator.choice(
        SPENDING_CATEGORIES,
        p={
            "low": [0.56, 0.11, 0.33],
            "mid": [0.35, 0.21, 0.44],
            "high": [0.21, 0.34, 0.45],
            "very_high": [0.13, 0.45, 0.42],
        }[income_bucket],
    )
    if employment_type == "student":
        spending_category = generator.choice(["essentials", "mixed"], p=[0.62, 0.38])

    account_low, account_high = profile["account_age_range"]
    account_age_months = int(generator.integers(account_low, account_high + 1))
    if employment_type == "student":
        account_age_months = min(account_age_months, int(generator.integers(1, 25)))
    account_age_months = min(max(account_age_months, 1), 120)

    delinquency_count = int(
        generator.choice(np.arange(6), p=profile["delinquency_probs"])
    )
    if payment_streak >= 24:
        delinquency_count = max(0, delinquency_count - 1)
    if utilization_ratio >= 0.75 and delinquency_count < 5:
        delinquency_count += int(generator.choice([0, 1], p=[0.65, 0.35]))
    delinquency_count = min(delinquency_count, 5)

    tx_low, tx_high = profile["transaction_range"]
    transaction_frequency = int(generator.integers(tx_low, tx_high + 1))
    if spending_category == "lifestyle":
        transaction_frequency = min(
            40, transaction_frequency + int(generator.integers(2, 7))
        )
    elif spending_category == "essentials":
        transaction_frequency = max(
            1, transaction_frequency - int(generator.integers(0, 4))
        )

    initial_credit_limit = _sample_credit_limit(generator, risk_tier, income_bucket)

    return UserSchema(
        user_id=f"USER_{user_index:05d}",
        age_bucket=age_bucket,
        income_bucket=income_bucket,
        cibil_score=cibil_score,
        employment_type=employment_type,
        payment_streak=payment_streak,
        utilization_ratio=round(utilization_ratio, 4),
        spending_category=spending_category,
        account_age_months=account_age_months,
        delinquency_count=delinquency_count,
        transaction_frequency=transaction_frequency,
        risk_tier=risk_tier,
        initial_credit_limit=initial_credit_limit,
    )


def generate_users(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible synthetic user portfolio."""

    if n <= 0:
        raise ValueError("n must be > 0")
    generator = _rng(seed)
    risk_tiers = generator.choice(
        RISK_TIER_ORDER, size=n, p=list(RISK_TIER_WEIGHTS.values())
    )
    users = [
        _sample_user(generator, index, risk_tier)
        for index, risk_tier in enumerate(risk_tiers, start=1)
    ]
    return pd.DataFrame(asdict(user) for user in users)


def simulate_month(
    users_df: pd.DataFrame,
    month_num: int,
    economic_stress: float = 1.0,
) -> pd.DataFrame:
    """Simulate one month of spending and default outcomes."""

    if month_num < 1:
        raise ValueError("month_num must be >= 1")
    if economic_stress <= 0:
        raise ValueError("economic_stress must be > 0")
    if "initial_credit_limit" not in users_df.columns:
        raise ValueError("users_df must include an initial_credit_limit column")

    month_seed = 10_000 + month_num
    generator = _rng(month_seed)
    working_df = users_df.copy()
    credit_limits = working_df["initial_credit_limit"].to_numpy(dtype=float)
    if np.any(~np.isfinite(credit_limits)) or np.any(credit_limits <= 0):
        raise ValueError(
            "initial_credit_limit values must all be positive finite numbers"
        )

    target_ratio = generator.normal(loc=0.60, scale=0.12, size=len(working_df))
    spend_scaler = (
        working_df["income_bucket"].map(INCOME_SPEND_MULTIPLIER).to_numpy(dtype=float)
    )
    amount_spent = credit_limits * np.clip(
        target_ratio * spend_scaler,
        0.40,
        0.80,
    )
    amount_spent = np.minimum(amount_spent, credit_limits)

    current_utilization = np.clip(
        amount_spent / credit_limits,
        0.0,
        1.0,
    )

    base_rates = (
        working_df["risk_tier"].map(RISK_DEFAULT_BASE_RATE).to_numpy(dtype=float)
    )
    stress_sensitivity = (
        working_df["risk_tier"].map(RISK_STRESS_SENSITIVITY).to_numpy(dtype=float)
    )
    stored_utilization = working_df["utilization_ratio"].to_numpy(dtype=float)
    delinquency_penalty = 1.0 + 0.02 * working_df["delinquency_count"].to_numpy(
        dtype=float
    )
    streak_discount = np.clip(
        1.0 - 0.002 * working_df["payment_streak"].to_numpy(dtype=float), 0.92, 1.0
    )

    utilization_multiplier = (
        0.81 + 0.20 * current_utilization + 0.08 * stored_utilization
    )
    stress_multiplier = 1.0 + max(economic_stress - 1.0, 0.0) * stress_sensitivity
    if economic_stress < 1.0:
        stress_multiplier = np.clip(
            1.0 - (1.0 - economic_stress) * (0.20 + 0.30 * stress_sensitivity),
            0.75,
            None,
        )

    default_probability = np.clip(
        base_rates
        * utilization_multiplier
        * stress_multiplier
        * delinquency_penalty
        * streak_discount,
        0.0005,
        0.95,
    )

    did_default = generator.random(len(working_df)) < default_probability
    outstanding_amount = np.where(
        did_default,
        amount_spent * generator.uniform(0.72, 1.0, size=len(working_df)),
        amount_spent * generator.uniform(0.08, 0.35, size=len(working_df)),
    )

    return pd.DataFrame(
        {
            "user_id": working_df["user_id"],
            "month": month_num,
            "amount_spent": np.round(amount_spent, 2),
            "did_default": did_default.astype(bool),
            "outstanding_amount": np.round(outstanding_amount, 2),
        }
    )
