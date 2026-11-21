import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data/personal_finance_tracker_dataset.csv"
TARGET = "actual_savings"

# Exclude identifiers, direct/leaky derived fields, and unrelated labels.
NUMERIC_FEATURES = [
    "monthly_income",
    "monthly_expense_total",
    "budget_goal",
    "credit_score",
    "debt_to_income_ratio",
    "loan_payment",
    "investment_amount",
    "subscription_services",
    "emergency_fund",
    "transaction_count",
    "discretionary_spending",
    "essential_spending",
    "rent_or_mortgage",
]
CATEGORICAL_FEATURES = [
    "financial_scenario",
    "income_type",
    "category",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def main():
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET, "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", TARGET]).copy()
    df = df.drop_duplicates()  # exact duplicate rows only
    df = df.sort_values("date").reset_index(drop=True)

    # Chronological split: all rows from later dates stay in test set.
    unique_dates = sorted(df["date"].dropna().unique())
    if len(unique_dates) < 2:
        raise ValueError("Need at least two distinct dates for chronological split.")
    split_index = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * 0.8)))
    cutoff = pd.Timestamp(unique_dates[split_index])

    train_df = df[df["date"] < cutoff]
    test_df = df[df["date"] >= cutoff]
    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split created an empty train/test set.")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model_specs = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, min_samples_leaf=2
        ),
    }
    results = {}

    print(f"Rows after exact duplicate removal: {len(df)}")
    print(f"Training rows: {len(train_df)} | Testing rows: {len(test_df)}")
    print(f"Test period begins: {cutoff.date()}")

    for name, estimator in model_specs.items():
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        results[name] = {"pipeline": pipe, "mae": mae, "r2": r2}
        print(f"\n{name}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.3f}")

    # Pick the model with lower MAE on the held-out later dates.
    best_name = min(results, key=lambda n: results[n]["mae"])
    best_pipe = results[best_name]["pipeline"]

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": best_pipe,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "model_name": best_name,
        "metrics": {
            name: {"mae": float(info["mae"]), "r2": float(info["r2"])}
            for name, info in results.items()
        },
        "test_start_date": str(cutoff.date()),
        "note": "Model trained on user-provided dataset; validate provenance and suitability before real-world use."
    }, "models/savings_model.joblib")

    print(f"\nSelected model by lower test MAE: {best_name}")
    print("Saved: models/savings_model.joblib")
    print("Note: metrics only describe this dataset's held-out later-date rows.")

if __name__ == "__main__":
    main()
