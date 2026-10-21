"""Simple food delivery delay prediction model (scikit-learn).

Predicts how many minutes an order will be late vs. the promised ETA.
Uses synthetic data so it runs out of the box; swap in your own CSV
by replacing make_data() with pd.read_csv(...).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG = np.random.default_rng(42)

NUMERIC = ["distance_km", "prep_time_min", "num_items", "courier_experience_months", "hour"]
CATEGORICAL = ["traffic", "weather", "is_weekend"]


def make_data(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic delivery records with a realistic-ish delay signal."""
    df = pd.DataFrame({
        "distance_km": RNG.gamma(2.0, 1.8, n).clip(0.3, 15),
        "prep_time_min": RNG.normal(15, 5, n).clip(5, 40),
        "num_items": RNG.integers(1, 9, n),
        "courier_experience_months": RNG.integers(0, 60, n),
        "hour": RNG.integers(10, 23, n),
        "traffic": RNG.choice(["low", "medium", "high"], n, p=[0.3, 0.45, 0.25]),
        "weather": RNG.choice(["clear", "rain", "storm"], n, p=[0.7, 0.25, 0.05]),
        "is_weekend": RNG.choice(["yes", "no"], n, p=[0.3, 0.7]),
    })

    traffic_effect = df["traffic"].map({"low": 0, "medium": 3, "high": 8})
    weather_effect = df["weather"].map({"clear": 0, "rain": 3, "storm": 9})
    rush_hour = df["hour"].isin([12, 13, 18, 19, 20]).astype(int) * 4
    weekend = (df["is_weekend"] == "yes").astype(int) * 2

    df["delay_min"] = (
        -4
        + 1.2 * df["distance_km"]
        + 0.35 * (df["prep_time_min"] - 15)
        + 0.5 * df["num_items"]
        - 0.06 * df["courier_experience_months"]
        + traffic_effect + weather_effect + rush_hour + weekend
        + RNG.normal(0, 2.5, n)  # noise
    )
    return df


def build_model() -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("prep", preprocess),
        ("model", GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        )),
    ])


def main() -> None:
    df = make_data()
    X, y = df[NUMERIC + CATEGORICAL], df["delay_min"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f} min")
    print(f"R²:  {r2_score(y_test, preds):.3f}")

    # Which features matter most?
    imp = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
    ranking = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
    print("\nFeature importance (permutation):")
    print(ranking.round(3).to_string())

    # Predict a new order
    new_order = pd.DataFrame([{
        "distance_km": 6.5, "prep_time_min": 22, "num_items": 5,
        "courier_experience_months": 4, "hour": 19,
        "traffic": "high", "weather": "rain", "is_weekend": "no",
    }])
    print(f"\nPredicted delay for sample order: {model.predict(new_order)[0]:.1f} min")


if __name__ == "__main__":
    main()
