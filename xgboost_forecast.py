"""XGBoost wind power model driven by meteorological features.

Usage: python src/xgboost_forecast.py data/Koru_Wind.csv
"""
import sys

import pandas as pd
import xgboost as xgb

from metrics import evaluate

FEATURES = ["Wind Speed (m/s)", "Temperature (°C)", "Air Density (kg/m³)"]
TARGET = "Total Power (MW)"


def main(path: str) -> None:
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df = df.sort_values("datetime")

    X, y = df[FEATURES], df[TARGET]

    # Chronological 80/20 split: train on the past, test on the future (no look-ahead)
    cut = int(len(df) * 0.8)
    X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    for k, v in evaluate(y_test, y_pred).items():
        print(f"{k}: {v:.3f}")

    pd.DataFrame({"Actual (MW)": y_test.values, "Predicted (MW)": y_pred}).to_csv(
        "xgboost_predictions.csv", index=False
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/Koru_Wind.csv")
