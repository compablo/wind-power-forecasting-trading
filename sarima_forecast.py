"""SARIMA day-ahead wind power forecast (3-hour resolution, 24 h horizon).

Usage: python src/sarima_forecast.py data/Koru_Wind_EnhancedData.csv
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from metrics import evaluate

TARGET = "Total Power (MW)"
HORIZON = 8  # 8 x 3 h = 24 h ahead


def load(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.drop_duplicates(subset="datetime").set_index("datetime").asfreq("h")
    df = df.interpolate()
    return df[TARGET].resample("3h").mean()


def main(path: str) -> None:
    series = load(path)

    adf_stat, p_value, *_ = adfuller(series.dropna())
    print(f"ADF statistic: {adf_stat:.3f}  p-value: {p_value:.4f}")

    # Hold out the last 24 h so the forecast is scored on data the model never saw
    train, test = series.iloc[:-HORIZON], series.iloc[-HORIZON:]

    model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 1, 1, 8))
    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=HORIZON)

    for k, v in evaluate(test, forecast).items():
        print(f"{k}: {v:.3f}")

    plt.figure(figsize=(10, 4))
    plt.plot(series.index[-100:], series.iloc[-100:], label="Observed")
    plt.plot(forecast.index, forecast, label="SARIMA forecast", color="red")
    plt.ylabel(TARGET)
    plt.title("SARIMA forecast vs observed")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sarima_forecast.png", dpi=150)

    forecast.rename("Forecast (MW)").to_csv("sarima_forecast_results.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/Koru_Wind_EnhancedData.csv")
