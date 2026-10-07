# Wind Power Forecasting for Renewable Energy Trading

MSc thesis project, University of Portsmouth (2024): *Enhancing Renewable Energy Trading Operations through Advanced Data Science*.

Short-term wind generation forecasts drive day-ahead bids and intraday position management. Forecast errors turn directly into imbalance costs. This project compares four approaches for forecasting output at **Koru Wind Farm (Türkiye, Vestas V112-3.3 MW turbines)** over eight test days covering different seasons:

| Model | Idea |
|---|---|
| Physical Model 1 | Wind power equation `P = ½ ρ A v³ Cp` using turbine parameters |
| Physical Model 2 | Interpolation on the manufacturer power curve |
| SARIMA | Seasonal time-series model on 3-hour aggregated output |
| XGBoost | Gradient boosting on wind speed, temperature and air density |

## Results

![R² by test period](figures/r2_by_period.png)

- XGBoost had the **lowest RMSE in 7 of 8 test periods** and the highest R² in all 8.
- Physical and SARIMA models broke down in volatile summer and early-autumn conditions (negative R²). XGBoost stayed positive because it captures the non-linear wind-speed/output relationship.
- In stable winter and spring conditions all models converged (R² of about 0.70–0.86), and XGBoost reached MAPE of 8.6–9.8%.

Full tables and discussion: [docs/MSc_Thesis_Eren_Cam.pdf](docs/MSc_Thesis_Eren_Cam.pdf)

## Trading relevance

- **Day-ahead:** better generation forecasts mean smaller gaps between nominated and delivered volumes.
- **Intraday:** a model that adapts to changing meteorology supports re-forecasting and position correction closer to delivery.
- **Imbalance cost:** lower forecast error reduces exposure to balancing market prices.

## Running the code

```bash
pip install -r requirements.txt
python src/sarima_forecast.py data/Koru_Wind_EnhancedData.csv
python src/xgboost_forecast.py data/Koru_Wind.csv
```

See [data/README.md](data/README.md) for the data sources and expected columns.

> Note: the scripts here are a cleaned-up version of the thesis appendix code. They hold out the forecast horizon for SARIMA and use a chronological train/test split for XGBoost, which avoids look-ahead bias. Re-running them can give somewhat different numbers from the thesis tables.

## Next steps

- Hybrid physical + ML models
- SCADA (turbine-level) data and NWP weather forecasts as inputs
- Translating forecast error into imbalance cost (€/MWh) using balancing market prices

## Author

**Eren Çam**, Energy Market Analyst · [LinkedIn](https://www.linkedin.com/in/eerencam/)
