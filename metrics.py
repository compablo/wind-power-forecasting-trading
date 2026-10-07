"""Shared forecast evaluation metrics."""
import numpy as np


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    nonzero = y_true != 0  # MAPE is undefined when actual output is 0 MW
    mape = float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"RMSE (MW)": rmse, "MAE (MW)": mae, "MAPE (%)": mape, "R2": r2}
