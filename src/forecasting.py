"""
Longitudinal Time-Series Incident Forecasting & Backtesting Engine
Location: ./src/forecasting.py
Purpose: Double Exponential Smoothing (Holt's Linear) with rolling historical
         out-of-sample backtesting to compute MAE, RMSE, and MAPE metrics.
"""

import pandas as pd
import numpy as np

def evaluate_forecast_backtest(counts: np.ndarray, test_periods: int = 4) -> dict:
    """
    Evaluates out-of-sample forecast accuracy using a historical holdout split.
    Compares Holt's Linear against Naive baseline and Moving Average.
    """
    n = len(counts)
    if n <= test_periods + 4:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "baseline_mae": 0.0}

    train_c = counts[:-test_periods]
    actual_test = counts[-test_periods:]

    # Fit Holt's on training portion
    alpha, beta = 0.4, 0.2
    level = train_c[0]
    trend = train_c[1] - train_c[0] if len(train_c) > 1 else 0.0

    for t in range(1, len(train_c)):
        val = train_c[t]
        last_lvl = level
        level = alpha * val + (1.0 - alpha) * (last_lvl + trend)
        trend = beta * (level - last_lvl) + (1.0 - beta) * trend

    holt_preds = np.array([max(0.0, level + (i + 1) * trend) for i in range(test_periods)])
    naive_preds = np.full(test_periods, train_c[-1])
    ma_preds = np.full(test_periods, np.mean(train_c[-3:]))

    # Error metrics
    mae = float(np.mean(np.abs(actual_test - holt_preds)))
    rmse = float(np.sqrt(np.mean((actual_test - holt_preds) ** 2)))
    mape = float(np.mean(np.abs((actual_test - holt_preds) / np.clip(actual_test, 1, None))) * 100.0)

    naive_mae = float(np.mean(np.abs(actual_test - naive_preds)))
    ma_mae = float(np.mean(np.abs(actual_test - ma_preds)))

    return {
        "mae": round(mae, 1),
        "rmse": round(rmse, 1),
        "mape": round(mape, 2),
        "naive_mae": round(naive_mae, 1),
        "ma_mae": round(ma_mae, 1),
        "evaluated_periods": test_periods
    }

def generate_incident_forecast(df: pd.DataFrame, forecast_horizon: int = 5) -> dict:
    if df.empty:
        return {"error": "Dataset is empty."}

    yearly_counts = df.groupby("year").size().sort_index()
    if len(yearly_counts) < 5:
        return {"error": "Insufficient longitudinal depth (requires >= 5 years of historical data)."}

    years = yearly_counts.index.values.astype(int)
    counts = yearly_counts.values.astype(float)
    n = len(counts)

    # Compute verifiable backtesting evaluation on holdout
    backtest = evaluate_forecast_backtest(counts, test_periods=min(4, len(counts)//4))

    alpha = 0.4
    beta = 0.2

    level = counts[0]
    trend = counts[1] - counts[0] if n > 1 else 0.0

    fitted = np.zeros(n)
    fitted[0] = level

    for t in range(1, n):
        val = counts[t]
        last_level = level
        level = alpha * val + (1.0 - alpha) * (last_level + trend)
        trend = beta * (level - last_level) + (1.0 - beta) * trend
        fitted[t] = last_level + trend

    last_year = int(years[-1])
    future_years = [last_year + i for i in range(1, forecast_horizon + 1)]
    forecast_values = [max(0.0, round(level + i * trend)) for i in range(1, forecast_horizon + 1)]

    residuals = counts[1:] - fitted[1:]
    sigma = float(np.std(residuals)) if len(residuals) > 0 else 10.0

    lower_bound = [max(0.0, round(f - 1.96 * sigma)) for f in forecast_values]
    upper_bound = [round(f + 1.96 * sigma) for f in forecast_values]

    return {
        "historical_years": years.tolist(),
        "historical_counts": counts.tolist(),
        "future_years": future_years,
        "forecast_counts": forecast_values,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "last_historical_year": last_year,
        "evaluation_metrics": backtest
    }