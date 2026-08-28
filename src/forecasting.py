"""
Multi-Model Longitudinal Time Series Forecasting & Out-of-Sample Backtesting Engine
Location: ./src/forecasting.py
Purpose: Benchmarks Naive Persistence, Moving Average, SES, and Holt's Linear models
         via Out-of-Sample Holdout and Walk-Forward Validation, with unified forward projection.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

EPSILON = 1e-6

def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Computes robust, division-by-zero safe regression metrics."""
    err = actual - predicted
    abs_err = np.abs(err)
    
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    
    # Division-by-zero stabilized MAPE
    denom_mape = np.maximum(np.abs(actual), 1.0)
    mape = float(np.mean(abs_err / denom_mape) * 100.0)
    
    # Symmetric MAPE (sMAPE) bounded between 0% and 200%
    denom_smape = np.abs(actual) + np.abs(predicted) + EPSILON
    smape = float(np.mean(2.0 * abs_err / denom_smape) * 100.0)

    return {
        "MAE (Events)": round(mae, 1),
        "RMSE": round(rmse, 1),
        "MAPE (%)": round(mape, 2),
        "sMAPE (%)": round(smape, 2)
    }

def _fit_predict_model(model_name: str, train_c: np.ndarray, horizon: int) -> np.ndarray:
    """Fits model strictly on training portion and projects forward h steps."""
    n = len(train_c)
    if n == 0:
        return np.zeros(horizon)

    if model_name == "Naive Persistence Baseline":
        return np.full(horizon, max(0.0, train_c[-1]))

    elif model_name == "Moving Average (3-Period MA)":
        window = min(3, n)
        ma_val = float(np.mean(train_c[-window:]))
        return np.full(horizon, max(0.0, ma_val))

    elif model_name == "Simple Exponential Smoothing (SES)":
        alpha = 0.3
        level = train_c[0]
        for val in train_c[1:]:
            level = alpha * val + (1.0 - alpha) * level
        return np.full(horizon, max(0.0, level))

    elif model_name == "Holt's Linear Exponential Smoothing":
        alpha, beta = 0.4, 0.2
        level = train_c[0]
        trend = train_c[1] - train_c[0] if n > 1 else 0.0
        for val in train_c[1:]:
            last_lvl = level
            level = alpha * val + (1.0 - alpha) * (last_lvl + trend)
            trend = beta * (level - last_lvl) + (1.0 - beta) * trend
        return np.array([max(0.0, level + (i + 1) * trend) for i in range(horizon)])

    return np.full(horizon, max(0.0, train_c[-1]))

def run_forecasting_models(counts: np.ndarray, test_periods: int = 4) -> Dict[str, Any]:
    """
    Evaluates out-of-sample holdout validation without lookahead leakage.
    Returns model leaderboard, backtest prediction trajectories, and competitive deltas.
    """
    n = len(counts)
    if n <= test_periods + 3:
        return {"error": "Insufficient historical points for backtesting."}

    train_c = counts[:-test_periods]
    actual_test = counts[-test_periods:]

    model_names = [
        "Naive Persistence Baseline",
        "Moving Average (3-Period MA)",
        "Simple Exponential Smoothing (SES)",
        "Holt's Linear Exponential Smoothing"
    ]

    leaderboard = []
    backtest_predictions = {}

    for name in model_names:
        preds = _fit_predict_model(name, train_c, horizon=test_periods)
        mets = _compute_metrics(actual_test, preds)
        
        entry = {"Model": name, **mets}
        leaderboard.append(entry)
        backtest_predictions[name] = preds.tolist()

    df_leaderboard = pd.DataFrame(leaderboard).sort_values("MAE (Events)").reset_index(drop=True)
    df_leaderboard["Rank"] = [f"#{i+1}" for i in range(len(df_leaderboard))]
    
    # Reorder columns
    df_leaderboard = df_leaderboard[["Rank", "Model", "MAE (Events)", "RMSE", "MAPE (%)", "sMAPE (%)"]]

    best_model_name = df_leaderboard.iloc[0]["Model"]
    best_mae = float(df_leaderboard.iloc[0]["MAE (Events)"])
    runner_up_mae = float(df_leaderboard.iloc[1]["MAE (Events)"])
    
    mae_advantage_pct = round(((runner_up_mae - best_mae) / max(0.1, runner_up_mae)) * 100.0, 1)

    return {
        "leaderboard": df_leaderboard,
        "best_model_name": best_model_name,
        "best_mae": best_mae,
        "runner_up_mae": runner_up_mae,
        "mae_advantage_pct": mae_advantage_pct,
        "backtest_predictions": backtest_predictions,
        "actual_holdout": actual_test.tolist(),
        "holdout_periods": test_periods
    }

def run_walk_forward_validation(counts: np.ndarray, min_train: int = 15) -> pd.DataFrame:
    """
    Executes expanding-window Walk-Forward (Rolling) 1-step validation.
    Trains on t_0..t_k and evaluates out-of-sample on t_{k+1}.
    """
    n = len(counts)
    if n <= min_train + 3:
        return pd.DataFrame()

    model_names = [
        "Naive Persistence Baseline",
        "Moving Average (3-Period MA)",
        "Simple Exponential Smoothing (SES)",
        "Holt's Linear Exponential Smoothing"
    ]

    wf_results = {name: [] for name in model_names}
    actuals = []

    for t in range(min_train, n):
        train_split = counts[:t]
        actual_val = counts[t]
        actuals.append(actual_val)

        for name in model_names:
            p = _fit_predict_model(name, train_split, horizon=1)[0]
            wf_results[name].append(p)

    actuals_arr = np.array(actuals)
    summary = []
    for name in model_names:
        preds_arr = np.array(wf_results[name])
        mets = _compute_metrics(actuals_arr, preds_arr)
        summary.append({"Model": name, **mets})

    df_wf = pd.DataFrame(summary).sort_values("MAE (Events)").reset_index(drop=True)
    df_wf["Rank"] = [f"#{i+1}" for i in range(len(df_wf))]
    return df_wf[["Rank", "Model", "MAE (Events)", "RMSE", "MAPE (%)", "sMAPE (%)"]]

def generate_incident_forecast(df: pd.DataFrame, forecast_horizon: int = 5, forced_model: str = None) -> Dict[str, Any]:
    """
    Generates time-series projections dynamically using the optimal backtested model
    (or user-selected benchmark), computing empirical residual standard errors for analytical bounds.
    """
    if df.empty:
        return {"error": "Dataset is empty."}

    yearly_counts = df.groupby("year").size().sort_index()
    if len(yearly_counts) < 5:
        return {"error": "Requires >= 5 years of historical records."}

    years = yearly_counts.index.values.astype(int)
    counts = yearly_counts.values.astype(float)
    n = len(counts)

    # 1. Evaluate Backtesting
    backtest_res = run_forecasting_models(counts, test_periods=min(4, max(2, len(counts) // 5)))
    if "error" in backtest_res:
        return backtest_res

    # 2. Determine Selected Model for Forward Projection
    active_model = forced_model if forced_model else backtest_res["best_model_name"]

    # 3. Fit on full available dataset to extrapolate future horizon
    forecast_values = _fit_predict_model(active_model, counts, horizon=forecast_horizon)
    forecast_values = [max(0.0, round(float(v))) for v in forecast_values]

    # 4. In-Sample Residual Variance for Empirical Analytical Uncertainty Bounds
    fitted_in_sample = []
    for t in range(1, n):
        p = _fit_predict_model(active_model, counts[:t], horizon=1)[0]
        fitted_in_sample.append(p)
    
    residuals = counts[1:] - np.array(fitted_in_sample)
    sigma = float(np.std(residuals)) if len(residuals) > 0 else 10.0
    res_mean = float(np.mean(residuals)) if len(residuals) > 0 else 0.0

    last_year = int(years[-1])
    future_years = [last_year + i for i in range(1, forecast_horizon + 1)]

    lower_bound = [max(0.0, round(f - 1.96 * sigma)) for f in forecast_values]
    upper_bound = [round(f + 1.96 * sigma) for f in forecast_values]

    # Direction & Delta Analysis
    last_observed_val = float(counts[-1])
    final_forecast_val = float(forecast_values[-1])
    abs_delta = round(final_forecast_val - last_observed_val, 1)
    pct_delta = round((abs_delta / max(1.0, last_observed_val)) * 100.0, 1)

    if abs_delta > (last_observed_val * 0.05):
        direction = "Increasing / Escalating"
        dir_color = "#F85149"
    elif abs_delta < -(last_observed_val * 0.05):
        direction = "Declining / De-escalating"
        dir_color = "#39D353"
    else:
        direction = "Stable / Range-Bound"
        dir_color = "#58A6FF"

    return {
        "historical_years": years.tolist(),
        "historical_counts": counts.tolist(),
        "future_years": future_years,
        "forecast_counts": forecast_values,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "last_historical_year": last_year,
        "last_observed_value": last_observed_val,
        "final_forecast_value": final_forecast_val,
        "absolute_change": abs_delta,
        "percentage_change": pct_delta,
        "forecast_direction": direction,
        "direction_color": dir_color,
        "active_model_name": active_model,
        "is_optimal_model": (active_model == backtest_res["best_model_name"]),
        "backtest_results": backtest_res,
        "residual_mean": round(res_mean, 2),
        "residual_std": round(sigma, 2),
        "residuals": residuals.tolist()
    }