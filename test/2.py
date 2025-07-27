#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Multi-Model Indian Stock Market Analysis

Features:
- Downloads historical stock data for leading NSE stocks
- Computes top gainers/losers per day and filters with Z-score outlier detection
- Calculates technical indicators using 'ta'
- Fits multiple models: LinearRegression, Ridge, Lasso, RandomForest, XGBoost, SVR, GradientBoosting
- Selects best model by validation adj-R²
- Trains a meta-LR (stacking) on all base model outputs for "hybrid" predictions
- Prints out equations for linear/stacking regressors for full transparency

Dependencies:
  pip install yfinance ta scikit-learn joblib xgboost pandas numpy scipy
"""

import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import zscore
from ta import add_all_ta_features, utils as ta_utils
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.svm import SVR
from joblib import Parallel, delayed

# --- Add if using XGBoost
try:
    from xgboost import XGBRegressor
    has_xgb = True
except ImportError:
    has_xgb = False

# CONFIGURABLE PARAMETERS
TOP_N = 5
Z_THRESHOLD = 2.5
LOOKBACK_CAL_D = 15
BACKTEST_DAYS = 5
PREDICT_HORIZON = 2
N_JOBS = -1

def get_tickers():
    # Replace/extend as needed for your universe
    return [t + ".NS" for t in [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC", "KOTAKBANK",
        "LT", "ICICIBANK", "HINDUNILVR", "SBIN", "BAJFINANCE", "MARUTI",
        "BHARTIARTL", "ULTRACEMCO", "NESTLEIND", "ONGC"
    ]]

def fetch_history(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = df.columns.str.capitalize()
        df["Ticker"] = ticker
        return df
    except Exception:
        return None

def parallel_download(tickers, start, end):
    dfs = Parallel(n_jobs=N_JOBS)(
        delayed(fetch_history)(t, start, end) for t in tickers
    )
    return pd.concat([d for d in dfs if d is not None], axis=0)

def compute_daily_pct(df):
    df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()*100
    return df

def top_movers(df, n=TOP_N):
    movers = []
    for date, grp in df.groupby(df.index):
        g = grp.nlargest(n, "Daily_Return")
        l = grp.nsmallest(n, "Daily_Return")
        movers.append(pd.concat([g, l]))
    return pd.concat(movers)

def remove_outliers(df, z_thresh=Z_THRESHOLD):
    df["Z"] = zscore(df["Daily_Return"].fillna(0))
    return df[np.abs(df["Z"]) <= z_thresh].drop(columns="Z")

def add_ta_features(df):
    df_copy = df.copy()
    df_copy.reset_index(inplace=True)
    df_copy = ta_utils.dropna(df_copy)
    df_copy = add_all_ta_features(
        df_copy,
        open="Open", high="High", low="Low", close="Close",
        volume="Volume", fillna=True
    )
    df_copy.set_index("Date", inplace=True)
    return df_copy

def prepare_model_data(filtered_df):
    filtered_df["Next_Return"] = filtered_df.groupby("Ticker")["Daily_Return"].shift(-1)
    data = filtered_df.dropna(subset=["Next_Return"]).copy()
    feature_cols = [col for col in data.columns if col not in ["Ticker", "Next_Return"]]
    X = data[feature_cols]
    y = data["Next_Return"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, feature_cols, data

def adj_r2(r2, p, n):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def print_equation(lr, feature_names, title="Mathematical prediction equation"):
    equation = f"{title}:\nDaily_Return = {lr.intercept_:.4f} "
    for coef, col in zip(lr.coef_, feature_names):
        equation += f"+ ({coef:.4f} × {col}) "
    print(equation)

def main():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=LOOKBACK_CAL_D)
    tickers = get_tickers()
    print(f"Downloading {len(tickers)} tickers from {start_date.date()} to {end_date.date()} …")
    df = parallel_download(
        tickers, start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d")
    )
    if df.empty:
        print("No data retrieved. Exiting.")
        return
    df = compute_daily_pct(df)
    movers_df = top_movers(df)
    movers_filtered = remove_outliers(movers_df)
    feature_df = add_ta_features(movers_filtered)
    X, y, scaler, feat_cols, train_df = prepare_model_data(feature_df)
    # --- Split for validation (rolling last BACKTEST_DAYS out)
    split_n = X.shape[0] - BACKTEST_DAYS * len(tickers) # Approx
    X_train, y_train = X[:split_n], y[:split_n]
    X_valid, y_valid = X[split_n:], y[split_n:]

    MODELS = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "RandomForest": RandomForestRegressor(n_estimators=80, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=80, random_state=42),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.2)
    }
    if has_xgb:
        MODELS["XGBoost"] = XGBRegressor(n_estimators=80, random_state=42, verbosity=0)

    preds_valid_all = []
    all_model_names = []
    best_r2 = -np.inf
    best_model = None
    best_name = None
    best_feat = feat_cols

    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        predv = model.predict(X_valid)
        r2 = r2_score(y_valid, predv)
        preds_valid_all.append(predv.reshape(-1,1))
        all_model_names.append(name)
        print(f"{name} validation R²: {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    preds_stack = np.hstack(preds_valid_all) # shape: (samples, n_models)
    # Train meta-model: Linear Regression on above model predictions
    meta_lr = LinearRegression()
    meta_lr.fit(preds_stack, y_valid)
    print("\nStacking blend weights (meta-model):")
    for w, n in zip(meta_lr.coef_, all_model_names):
        print(f"{n}: {w:.3f}")
    print(f"Meta-model intercept: {meta_lr.intercept_:.3f}")

    print("\n----\nInterpretable Equations")
    # Print mathematical equation for best linear regressor (if chosen)
    if isinstance(MODELS["LinearRegression"], LinearRegression):
        lr = MODELS["LinearRegression"]
        lr.fit(X_train, y_train)
        print_equation(lr, feat_cols, title="Best Linear Regression formula")
    # Print meta-equation for stack
    meta_eq = "Stacked Hybrid Equation:\nDaily_Return = "
    meta_eq += f"{meta_lr.intercept_:.4f} "
    for w, n in zip(meta_lr.coef_, all_model_names):
        meta_eq += f"+ ({w:.4f} × {n}_prediction) "
    print(meta_eq)

    # --- Predict for next PREDICT_HORIZON sessions
    fut_rows = (feature_df.groupby("Ticker").tail(1).reset_index(drop=True))
    Xf = scaler.transform(fut_rows[feat_cols])
    base_preds = []
    for name, model in MODELS.items():
        base_preds.append(model.predict(Xf).reshape(-1,1))
    base_preds = np.hstack(base_preds)
    stacked_pred = meta_lr.predict(base_preds)
    fut_rows["Hybrid_Predicted_Return"] = stacked_pred
    for i, n in enumerate(all_model_names):
        fut_rows[f"{n}_Return"] = base_preds[:,i]
    fut_rows[["Ticker","Hybrid_Predicted_Return"] + [f"{n}_Return" for n in all_model_names]].to_csv(
        "output/hybrid_predictions.csv", index=False)
    print("\nTop gainers (hybrid prediction):")
    print(fut_rows[["Ticker", "Hybrid_Predicted_Return"]].sort_values(
        "Hybrid_Predicted_Return", ascending=False).head(7))

if __name__ == "__main__":
    if not os.path.isdir("output"):
        os.makedirs("output")
    main()
