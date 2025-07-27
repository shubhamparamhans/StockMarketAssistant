#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nse_bse_analysis.py
------------------------------------------------
End-to-end pipeline to:
  1. Download the past 15 calendar-day price history for all NSE 500 stocks
     (≈10 trading sessions) via Yahoo Finance.
  2. Identify the top N daily gainers and losers (default N = 5).
  3. Remove extreme outliers based on Z-score of daily % return.
  4. Compute a rich technical-indicator feature set.
  5. Build a grid-search framework to discover the best linear model that
     explains next-day returns.
  6. Back-test on the most recent 5 trading days and generate two-day
     forward predictions for a watch-list.

Author  : Shubham Paramhans
Created : 2025-07-27
Licence : MIT
"""

# ────────────────────────────────────────────────────────────────────────────
# 1. LIBRARIES AND GLOBAL SETTINGS
# ────────────────────────────────────────────────────────────────────────────
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf                                    # [2]
from scipy.stats import zscore
from ta import add_all_ta_features, utils as ta_utils    # [6]
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import Parallel, delayed                    # for speed

# CONFIGURABLE PARAMS
TOP_N            = 5        # gainers / losers extracted each day
Z_THRESHOLD      = 2.5      # outlier cut-off
LOOKBACK_CAL_D   = 15       # calendar days downloaded (≈10 sessions)
BACKTEST_DAYS    = 5        # recent trading days for validation
PREDICT_HORIZON  = 2        # days to forecast
N_JOBS           = -1       # use all CPU cores

# ────────────────────────────────────────────────────────────────────────────
# 2. HELPER UTILITIES
# ────────────────────────────────────────────────────────────────────────────
def get_tickers(filter_set="NSE500"):
    """
    Returns a list of NSE/BSE Yahoo tickers.
    Uses a static CSV from Kaggle (NSE tickers + .NS suffix) if available;
    otherwise queries yfinance for NIFTY-50 as a fallback.
    """
    kaggle_csv = "EQUITY_L.csv"  # expects this in working dir (optional)
    if os.path.isfile(kaggle_csv):
        df = pd.read_csv(kaggle_csv)
        symbols = (df["Yahoo_Ticker"].dropna().astype(str) + ".NS").unique().tolist()
    else:
        symbols = [s + ".NS" for s in
                   ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC", "KOTAKBANK",
                    "LT", "ICICIBANK", "HINDUNILVR", "SBIN"]]
    return symbols

def fetch_history(ticker, start, end):
    """Wrapper around yfinance download for one symbol."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:                                     # handle delisted / illiquid
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = df.columns.str.capitalize()
        df["Ticker"] = ticker
        return df
    except Exception:
        return None

def parallel_download(tickers, start, end):
    """Parallelised price download."""
    dfs = Parallel(n_jobs=N_JOBS)(
        delayed(fetch_history)(t, start, end) for t in tickers
    )
    return pd.concat([d for d in dfs if d is not None], axis=0)

def compute_daily_pct(df):
    """Add Daily_Return column (%)."""
    df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()*100
    return df

def top_movers(df, n=TOP_N):
    """Return top-N gainers and losers for each date."""
    movers = []
    for date, grp in df.groupby(df.index):
        g = grp.nlargest(n, "Daily_Return")
        l = grp.nsmallest(n, "Daily_Return")
        movers.append(pd.concat([g, l]))
    return pd.concat(movers)

def remove_outliers(df, z_thresh=Z_THRESHOLD):
    """Filter rows with extreme returns using Z-score."""
    df["Z"] = zscore(df["Daily_Return"].fillna(0))
    return df[np.abs(df["Z"]) <= z_thresh].drop(columns="Z")

def add_ta_features(df):
    """Compute 60+ technical indicators using `ta` library."""
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
    """Create feature / target frames suitable for ML."""
    # Target = NEXT-day return (%)
    filtered_df["Next_Return"] = filtered_df.groupby("Ticker")["Daily_Return"].shift(-1)

    # Drop last day (target = NaN)
    data = filtered_df.dropna(subset=["Next_Return"]).copy()

    feature_cols = [col for col in data.columns
                    if col not in ["Ticker", "Next_Return"]]
    X = data[feature_cols]
    y = data["Next_Return"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, feature_cols

# ────────────────────────────────────────────────────────────────────────────
# 3. CORE PIPELINE
# ────────────────────────────────────────────────────────────────────────────
def main():
    # ── 3.1 Download raw prices ──────────────────────────────────────────
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=LOOKBACK_CAL_D)
    tickers    = get_tickers()
    print(f"Downloading {len(tickers)} tickers from {start_date.date()} to {end_date.date()} …")

    df = parallel_download(tickers, start=start_date.strftime("%Y-%m-%d"),
                           end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"))

    if df.empty:
        print("No data retrieved. Exiting.")
        return

    df = compute_daily_pct(df)

    # ── 3.2 Extract top gainers / losers each day ────────────────────────
    movers_df       = top_movers(df)
    movers_filtered = remove_outliers(movers_df)

    # ── 3.3 Technical-indicator engineering ──────────────────────────────
    feature_df = add_ta_features(movers_filtered)

    #```
    # ── 3.4 Modelling: grid-search over linear‐model subsets ───────────────
    # We will brute-force search all k-feature subsets up to k=7
    # to find the best adjusted-R² linear model.
    X, y, scaler, feat_cols = prepare_model_data(feature_df)

    best_adj_r2, best_cols, best_model = -np.inf, None, None
    n_samples = X.shape

    def adj_r2(r2, p, n):
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    for k in range(1, min(8, len(feat_cols)+1)):
        from itertools import combinations
        for cols in combinations(range(len(feat_cols)), k):
            model = LinearRegression()
            model.fit(X[:, cols], y)
            r2 = r2_score(y, model.predict(X[:, cols]))
            adj = adj_r2(r2, k, n_samples)
            if adj > best_adj_r2:
                best_adj_r2 = adj
                best_cols   = cols
                best_model  = model

    print(f"\nBest model uses {len(best_cols)} features "
          f"with adj-R²={best_adj_r2:.4f}")

    chosen_features = [feat_cols[i] for i in best_cols]
    print("Selected predictors:", ", ".join(chosen_features))

    # ── 3.5 Back-test on last BACKTEST_DAYS ───────────────────────────────
    latest_trading_days = sorted(feature_df.index.unique())[-BACKTEST_DAYS:]
    backtest_mask = feature_df.index.isin(latest_trading_days)
    X_bt = scaler.transform(feature_df.loc[backtest_mask, feat_cols])[:, best_cols]
    y_bt = feature_df.loc[backtest_mask, "Next_Return"]
    preds_bt = best_model.predict(X_bt)

    bt_r2  = r2_score(y_bt, preds_bt)
    bt_mae = mean_absolute_error(y_bt, preds_bt)
    bt_dir = np.mean(np.sign(preds_bt) == np.sign(y_bt))

    print(f"\nBack-test ({BACKTEST_DAYS} sessions): "
          f"R²={bt_r2:.4f} | MAE={bt_mae:.2f}% | Directional Accuracy={bt_dir*100:.1f}%")

    # ── 3.6 Two-day forward prediction watch-list ─────────────────────────
    future_dates = pd.date_range(end_date + timedelta(days=1),
                                 periods=PREDICT_HORIZON, freq="B")

    watchlist = []
    for fut_day in future_dates:
        # Use latest available feature row for each ticker
        fut_rows = (feature_df.groupby("Ticker")
                               .tail(1)
                               .reset_index(drop=True))

        X_fut = scaler.transform(fut_rows[feat_cols])[:, best_cols]
        fut_rows["Pred_Return_%"] = best_model.predict(X_fut)
        fut_rows["Pred_Date"]     = fut_day.date()
        watchlist.append(fut_rows[["Pred_Date", "Ticker", "Pred_Return_%"]])

    watchlist_df = pd.concat(watchlist).sort_values("Pred_Return_%", ascending=False)

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    outdir    = "output"
    os.makedirs(outdir, exist_ok=True)

    feature_df.to_csv(f"{outdir}/feature_matrix_{timestamp}.csv")
    watchlist_df.to_csv(f"{outdir}/watchlist_{timestamp}.csv", index=False)

    print(f"\nWatch-list for next {PREDICT_HORIZON} trading days "
          f"written to {outdir}/watchlist_{timestamp}.csv")

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
