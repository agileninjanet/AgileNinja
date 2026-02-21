"""
data_loader.py
==============
Fetches BTC-USD hourly OHLCV data for the last 730 days from Yahoo Finance.
Returns a clean, feature-engineered DataFrame ready for HMM training and
backtesting.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

TICKER = "BTC-USD"
PERIOD_DAYS = 730
INTERVAL = "1h"


def fetch_data(ticker: str = TICKER, days: int = PERIOD_DAYS) -> pd.DataFrame:
    """Download hourly OHLCV data and engineer all required features."""
    # yfinance max for 1h is 730 days – use period string
    raw = yf.download(
        ticker,
        period=f"{days}d",
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check your connection.")

    # Flatten multi-level columns produced by yfinance ≥ 0.2.x
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    # ── Feature Engineering ──────────────────────────────────────────────────

    # 1. Log Returns (primary HMM signal)
    df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))

    # 2. Normalised range: (High – Low) / Close  (proxy for intra-bar volatility)
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]

    # 3. Volume Volatility: rolling 24-bar z-score of volume
    vol_ma = df["Volume"].rolling(24).mean()
    vol_std = df["Volume"].rolling(24).std()
    df["VolVol"] = (df["Volume"] - vol_ma) / (vol_std + 1e-9)

    # ── Technical Indicators (needed by the strategy / backtester) ───────────

    # RSI (14)
    df["RSI"] = _rsi(df["Close"], 14)

    # Momentum: (Close / Close[N] – 1)  with N = 12 bars (~12 h)
    df["Momentum"] = df["Close"].pct_change(12) * 100  # in percent

    # Rolling volatility: annualised std of 24-bar log returns (in percent)
    df["Volatility"] = df["Returns"].rolling(24).std() * np.sqrt(24 * 365) * 100

    # Volume SMA (20)
    df["VolSMA20"] = df["Volume"].rolling(20).mean()

    # ADX (14)
    df["ADX"] = _adx(df, 14)

    # EMA 50 and EMA 200
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # MACD and Signal line
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)

    # Remove timezone info for Streamlit/Plotly compatibility
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


# ── Private helpers ──────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), high - high.shift(1), 0)
    dm_plus = np.where(dm_plus < 0, 0, dm_plus)
    dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), low.shift(1) - low, 0)
    dm_minus = np.where(dm_minus < 0, 0, dm_minus)

    dm_plus_s = pd.Series(dm_plus, index=df.index)
    dm_minus_s = pd.Series(dm_minus, index=df.index)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    di_plus = 100 * dm_plus_s.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-9)
    di_minus = 100 * dm_minus_s.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-9)

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    return dx.ewm(alpha=1 / period, adjust=False).mean()
