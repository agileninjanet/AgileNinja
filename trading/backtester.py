"""
backtester.py
=============
Hidden Markov Model engine + Selective Moonshot strategy + backtester.

Architecture
------------
  HMMEngine        – fits a 7-state GaussianHMM and labels regimes.
  StrategySignal   – generates entry/exit signals from the HMM + 8 confirmations.
  Backtester       – simulates trades, enforces cooldown & leverage, records metrics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from hmmlearn import hmm

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
N_STATES = 7
LEVERAGE = 2.5
COOLDOWN_HOURS = 48
MIN_CONFIRMATIONS = 7  # out of 8

# Regime label names (assigned after HMM training)
LABEL_BULL = "Bull Run"
LABEL_BEAR = "Bear / Crash"
LABEL_CHOP = "Choppy Noise"
LABEL_NAMES = [
    "Bull Run",
    "Mild Bull",
    "Neutral",
    "Low-Vol Drift",
    "Choppy Noise",
    "Mild Bear",
    "Bear / Crash",
]


# ── HMM Engine ───────────────────────────────────────────────────────────────

class HMMEngine:
    """
    Fits a Gaussian HMM on three features:
      - Log Returns
      - Normalised Range (High-Low)/Close
      - Volume Volatility (z-score)

    After fitting, states are sorted by their mean Return so we can
    deterministically map state index → regime label.
    """

    def __init__(self, n_states: int = N_STATES, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self.state_order: List[int] = []   # sorted state indices (low→high return)
        self.bull_state: int = -1
        self.bear_state: int = -1

    def fit(self, df: pd.DataFrame) -> "HMMEngine":
        features = df[["Returns", "Range", "VolVol"]].values
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            tol=1e-4,
            random_state=self.random_state,
        )
        self.model.fit(features)

        # Sort states by mean Return (ascending) to get deterministic labels
        mean_returns = self.model.means_[:, 0]          # column 0 = Returns
        self.state_order = list(np.argsort(mean_returns))  # lowest → highest

        # Bear = first (lowest return), Bull = last (highest return)
        self.bear_state = self.state_order[0]
        self.bull_state = self.state_order[-1]
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw HMM state indices for each row in df."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        features = df[["Returns", "Range", "VolVol"]].values
        return self.model.predict(features)

    def label_state(self, state_idx: int) -> str:
        """Map a raw HMM state index to a human-readable regime label."""
        rank = self.state_order.index(state_idx)   # 0 = lowest return
        return LABEL_NAMES[rank]

    def is_bullish(self, state_idx: int) -> bool:
        return state_idx == self.bull_state

    def is_bearish(self, state_idx: int) -> bool:
        return state_idx == self.bear_state


# ── Strategy Signal Generator ────────────────────────────────────────────────

@dataclass
class Confirmation:
    name: str
    met: bool


def _check_confirmations(row: pd.Series) -> List[Confirmation]:
    """Evaluate all 8 technical confirmations for a single bar."""
    return [
        Confirmation("RSI < 90",           bool(row["RSI"] < 90)),
        Confirmation("Momentum > 1%",       bool(row["Momentum"] > 1.0)),
        Confirmation("Volatility < 6%",     bool(row["Volatility"] < 6.0)),
        Confirmation("Volume > SMA20",      bool(row["Volume"] > row["VolSMA20"])),
        Confirmation("ADX > 25",            bool(row["ADX"] > 25)),
        Confirmation("Price > EMA50",       bool(row["Close"] > row["EMA50"])),
        Confirmation("Price > EMA200",      bool(row["Close"] > row["EMA200"])),
        Confirmation("MACD > Signal",       bool(row["MACD"] > row["MACDSignal"])),
    ]


def generate_signals(df: pd.DataFrame, engine: HMMEngine) -> pd.DataFrame:
    """
    Add regime labels and entry/exit signals to the DataFrame.

    New columns
    -----------
    State       : raw HMM state integer
    Regime      : human-readable regime label
    IsBull      : True when HMM is in Bull Run state
    IsBear      : True when HMM is in Bear/Crash state
    Votes       : number of confirmations met (0–8)
    EntrySignal : True when all strategy conditions are satisfied
    ExitSignal  : True when regime flips to Bear/Crash
    """
    out = df.copy()
    states = engine.predict(out)
    out["State"] = states
    out["Regime"] = [engine.label_state(s) for s in states]
    out["IsBull"] = [engine.is_bullish(s) for s in states]
    out["IsBear"] = [engine.is_bearish(s) for s in states]

    votes = []
    for _, row in out.iterrows():
        confs = _check_confirmations(row)
        votes.append(sum(c.met for c in confs))

    out["Votes"] = votes
    out["EntrySignal"] = out["IsBull"] & (out["Votes"] >= MIN_CONFIRMATIONS)
    out["ExitSignal"] = out["IsBear"]

    return out


# ── Trade Record ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_pct: float = 0.0          # leveraged PnL %
    pnl_dollar: float = 0.0       # leveraged PnL $ (per $1 invested)


# ── Backtester ───────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)   # signal df

    # Summary metrics (populated by Backtester.run)
    total_return_pct: float = 0.0
    win_rate_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    num_trades: int = 0
    avg_trade_pct: float = 0.0


class Backtester:
    """
    Simulates the Selective Moonshot strategy bar-by-bar.

    Rules
    -----
    • Entry  : EntrySignal is True and we are not in cooldown.
    • Exit   : ExitSignal (regime → Bear) OR end-of-data.
    • Cooldown: 48 hours of no-entry after any exit.
    • Leverage: PnL is multiplied by LEVERAGE (2.5×).
    """

    def __init__(self, leverage: float = LEVERAGE, cooldown_hours: int = COOLDOWN_HOURS):
        self.leverage = leverage
        self.cooldown_hours = cooldown_hours

    def run(self, df_signals: pd.DataFrame) -> BacktestResult:
        result = BacktestResult(df=df_signals)
        trades: List[Trade] = []

        in_trade = False
        entry_price = 0.0
        entry_time: pd.Timestamp = pd.Timestamp("1970-01-01")
        cooldown_until: pd.Timestamp = pd.Timestamp("1970-01-01")

        equity = 1.0          # start with $1 (normalised)
        equity_curve: List[float] = []
        equity_times: List[pd.Timestamp] = []

        for ts, row in df_signals.iterrows():
            ts = pd.Timestamp(ts)

            # ── Exit logic (checked first) ───────────────────────────────────
            if in_trade and row["ExitSignal"]:
                exit_price = float(row["Close"])
                raw_ret = (exit_price - entry_price) / entry_price
                leveraged_ret = raw_ret * self.leverage
                equity *= (1 + leveraged_ret)

                trade = Trade(
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=ts,
                    exit_price=exit_price,
                    exit_reason="Regime → Bear/Crash",
                    pnl_pct=leveraged_ret * 100,
                    pnl_dollar=leveraged_ret,
                )
                trades.append(trade)
                in_trade = False
                cooldown_until = ts + pd.Timedelta(hours=self.cooldown_hours)

            # ── Entry logic ──────────────────────────────────────────────────
            elif not in_trade and row["EntrySignal"] and ts >= cooldown_until:
                in_trade = True
                entry_price = float(row["Close"])
                entry_time = ts

            # Track equity while in trade (mark-to-market)
            if in_trade:
                current_price = float(row["Close"])
                mtm_ret = (current_price - entry_price) / entry_price * self.leverage
                equity_curve.append(equity * (1 + mtm_ret))
            else:
                equity_curve.append(equity)

            equity_times.append(ts)

        # Close any open position at end of data
        if in_trade:
            last_row = df_signals.iloc[-1]
            last_ts = df_signals.index[-1]
            exit_price = float(last_row["Close"])
            raw_ret = (exit_price - entry_price) / entry_price
            leveraged_ret = raw_ret * self.leverage
            equity *= (1 + leveraged_ret)
            trade = Trade(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=pd.Timestamp(last_ts),
                exit_price=exit_price,
                exit_reason="End of Data",
                pnl_pct=leveraged_ret * 100,
                pnl_dollar=leveraged_ret,
            )
            trades.append(trade)

        # ── Build equity series & compute metrics ────────────────────────────
        eq_series = pd.Series(equity_curve, index=equity_times)
        result.equity_curve = eq_series
        result.trades = trades
        result.num_trades = len(trades)

        if trades:
            pnls = [t.pnl_pct for t in trades]
            wins = [p for p in pnls if p > 0]
            result.total_return_pct = (eq_series.iloc[-1] - 1.0) * 100
            result.win_rate_pct = len(wins) / len(pnls) * 100
            result.avg_trade_pct = float(np.mean(pnls))
            result.max_drawdown_pct = _max_drawdown(eq_series) * 100
            result.sharpe_ratio = _sharpe(eq_series)

        return result


# ── Metric helpers ────────────────────────────────────────────────────────────

def _max_drawdown(eq: pd.Series) -> float:
    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max
    return float(drawdown.min())   # negative number


def _sharpe(eq: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe on hourly equity log-returns."""
    log_ret = np.log(eq / eq.shift(1)).dropna()
    if log_ret.std() == 0:
        return 0.0
    return float((log_ret.mean() - risk_free) / log_ret.std() * np.sqrt(24 * 365))


# ── Convenience entry-point ───────────────────────────────────────────────────

def run_full_backtest(df: pd.DataFrame) -> tuple[BacktestResult, HMMEngine]:
    """
    Fit HMM, generate signals, and run backtest in one call.
    Returns (result, engine) so the dashboard can reuse the engine.
    """
    engine = HMMEngine(n_states=N_STATES).fit(df)
    df_signals = generate_signals(df, engine)
    result = Backtester().run(df_signals)
    return result, engine
