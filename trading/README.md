# AI Regime Terminal – HMM Trading System

A probabilistic market-regime engine built on a **7-state Gaussian Hidden Markov
Model**, layered with a "Selective Moonshot" entry strategy and institutional-grade
risk management.

## Architecture

```
trading/
├── data_loader.py   # Fetch BTC-USD 1 h OHLCV + feature engineering
├── backtester.py    # HMM engine, signal generator, backtester, metrics
├── app.py           # Streamlit dashboard (Plotly charts)
└── requirements.txt
```

## Quick Start

```bash
pip install -r trading/requirements.txt
streamlit run trading/app.py
```

## How It Works

### 1. The HMM Engine (`HMMEngine`)
- Trains a `GaussianHMM` with **7 components** on 730 days of hourly data.
- Features: **Log Returns**, **Normalised Range** `(H-L)/C`, **Volume Volatility** (z-score).
- States are sorted by mean return → deterministic labels from *Bear/Crash* → *Bull Run*.

### 2. The Strategy (`StrategySignal`)
Entry requires **HMM = Bull Run** AND **≥ 7 of 8** confirmations:

| # | Confirmation      | Threshold            |
|---|-------------------|----------------------|
| 1 | RSI               | < 90                 |
| 2 | Momentum          | > 1 % (12-bar)       |
| 3 | Volatility        | < 6 % (annualised)   |
| 4 | Volume            | > 20-bar SMA         |
| 5 | ADX               | > 25                 |
| 6 | Price vs EMA 50   | Price > EMA 50       |
| 7 | Price vs EMA 200  | Price > EMA 200      |
| 8 | MACD              | MACD > Signal line   |

### 3. Risk Management (`Backtester`)
- **Exit trigger**: Regime flips to *Bear/Crash*.
- **Cooldown**: Hard 48-hour no-entry window after every exit.
- **Leverage**: 2.5× simulated on PnL.

### 4. Dashboard Tabs
| Tab | Content |
|-----|---------|
| Price & Regimes | Candlestick + coloured regime backgrounds, EMA 50/200, trade markers, RSI, MACD |
| Equity Curve | Strategy vs Buy-&-Hold, drawdown overlay, summary stats |
| Regime Analysis | Time allocation pie, avg return per regime, vote histogram, transition matrix |
| Trade Log | Full trade table, per-trade PnL waterfall |

---
> **Disclaimer**: Educational and research purposes only. Not financial advice.
> Past simulated performance is not indicative of future results.
