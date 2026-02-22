"""
app.py
======
Streamlit dashboard for the AI Regime Terminal (HMM Trading System).

Run with:
    streamlit run trading/app.py
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_loader import fetch_data
from backtester import (
    LABEL_BEAR,
    LABEL_BULL,
    LABEL_NAMES,
    BacktestResult,
    HMMEngine,
    Trade,
    run_full_backtest,
)

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Regime Terminal – HMM Trading System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette per regime ─────────────────────────────────────────────────
REGIME_COLORS = {
    "Bull Run":       "#00e676",   # neon green
    "Mild Bull":      "#69f0ae",   # lighter green
    "Neutral":        "#b0bec5",   # grey-blue
    "Low-Vol Drift":  "#80deea",   # teal
    "Choppy Noise":   "#ffe082",   # amber
    "Mild Bear":      "#ff8a65",   # salmon
    "Bear / Crash":   "#ff1744",   # red
}

REGIME_ALPHA = {k: "rgba" + c[3:].replace(")", ",0.18)") if c.startswith("rgb") else c
                for k, c in REGIME_COLORS.items()}


def _hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Caching ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def load_and_run() -> tuple[pd.DataFrame, BacktestResult, HMMEngine]:
    df = fetch_data()
    result, engine = run_full_backtest(df)
    return df, result, engine


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_pct(val: float, decimals: int = 2) -> str:
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.{decimals}f}%"


def fmt_ratio(val: float) -> str:
    return f"{val:.2f}"


def _regime_band_shapes(df_signals: pd.DataFrame) -> list[dict]:
    """
    Build Plotly layout shapes that colour the background by regime.
    Consecutive identical regimes are merged into a single band.
    """
    shapes = []
    regimes = df_signals["Regime"].values
    times = df_signals.index

    start = 0
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1] or i == len(regimes) - 1:
            regime = regimes[start]
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=str(times[start]),
                    x1=str(times[i]),
                    y0=0,
                    y1=1,
                    fillcolor=_hex_to_rgba(REGIME_COLORS.get(regime, "#888888")),
                    line_width=0,
                    layer="below",
                )
            )
            start = i
    return shapes


# ── Main layout ───────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📡 AI Regime Terminal")
        st.markdown("**Model:** 7-State Gaussian HMM")
        st.markdown("**Asset:** BTC-USD (Hourly)")
        st.markdown("**Leverage:** 2.5×")
        st.markdown("**Cooldown:** 48 h post-exit")
        st.markdown("---")
        st.markdown("### Regime Legend")
        for name in LABEL_NAMES:
            color = REGIME_COLORS.get(name, "#888")
            st.markdown(
                f'<span style="color:{color}; font-size:18px;">■</span> {name}',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        refresh = st.button("🔄 Refresh Data & Rerun")
        if refresh:
            st.cache_data.clear()
            st.rerun()

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Fetching BTC-USD data and training HMM…"):
        try:
            df, result, engine = load_and_run()
        except Exception as exc:
            st.error(f"Data load failed: {exc}")
            st.stop()

    df_signals = result.df

    # ── Hero metrics row ──────────────────────────────────────────────────────
    st.markdown("# 📡 AI Regime Terminal")
    st.markdown(
        "_Hidden Markov Model | 7 Regimes | Selective Moonshot Strategy | 2.5× Leverage_"
    )
    st.markdown("---")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    metric_style = "font-size:28px; font-weight:700;"

    def _colored(val: float, text: str) -> str:
        color = "#00e676" if val >= 0 else "#ff1744"
        return f'<span style="color:{color}; {metric_style}">{text}</span>'

    col1.markdown("**Total Return**")
    col1.markdown(_colored(result.total_return_pct, fmt_pct(result.total_return_pct)), unsafe_allow_html=True)

    col2.markdown("**Win Rate**")
    col2.markdown(f'<span style="{metric_style}">{fmt_pct(result.win_rate_pct)}</span>', unsafe_allow_html=True)

    col3.markdown("**Max Drawdown**")
    col3.markdown(_colored(result.max_drawdown_pct, fmt_pct(result.max_drawdown_pct)), unsafe_allow_html=True)

    col4.markdown("**Sharpe Ratio**")
    col4.markdown(f'<span style="{metric_style}">{fmt_ratio(result.sharpe_ratio)}</span>', unsafe_allow_html=True)

    col5.markdown("**Total Trades**")
    col5.markdown(f'<span style="{metric_style}">{result.num_trades}</span>', unsafe_allow_html=True)

    col6.markdown("**Avg Trade**")
    col6.markdown(_colored(result.avg_trade_pct, fmt_pct(result.avg_trade_pct)), unsafe_allow_html=True)

    st.markdown("---")

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Price & Regimes", "📊 Equity Curve", "🔬 Regime Analysis", "📋 Trade Log"]
    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 – Price chart with coloured regime backgrounds
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("BTC-USD Price with HMM Regime Overlays")

        # Subsample for performance: show last 90 days on load, allow zoom
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.60, 0.20, 0.20],
            vertical_spacing=0.03,
            subplot_titles=("BTC-USD Price", "RSI (14)", "MACD"),
        )

        # Price candlestick
        fig.add_trace(
            go.Candlestick(
                x=df_signals.index,
                open=df_signals["Open"],
                high=df_signals["High"],
                low=df_signals["Low"],
                close=df_signals["Close"],
                name="BTC-USD",
                increasing_line_color="#00e676",
                decreasing_line_color="#ff1744",
                increasing_fillcolor="#00e676",
                decreasing_fillcolor="#ff1744",
            ),
            row=1, col=1,
        )

        # EMA 50 & EMA 200
        fig.add_trace(
            go.Scatter(x=df_signals.index, y=df_signals["EMA50"],
                       name="EMA 50", line=dict(color="#ffab40", width=1)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df_signals.index, y=df_signals["EMA200"],
                       name="EMA 200", line=dict(color="#40c4ff", width=1.5, dash="dot")),
            row=1, col=1,
        )

        # Trade markers
        for trade in result.trades:
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time],
                    y=[trade.entry_price],
                    mode="markers",
                    marker=dict(symbol="triangle-up", color="#00e676", size=12),
                    name="Entry",
                    showlegend=False,
                ),
                row=1, col=1,
            )
            if trade.exit_time:
                color = "#00e676" if trade.pnl_pct >= 0 else "#ff1744"
                fig.add_trace(
                    go.Scatter(
                        x=[trade.exit_time],
                        y=[trade.exit_price],
                        mode="markers",
                        marker=dict(symbol="triangle-down", color=color, size=12),
                        name="Exit",
                        showlegend=False,
                    ),
                    row=1, col=1,
                )

        # RSI
        fig.add_trace(
            go.Scatter(x=df_signals.index, y=df_signals["RSI"],
                       name="RSI", line=dict(color="#ce93d8", width=1)),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ff8a65", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#69f0ae", line_width=1, row=2, col=1)

        # MACD
        macd_colors = ["#00e676" if v >= 0 else "#ff1744"
                       for v in (df_signals["MACD"] - df_signals["MACDSignal"])]
        fig.add_trace(
            go.Bar(x=df_signals.index,
                   y=df_signals["MACD"] - df_signals["MACDSignal"],
                   name="MACD Hist", marker_color=macd_colors, opacity=0.7),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df_signals.index, y=df_signals["MACD"],
                       name="MACD", line=dict(color="#40c4ff", width=1)),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df_signals.index, y=df_signals["MACDSignal"],
                       name="Signal", line=dict(color="#ffab40", width=1)),
            row=3, col=1,
        )

        # Regime background shapes
        shapes = _regime_band_shapes(df_signals)
        fig.update_layout(
            shapes=shapes,
            height=750,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#e0e0e0"),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.02, x=0),
            margin=dict(l=60, r=20, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Regime colour key below chart
        cols = st.columns(len(LABEL_NAMES))
        for i, name in enumerate(LABEL_NAMES):
            color = REGIME_COLORS[name]
            cols[i].markdown(
                f'<div style="background:{color}22; border-left:4px solid {color}; '
                f'padding:6px 10px; border-radius:4px; font-size:12px;">{name}</div>',
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 – Equity Curve
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Strategy Equity Curve (2.5× Leverage)")

        eq = result.equity_curve
        buy_hold = df_signals["Close"] / df_signals["Close"].iloc[0]

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=eq.index, y=(eq - 1) * 100,
                name="HMM Strategy",
                line=dict(color="#00e676", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,230,118,0.08)",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=buy_hold.index, y=(buy_hold - 1) * 100,
                name="Buy & Hold BTC",
                line=dict(color="#40c4ff", width=1.5, dash="dot"),
            )
        )

        # Drawdown fill
        roll_max = eq.cummax()
        drawdown = (eq - roll_max) / roll_max * 100
        fig2.add_trace(
            go.Scatter(
                x=drawdown.index, y=drawdown,
                name="Drawdown",
                line=dict(color="#ff1744", width=1),
                fill="tozeroy",
                fillcolor="rgba(255,23,68,0.12)",
                yaxis="y2",
            )
        )

        fig2.update_layout(
            height=520,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#e0e0e0"),
            yaxis=dict(title="Return %"),
            yaxis2=dict(title="Drawdown %", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.02, x=0),
            margin=dict(l=60, r=60, t=60, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Stats table
        stats = {
            "Total Return": fmt_pct(result.total_return_pct),
            "Buy & Hold Return": fmt_pct((buy_hold.iloc[-1] - 1) * 100),
            "Win Rate": fmt_pct(result.win_rate_pct),
            "Max Drawdown": fmt_pct(result.max_drawdown_pct),
            "Sharpe Ratio": fmt_ratio(result.sharpe_ratio),
            "Total Trades": str(result.num_trades),
            "Avg Trade": fmt_pct(result.avg_trade_pct),
            "Leverage": "2.5×",
        }
        stat_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
        st.dataframe(stat_df, hide_index=True, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 – Regime Analysis
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("HMM Regime Distribution & Statistics")

        col_a, col_b = st.columns(2)

        # Pie chart – time spent in each regime
        regime_counts = df_signals["Regime"].value_counts()
        pie_fig = go.Figure(
            go.Pie(
                labels=regime_counts.index,
                values=regime_counts.values,
                marker_colors=[REGIME_COLORS.get(r, "#888") for r in regime_counts.index],
                hole=0.4,
                textinfo="label+percent",
            )
        )
        pie_fig.update_layout(
            title="Time Allocation by Regime",
            height=400,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            font=dict(color="#e0e0e0"),
            showlegend=False,
        )
        col_a.plotly_chart(pie_fig, use_container_width=True)

        # Bar chart – avg hourly return per regime
        regime_ret = (
            df_signals.groupby("Regime")["Returns"]
            .mean()
            .reindex(LABEL_NAMES)
            .dropna() * 100
        )
        bar_colors = [REGIME_COLORS.get(r, "#888") for r in regime_ret.index]
        bar_fig = go.Figure(
            go.Bar(
                x=regime_ret.index,
                y=regime_ret.values,
                marker_color=bar_colors,
                text=[fmt_pct(v) for v in regime_ret.values],
                textposition="outside",
            )
        )
        bar_fig.update_layout(
            title="Avg Hourly Return per Regime",
            height=400,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            font=dict(color="#e0e0e0"),
            yaxis_title="Avg Return %",
            showlegend=False,
        )
        col_b.plotly_chart(bar_fig, use_container_width=True)

        # Votes distribution for entry bars
        bull_bars = df_signals[df_signals["IsBull"]]
        if not bull_bars.empty:
            vote_hist = go.Figure(
                go.Histogram(
                    x=bull_bars["Votes"],
                    nbinsx=9,
                    marker_color="#00e676",
                    opacity=0.8,
                    name="Vote Count",
                )
            )
            vote_hist.add_vline(x=7, line_color="#ff1744", line_dash="dash",
                                annotation_text="Entry threshold (7)", annotation_position="top right")
            vote_hist.update_layout(
                title="Confirmation Votes Distribution (Bull Bars Only)",
                height=350,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                font=dict(color="#e0e0e0"),
                xaxis_title="Votes (out of 8)",
                yaxis_title="Frequency",
            )
            st.plotly_chart(vote_hist, use_container_width=True)

        # Regime transition heatmap
        st.subheader("Regime Transition Probability Matrix")
        present = df_signals["Regime"].values
        future = np.roll(present, -1)
        unique = LABEL_NAMES
        matrix = pd.DataFrame(0, index=unique, columns=unique, dtype=float)

        for cur, nxt in zip(present[:-1], future[:-1]):
            if cur in matrix.index and nxt in matrix.columns:
                matrix.loc[cur, nxt] += 1

        # Normalise rows
        row_sums = matrix.sum(axis=1)
        matrix = matrix.div(row_sums + 1e-9, axis=0) * 100
        matrix = matrix.loc[matrix.index.isin(regime_counts.index),
                             matrix.columns.isin(regime_counts.index)]

        heatmap = go.Figure(
            go.Heatmap(
                z=matrix.values,
                x=matrix.columns.tolist(),
                y=matrix.index.tolist(),
                colorscale="RdYlGn",
                text=np.round(matrix.values, 1),
                texttemplate="%{text}%",
                showscale=True,
            )
        )
        heatmap.update_layout(
            height=420,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            font=dict(color="#e0e0e0"),
        )
        st.plotly_chart(heatmap, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 – Trade Log
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader(f"Trade Log ({result.num_trades} trades, 2.5× Leveraged PnL)")

        if result.trades:
            rows = []
            for t in result.trades:
                pnl_color = "green" if t.pnl_pct >= 0 else "red"
                rows.append(
                    {
                        "Entry Time": str(t.entry_time)[:16],
                        "Entry Price": f"${t.entry_price:,.2f}",
                        "Exit Time": str(t.exit_time)[:16] if t.exit_time else "Open",
                        "Exit Price": f"${t.exit_price:,.2f}" if t.exit_price else "—",
                        "Exit Reason": t.exit_reason,
                        "Leveraged PnL %": fmt_pct(t.pnl_pct),
                    }
                )
            trade_df = pd.DataFrame(rows)
            st.dataframe(trade_df, hide_index=True, use_container_width=True)

            # PnL waterfall
            pnls = [t.pnl_pct for t in result.trades]
            cum_pnls = np.cumsum(pnls)
            bar_colors = ["#00e676" if p >= 0 else "#ff1744" for p in pnls]
            wf_fig = go.Figure()
            wf_fig.add_trace(
                go.Bar(
                    x=list(range(1, len(pnls) + 1)),
                    y=pnls,
                    marker_color=bar_colors,
                    name="Trade PnL %",
                )
            )
            wf_fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(pnls) + 1)),
                    y=cum_pnls,
                    name="Cumulative PnL %",
                    line=dict(color="#ffab40", width=2),
                    mode="lines+markers",
                )
            )
            wf_fig.update_layout(
                title="Per-Trade PnL (%) with Cumulative",
                height=400,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                font=dict(color="#e0e0e0"),
                xaxis_title="Trade #",
                yaxis_title="PnL %",
                hovermode="x unified",
            )
            st.plotly_chart(wf_fig, use_container_width=True)
        else:
            st.info("No completed trades found in the backtest window.")

    st.markdown("---")
    st.caption(
        "⚠️ This application is for **educational and research purposes only**. "
        "Past simulated performance is not indicative of future results. "
        "Leverage amplifies both gains and losses. Not financial advice."
    )


if __name__ == "__main__":
    main()
