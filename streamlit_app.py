"""
Live-updating correlation dashboard + trading journal
Usage:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
Notes:
    - Uses yfinance for price data (no API key).
    - Persists journal to a local SQLite DB file 'trading_journal.db'.
    - For 'live' updates Streamlit refreshes on a timer (default 60s).
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
from datetime import datetime, timezone
import plotly.express as px
import plotly.graph_objects as go
import io
import os

# ---------------------------
# Config / defaults
# ---------------------------
DB_PATH = "trading_journal.db"
DEFAULT_T1 = "QQQ"
DEFAULT_T2 = "SPY"
DEFAULT_INTERVAL = "1m"

st.set_page_config(layout="wide", page_title="Correlation Dashboard & Trading Journal")

# ---------------------------
# Utility: Data fetcher
# ---------------------------
@st.cache_data(ttl=30)
def fetch_data(ticker, period="7d", interval="1m"):
    """Fetch OHLCV data with improved error handling"""
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        
        # When downloading a single ticker, yfinance might return a flat or multi-level column index.
        # We need to flatten it to a consistent format.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Ensure we have all required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns for {ticker}")
            return pd.DataFrame()
        
        # Add ticker prefix to column names to avoid conflicts
        df = df[required_cols].copy()
        df.columns = [f"{ticker}_{col.lower()}" for col in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Data fetch error for {ticker}: {e}")
        return pd.DataFrame()

# ---------------------------
# Utility: Journal DB
# ---------------------------
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS journal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        ticker TEXT,
        direction TEXT,
        size REAL,
        entry_price REAL,
        exit_price REAL,
        pnl REAL,
        pnl_currency TEXT,
        sentiment TEXT,
        notes TEXT
    )
    """)
    conn.commit()
    return conn

def add_journal_entry(conn, entry):
    c = conn.cursor()
    c.execute("""
    INSERT INTO journal (timestamp,ticker,direction,size,entry_price,exit_price,pnl,pnl_currency,sentiment,notes)
    VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        entry["timestamp"],
        entry["ticker"],
        entry["direction"],
        entry["size"],
        entry["entry_price"],
        entry.get("exit_price"),
        entry.get("pnl"),
        entry.get("pnl_currency","USD"),
        entry.get("sentiment"),
        entry.get("notes")
    ))
    conn.commit()

def get_journal_df(conn):
    df = pd.read_sql_query("SELECT * FROM journal ORDER BY timestamp DESC", conn, parse_dates=["timestamp"])
    return df

# ---------------------------
# Sidebar: Controls
# ---------------------------
st.sidebar.title("Controls")
t1 = st.sidebar.text_input("Ticker 1 (left)", value=DEFAULT_T1).upper()
t2 = st.sidebar.text_input("Ticker 2 (right)", value=DEFAULT_T2).upper()
period = st.sidebar.selectbox("History period", options=["1d","5d","7d","30d","90d","180d","1y","2y"], index=2)
interval = st.sidebar.selectbox("Data interval", options=["1m","2m","5m","15m","30m","60m","90m","1d"], index=0)
rolling_window = st.sidebar.slider("Rolling correlation window (bars)", 5, 500, 60)
show_signals = st.sidebar.checkbox("Show trading signals", value=True)

# Initialize empty DataFrames and Series at global scope
merged = pd.DataFrame()
ratio = pd.Series(dtype=float)
zscore = pd.Series(dtype=float)
rolling_corr = pd.Series(dtype=float)

# ---------------------------
# Main UI layout
# ---------------------------
# Data fetching and processing
st.title(f"Correlation Dashboard — *{t1}* vs *{t2}*")
col1, col2 = st.columns([2,1])

# Fetch data first
df1 = fetch_data(t1, period=period, interval=interval)
df2 = fetch_data(t2, period=period, interval=interval)

# Process data if available
if not df1.empty and not df2.empty:
    # Merge dataframes on index
    merged = pd.concat([df1, df2], axis=1).dropna()
    
    if not merged.empty:
        # Calculate ratio and z-score
        ratio = merged[f"{t1}_close"] / merged[f"{t2}_close"]
        if len(ratio) >= 2 and rolling_window > 1:
            win = min(rolling_window, len(ratio))
            roll_mean = ratio.rolling(window=win).mean()
            roll_std = ratio.rolling(window=win).std(ddof=0).replace({0: np.nan})
            zscore = (ratio - roll_mean) / roll_std
            
            # Calculate correlation
            rolling_corr = merged[f"{t1}_close"].rolling(
                window=min(rolling_window, len(merged))
            ).corr(merged[f"{t2}_close"])

# Display charts and stats using the calculated data
with col1:
    st.subheader("Price charts")
    if df1.empty or df2.empty:
        st.warning("No data returned for one or both tickers. Check symbols and market hours.")
    elif merged.empty:
        st.warning("No overlapping data points between tickers.")
    else:
        # Enhanced price chart with prefixed column names
        fig = go.Figure()
        # Add candlestick for t1
        fig.add_trace(go.Candlestick(
            x=merged.index,
            open=merged[f'{t1}_open'],
            high=merged[f'{t1}_high'],
            low=merged[f'{t1}_low'],
            close=merged[f'{t1}_close'],
            name=t1,
            yaxis="y",
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        # Add candlestick for t2
        fig.add_trace(go.Candlestick(
            x=merged.index,
            open=merged[f'{t2}_open'],
            high=merged[f'{t2}_high'],
            low=merged[f'{t2}_low'],
            close=merged[f'{t2}_close'],
            name=t2,
            yaxis="y2",
            increasing_line_color='blue',
            decreasing_line_color='orange'
        ))
        fig.update_layout(
            yaxis=dict(title=f"{t1}"),
            yaxis2=dict(title=f"{t2}", overlaying="y", side="right"),
            height=400,
            margin=dict(l=40,r=40,t=30,b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        r_fig = go.Figure()
        r_fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name=f"{t1}/{t2} ratio"))
        if not ratio.empty:
            r_fig.add_hline(y=ratio.rolling(window=min(rolling_window, len(ratio))).mean().iloc[-1], line_dash="dash", annotation_text="Rolling mean")
        r_fig.update_layout(height=260, margin=dict(l=40,r=40,t=10,b=10))
        st.plotly_chart(r_fig, use_container_width=True)

        zs_fig = go.Figure()
        zs_fig.add_trace(go.Scatter(x=zscore.index, y=zscore, name="ratio z-score"))
        zs_fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="+2")
        zs_fig.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="-2")
        zs_fig.update_layout(height=220, margin=dict(l=40,r=40,t=10,b=10))
        st.plotly_chart(zs_fig, use_container_width=True)

with col2:
    st.subheader("Stats & Correlation")
    if merged.empty:
        st.warning("Awaiting valid data for both tickers...")
    else:
        # Enhanced stats display
        for ticker in [t1, t2]:
            with st.expander(f"{ticker} Stats", expanded=True):
                last_close = merged[f'{ticker}_close'].iloc[-1]
                last_open = merged[f'{ticker}_open'].iloc[-1]
                day_high = merged[f'{ticker}_high'].iloc[-1]
                day_low = merged[f'{ticker}_low'].iloc[-1]
                
                stats = {
                    "Last": f"${last_close:.2f}",
                    "Open": f"${last_open:.2f}",
                    "High": f"${day_high:.2f}",
                    "Low": f"${day_low:.2f}",
                    "Change": f"{((last_close/last_open)-1)*100:.2f}%",
                    "H/L Range": f"${day_high-day_low:.2f}",
                }
                st.json(stats)

        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Current Correlation", 
                     f"{rolling_corr.iloc[-1]:.3f}",
                     f"{rolling_corr.iloc[-1] - rolling_corr.iloc[-2]:.3f}")
        with metrics_col2:
            st.metric("Ratio", 
                     f"{(merged[f'{t1}_close'].iloc[-1] / merged[f'{t2}_close'].iloc[-1]):.3f}")

# ---------------------------
# Scatter / regression
# ---------------------------
st.markdown("---")
st.subheader("Scatter plot & linear fit (price co-movement)")
if not merged.empty:
    scatter_df = merged.dropna().tail(500)  # limit points
    sc_fig = px.scatter(scatter_df, x=f"{t2}_close", y=f"{t1}_close", trendline="ols", title=f"{t1} vs {t2} scatter")
    st.plotly_chart(sc_fig, use_container_width=True)

# ---------------------------
# Signals & notes
# ---------------------------
st.markdown("---")
st.subheader("Pair trading signal (simple z-score rules)")
if not merged.empty and show_signals and not zscore.empty:
    latest_z = zscore.iloc[-1]
    st.write(f"*Latest ratio z-score*: **{latest_z:.3f}**")
    if latest_z > 2:
        st.success(f"Signal: *Short the ratio* — consider short {t1} / long {t2} (mean-reversion).")
    elif latest_z < -2:
        st.success(f"Signal: *Long the ratio* — consider long {t1} / short {t2} (mean-reversion).")
    else:
        st.info("No strong signal (z-score within ±2).")
else:
    if show_signals:
        st.info("No signal: insufficient data to compute z-score.")

# ---------------------------
# Trading Journal UI
# ---------------------------
st.markdown("---")
st.subheader("Trading Journal")

conn = init_db(DB_PATH)

with st.expander("Add new journal entry"):
    with st.form("entry_form", clear_on_submit=True):
        ts = datetime.now(timezone.utc).astimezone().isoformat()
        entry_ticker = st.text_input("Ticker", value=t1).upper()
        direction = st.selectbox("Direction", options=["Long","Short"])
        size = st.number_input("Size (shares/contracts)", min_value=0.0, value=1.0, format="%.4f")
        entry_price = st.number_input("Entry price", min_value=0.0, format="%.6f")
        exit_price = st.number_input("Exit price (optional)", min_value=0.0, format="%.6f")
        pnl_currency = st.selectbox("P/L currency", options=["USD"], index=0)
        sentiment = st.selectbox("Sentiment", options=["Bullish","Neutral","Bearish","Speculative"])
        notes = st.text_area("Notes / rationale")
        submitted = st.form_submit_button("Add entry")
        if submitted:
            # compute realized pnl only if exit_price was provided (>0)
            pnl = None
            if exit_price > 0:
                if direction == "Long":
                    pnl = (exit_price - entry_price) * size
                else:
                    pnl = (entry_price - exit_price) * size
            entry = {
                "timestamp": ts,
                "ticker": entry_ticker,
                "direction": direction,
                "size": size,
                "entry_price": entry_price,
                "exit_price": exit_price if exit_price>0 else None,
                "pnl": pnl,
                "pnl_currency": pnl_currency,
                "sentiment": sentiment,
                "notes": notes
            }
            add_journal_entry(conn, entry)
            st.success("Journal entry saved.")

# Display journal
journal_df = get_journal_df(conn)
if journal_df.empty:
    st.info("Journal is empty. Add entries above.")
else:
    st.write(f"Showing {len(journal_df)} journal rows (most recent first).")
    # filters
    cols = st.columns([1,1,1,1])
    with cols[0]:
        ft = st.selectbox("Filter ticker", options=["All"] + sorted(journal_df['ticker'].dropna().unique().tolist()))
    with cols[1]:
        ff = st.selectbox("Filter direction", options=["All","Long","Short"])
    with cols[2]:
        fs = st.selectbox("Filter sentiment", options=["All"] + sorted(journal_df['sentiment'].dropna().unique().tolist()))
    with cols[3]:
        download_csv = st.button("Export CSV")

    q = journal_df.copy()
    if ft and ft != "All":
        q = q[q['ticker']==ft]
    if ff and ff != "All":
        q = q[q['direction']==ff]
    if fs and fs != "All":
        q = q[q['sentiment']==fs]

    st.dataframe(q, height=300)

    if download_csv:
        towrite = io.BytesIO()
        q.to_csv(towrite, index=False)
        towrite.seek(0)
        st.download_button("Download CSV", towrite, file_name="journal_export.csv", mime="text/csv")

    # Summaries
    st.markdown("**Journal summaries**")
    if "pnl" in journal_df.columns and journal_df['pnl'].notna().any():
        pnl_df = journal_df.dropna(subset=["pnl"])
        pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
        pnl_by_ticker = pnl_df.groupby("ticker")['pnl'].sum().reset_index().sort_values("pnl", ascending=False)
        fig_pnl = px.bar(pnl_by_ticker, x="ticker", y="pnl", title="Realized P/L by ticker")
        st.plotly_chart(fig_pnl, use_container_width=True)
    # sentiment breakdown
    sent_counts = journal_df['sentiment'].value_counts().reset_index()
    sent_counts.columns = ["sentiment","count"]
    fig_sent = px.pie(sent_counts, names="sentiment", values="count", title="Sentiment breakdown")
    st.plotly_chart(fig_sent, use_container_width=True)

st.markdown("---")
st.caption("Built with yfinance + Streamlit. Save the `trading_journal.db` file to persist your journal across sessions.")