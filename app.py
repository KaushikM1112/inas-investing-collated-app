
import os
import json
import time
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import streamlit as st

# Optional imports for live data & Google Sheets; app will still run without them.
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import gspread  # type: ignore
    from google.oauth2.service_account import Credentials  # type: ignore
except Exception:
    gspread = None
    Credentials = None

# -----------------------------
# App Config & Constants
# -----------------------------
st.set_page_config(
    page_title="Inas Investing — Collated App",
    page_icon="📈",
    layout="wide",
)

DEFAULT_HOLDINGS = [
    {"Ticker": "QQQ", "Quantity": 2, "CostBasis_AUD": 800, "Notes": "High-growth US tech ETF"},
    {"Ticker": "NDQ.AX", "Quantity": 5, "CostBasis_AUD": 1500, "Notes": "Nasdaq 100 (AUD)"},
    {"Ticker": "FAANG", "Quantity": 10, "CostBasis_AUD": 2200, "Notes": "Thematic high beta"},
    {"Ticker": "BTC-USD", "Quantity": 0.02, "CostBasis_AUD": 1800, "Notes": "Bitcoin"},
    {"Ticker": "ETH-USD", "Quantity": 0.3, "CostBasis_AUD": 1400, "Notes": "Ethereum"},
    {"Ticker": "GOLD.AX", "Quantity": 8, "CostBasis_AUD": 2000, "Notes": "GOLD ETF (physical gold exposure)"},
]

DEFAULT_TARGETS = {
    "QQQ": 0.20,
    "NDQ.AX": 0.20,
    "FAANG": 0.10,
    "BTC-USD": 0.20,
    "ETH-USD": 0.10,
    "GOLD.AX": 0.20,
}

APP_SECTIONS = [
    "Home — Overview & Allocation",
    "Live Tracker (with Lag Compensation)",
    "Gold Tracking (Live + Manual)",
    "Rebalance Advisor",
    "Alerts",
    "Data — Edit Holdings / Targets",
    "Settings — Google Sheets & Persistence",
    "Help",
]

LOCAL_HOLDINGS_PATH = "holdings.json"
LOCAL_ALERTS_PATH = "alerts.json"
LOCAL_TARGETS_PATH = "targets.json"

# -----------------------------
# Utility & Persistence
# -----------------------------

def _now_aware():
    return datetime.now(timezone.utc)

def load_local_json(path: str, fallback):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return fallback

def save_local_json(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False

def try_read_gsheets(sheet_name="Holdings"):
    """Read from Google Sheets if configured; else return None.

    Expect Streamlit secrets configured as:
    [gsheets]
    sheet_id = "your_google_sheet_id"
    worksheet = "Holdings"

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    client_email = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\n..."
    """
    if gspread is None or Credentials is None:
        return None

    try:
        gs_cfg = st.secrets.get("gsheets", {})
        svc = st.secrets.get("gcp_service_account", None)
        if not gs_cfg or not svc:
            return None

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(dict(svc), scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(gs_cfg["sheet_id"])
        ws = sh.worksheet(gs_cfg.get("worksheet", sheet_name))

        records = ws.get_all_records()
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        st.sidebar.warning(f"Google Sheets read skipped: {e}")
        return None

def try_write_gsheets(df: pd.DataFrame, sheet_name="Holdings"):
    if gspread is None or Credentials is None:
        return False

    try:
        gs_cfg = st.secrets.get("gsheets", {})
        svc = st.secrets.get("gcp_service_account", None)
        if not gs_cfg or not svc:
            return False

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(dict(svc), scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(gs_cfg["sheet_id"])
        ws = sh.worksheet(gs_cfg.get("worksheet", sheet_name))

        # Clear then update
        ws.clear()
        ws.update([df.columns.values.tolist()] + df.values.tolist())
        return True
    except Exception as e:
        st.sidebar.warning(f"Google Sheets write skipped: {e}")
        return False

def get_prices_yf(tickers: List[str]) -> Dict[str, float]:
    """Fetch latest prices using yfinance. Returns dict {ticker: price}.
    If yfinance is missing or fails, returns last known or 1.0 fallback.
    """
    prices = {}
    if yf is None:
        for t in tickers:
            prices[t] = 1.0
        return prices

    try:
        data = yf.download(tickers=tickers, period="1d", interval="1m", group_by="ticker", auto_adjust=True, threads=True, progress=False)
        # yfinance returns different shapes if single vs multi
        def last_close(frame):
            try:
                if isinstance(frame, pd.DataFrame):
                    return float(frame["Close"].dropna().iloc[-1])
            except Exception:
                pass
            return np.nan

        if isinstance(data, pd.DataFrame) and "Close" in data.columns:
            # Single ticker
            prices[tickers[0]] = float(data["Close"].dropna().iloc[-1])
        else:
            # Multi
            for t in tickers:
                try:
                    prices[t] = float(data[t]["Close"].dropna().iloc[-1])
                except Exception:
                    prices[t] = np.nan

        # Fallback for NaNs
        for t in tickers:
            if math.isnan(prices.get(t, np.nan)):
                prices[t] = 1.0
    except Exception:
        for t in tickers:
            prices[t] = 1.0
    return prices

def drift_compensation(prices_ts: List[Tuple[datetime, float]], horizon_minutes: int = 10) -> float:
    """Very simple linear drift model: recent slope * horizon + last price."""
    if len(prices_ts) < 2:
        return prices_ts[-1][1] if prices_ts else 1.0
    t = np.array([(p[0] - prices_ts[0][0]).total_seconds() / 60.0 for p in prices_ts], dtype=float)
    y = np.array([p[1] for p in prices_ts], dtype=float)
    # linear regression slope
    slope = 0.0
    try:
        A = np.vstack([t, np.ones_like(t)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        last_t = (prices_ts[-1][0] - prices_ts[0][0]).total_seconds() / 60.0
        est = slope * (last_t + horizon_minutes) + intercept
        return float(est)
    except Exception:
        return float(y[-1])

# -----------------------------
# Session State Initialization
# -----------------------------

if "holdings_df" not in st.session_state:
    # Try Google Sheets first
    df_gs = try_read_gsheets("Holdings")
    if df_gs is not None and set(df_gs.columns) >= {"Ticker", "Quantity"}:
        st.session_state.holdings_df = df_gs
    else:
        # fallback to local file or defaults
        holdings = load_local_json(LOCAL_HOLDINGS_PATH, DEFAULT_HOLDINGS)
        st.session_state.holdings_df = pd.DataFrame(holdings)

if "targets" not in st.session_state:
    st.session_state.targets = load_local_json(LOCAL_TARGETS_PATH, DEFAULT_TARGETS)

if "alerts" not in st.session_state:
    st.session_state.alerts = load_local_json(LOCAL_ALERTS_PATH, {
        # example alerts
        "BTC-USD": {"price_above": 120000, "price_below": 80000},
        "GOLD.AX": {"alloc_above": 0.25, "alloc_below": 0.15},
    })

if "price_cache" not in st.session_state:
    st.session_state.price_cache = {}  # {ticker: [(ts, price), ...]}
if "last_update" not in st.session_state:
    st.session_state.last_update = None

# -----------------------------
# Sidebar Navigation
# -----------------------------

st.sidebar.title("📦 Inas Investing — Collated")
section = st.sidebar.radio("Navigate", APP_SECTIONS, index=0)
st.sidebar.caption("Tip: Configure Google Sheets in Settings to persist data across sessions.")

# -----------------------------
# Core Computations
# -----------------------------

def compute_portfolio_values(holdings_df: pd.DataFrame, prices: Dict[str, float]) -> pd.DataFrame:
    df = holdings_df.copy()
    df["Price_AUD"] = df["Ticker"].map(prices).fillna(0.0)
    df["MarketValue_AUD"] = df["Quantity"] * df["Price_AUD"]
    return df

def allocation(df: pd.DataFrame) -> pd.Series:
    total = df["MarketValue_AUD"].sum()
    if total <= 0:
        return pd.Series(0, index=df["Ticker"], dtype=float)
    alloc = df.set_index("Ticker")["MarketValue_AUD"] / total
    return alloc

def rebalance_suggestions(current_alloc: pd.Series, targets: Dict[str, float], df_values: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    # Merge current vs target
    idx = sorted(set(current_alloc.index).union(targets.keys()))
    out = pd.DataFrame(index=idx, columns=["Target", "Current", "Diff", "Action", "Qty_Adjust"])
    out["Target"] = [targets.get(i, 0.0) for i in idx]
    out["Current"] = [float(current_alloc.get(i, 0.0)) for i in idx]
    out["Diff"] = out["Current"] - out["Target"]

    # Suggest buy/sell for those exceeding threshold
    total_mv = df_values["MarketValue_AUD"].sum()
    px_map = df_values.set_index("Ticker")["Price_AUD"].to_dict()
    qty_map = df_values.set_index("Ticker")["Quantity"].to_dict()

    for i in idx:
        diff = float(out.loc[i, "Diff"])
        if abs(diff) < threshold:
            out.loc[i, "Action"] = "OK"
            out.loc[i, "Qty_Adjust"] = 0.0
            continue
        action = "Sell" if diff > 0 else "Buy"
        delta_value = (-diff) * total_mv  # how much value to add/remove
        price = float(px_map.get(i, 0.0)) or 1.0
        qty_adj = delta_value / price
        out.loc[i, "Action"] = action
        out.loc[i, "Qty_Adjust"] = round(qty_adj, 4)

    return out

def evaluate_alerts(df_values: pd.DataFrame, targets: Dict[str, float], alerts: Dict) -> List[str]:
    notes = []
    if df_values.empty: 
        return notes

    prices = df_values.set_index("Ticker")["Price_AUD"].to_dict()
    allocs = allocation(df_values)

    for tkr, cfg in alerts.items():
        if not isinstance(cfg, dict): 
            continue
        p = prices.get(tkr)
        a = float(allocs.get(tkr, 0.0))

        if p is not None:
            if "price_above" in cfg and p > cfg["price_above"]:
                notes.append(f"⚠️ {tkr} price crossed above {cfg['price_above']:.2f} (now {p:.2f})")
            if "price_below" in cfg and p < cfg["price_below"]:
                notes.append(f"⚠️ {tkr} price fell below {cfg['price_below']:.2f} (now {p:.2f})")

        if "alloc_above" in cfg and a > cfg["alloc_above"]:
            notes.append(f"⚠️ {tkr} allocation above {cfg['alloc_above']*100:.1f}% (now {a*100:.1f}%)")
        if "alloc_below" in cfg and a < cfg["alloc_below"]:
            notes.append(f"⚠️ {tkr} allocation below {cfg['alloc_below']*100:.1f}% (now {a*100:.1f}%)")

    return notes

def maybe_fetch_prices(holdings_df: pd.DataFrame, force: bool = False) -> Dict[str, float]:
    tickers = sorted(set(holdings_df["Ticker"].dropna().astype(str).tolist()))
    prices = {}

    needs_update = force
    if st.session_state.last_update is None:
        needs_update = True
    else:
        # consider stale if > 5 minutes
        if (_now_aware() - st.session_state.last_update) > timedelta(minutes=5):
            needs_update = True

    if needs_update:
        new_prices = get_prices_yf(tickers)
        now = _now_aware()
        for t, px in new_prices.items():
            st.session_state.price_cache.setdefault(t, [])
            st.session_state.price_cache[t].append((now, float(px)))
            # keep only last 50 points
            st.session_state.price_cache[t] = st.session_state.price_cache[t][-50:]
        st.session_state.last_update = now

    # Use last known prices
    for t in tickers:
        hist = st.session_state.price_cache.get(t, [])
        prices[t] = hist[-1][1] if hist else 1.0

    return prices

# -----------------------------
# UI Sections
# -----------------------------

def ui_header():
    st.title("📈 Inas Investing — Collated App")
    st.caption("Live updates • Gold tracking • Rebalance advisor • Alerts • Google Sheets persistence • Lag compensation")

    if st.session_state.last_update:
        age_min = (_now_aware() - st.session_state.last_update).total_seconds() / 60.0
        badge = "🔴 Stale" if age_min > 10 else ("🟠 Aging" if age_min > 5 else "🟢 Fresh")
        st.info(f"Last price update: **{st.session_state.last_update.isoformat()}** — Status: {badge}")
    else:
        st.warning("Prices not fetched yet. Use the **Refresh Prices** button.")

    cols = st.columns(3)
    with cols[0]:
        if st.button("🔄 Refresh Prices", use_container_width=True):
            maybe_fetch_prices(st.session_state.holdings_df, force=True)
            st.experimental_rerun()
    with cols[1]:
        if st.button("💾 Save to Google Sheets (Holdings)", use_container_width=True):
            ok = try_write_gsheets(st.session_state.holdings_df, "Holdings")
            st.success("Saved to Google Sheets.") if ok else st.error("Could not save to Google Sheets.")
    with cols[2]:
        if st.button("📝 Save Locally", use_container_width=True):
            save_local_json(LOCAL_HOLDINGS_PATH, st.session_state.holdings_df.to_dict(orient="records"))
            save_local_json(LOCAL_TARGETS_PATH, st.session_state.targets)
            save_local_json(LOCAL_ALERTS_PATH, st.session_state.alerts)
            st.success("Saved holdings, targets, and alerts locally.")

def section_home():
    st.subheader("Overview & Allocation")
    prices = maybe_fetch_prices(st.session_state.holdings_df, force=False)
    df_vals = compute_portfolio_values(st.session_state.holdings_df, prices)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(df_vals, use_container_width=True)
    with c2:
        total = float(df_vals["MarketValue_AUD"].sum())
        st.metric("Portfolio Value (AUD)", f"{total:,.2f}")
        alloc_series = allocation(df_vals).sort_values(ascending=False)
        st.write("**Allocation**")
        st.bar_chart(alloc_series)

def section_live_tracker():
    st.subheader("Live Tracker with Lag Compensation")
    st.caption("Shows latest prices and a simple extrapolation if data is stale.")

    prices = maybe_fetch_prices(st.session_state.holdings_df, force=False)
    df_vals = compute_portfolio_values(st.session_state.holdings_df, prices)

    # Drift compensation per ticker if stale > 5 min
    if st.session_state.last_update and (_now_aware() - st.session_state.last_update) > timedelta(minutes=5):
        st.warning("Data appears stale. Applying drift compensation (+10min linear extrapolation).")
        comp_prices = {}
        for t, hist in st.session_state.price_cache.items():
            comp_prices[t] = drift_compensation(hist, 10)
        df_vals["Price_AUD"] = df_vals["Ticker"].map(comp_prices).fillna(df_vals["Price_AUD"])
        df_vals["MarketValue_AUD"] = df_vals["Quantity"] * df_vals["Price_AUD"]

    st.dataframe(df_vals, use_container_width=True)

def section_gold_tracking():
    st.subheader("Gold Tracking (Live + Manual)")
    st.caption("Track exposure via GOLD.AX or enter manual spot price and ounces.")

    prices = maybe_fetch_prices(st.session_state.holdings_df, force=False)
    gold_tickers = [t for t in st.session_state.holdings_df["Ticker"].tolist() if "GOLD" in str(t).upper() or "XAU" in str(t).upper()]
    default_gold_tkr = gold_tickers[0] if gold_tickers else "GOLD.AX"
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Live Gold via Ticker**")
        gold_tkr = st.text_input("Gold Ticker (e.g., GOLD.AX or XAUUSD=X)", value=default_gold_tkr)
        gold_px = prices.get(gold_tkr, np.nan)
        st.metric(f"{gold_tkr} Price (AUD est.)", f"{gold_px:.2f}" if not np.isnan(gold_px) else "N/A")

    with col2:
        st.write("**Manual Gold Entry**")
        manual_oz = st.number_input("Ounces held", min_value=0.0, value=0.0, step=0.1)
        manual_spot = st.number_input("Manual Spot (AUD/oz)", min_value=0.0, value=0.0, step=1.0)
        if manual_oz and manual_spot:
            st.metric("Manual Gold Value (AUD)", f"{manual_oz * manual_spot:,.2f}")

def section_rebalance():
    st.subheader("Rebalance Advisor")
    st.caption("Compares current allocation with targets and suggests buy/sell qty adjustments.")

    prices = maybe_fetch_prices(st.session_state.holdings_df, force=False)
    df_vals = compute_portfolio_values(st.session_state.holdings_df, prices)
    cur_alloc = allocation(df_vals)
    threshold = st.slider("Suggestion threshold (abs % diff)", 0.0, 0.1, 0.02, 0.005)

    out = rebalance_suggestions(cur_alloc, st.session_state.targets, df_vals, threshold)
    st.dataframe(out, use_container_width=True)

def section_alerts():
    st.subheader("Alerts")
    st.caption("Set price and allocation alerts. Alerts are checked on refresh.")

    prices = maybe_fetch_prices(st.session_state.holdings_df, force=False)
    df_vals = compute_portfolio_values(st.session_state.holdings_df, prices)

    # Editor
    st.write("**Edit Alerts**")
    tickers = st.session_state.holdings_df["Ticker"].astype(str).tolist()
    selected = st.selectbox("Ticker", options=tickers)
    cfg = st.session_state.alerts.get(selected, {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pa = st.number_input("Price Above", value=float(cfg.get("price_above", 0.0)), step=1.0)
    with c2:
        pb = st.number_input("Price Below", value=float(cfg.get("price_below", 0.0)), step=1.0)
    with c3:
        aa = st.number_input("Alloc Above (0-1)", value=float(cfg.get("alloc_above", 0.0)), step=0.01)
    with c4:
        ab = st.number_input("Alloc Below (0-1)", value=float(cfg.get("alloc_below", 0.0)), step=0.01)

    if st.button("Save Alert"):
        st.session_state.alerts[selected] = {
            "price_above": pa if pa > 0 else None,
            "price_below": pb if pb > 0 else None,
            "alloc_above": aa if aa > 0 else None,
            "alloc_below": ab if ab > 0 else None,
        }
        save_local_json(LOCAL_ALERTS_PATH, st.session_state.alerts)
        st.success("Alert saved.")

    # Evaluation
    notes = evaluate_alerts(df_vals, st.session_state.targets, st.session_state.alerts)
    st.write("**Alert Feed**")
    if notes:
        for n in notes:
            st.warning(n)
    else:
        st.info("No alerts currently triggered.")

def section_data():
    st.subheader("Data — Edit Holdings / Targets")

    st.write("**Holdings**")
    st.caption("Edit your holdings below. Use valid ticker symbols for live pricing via yfinance.")
    st.session_state.holdings_df = st.data_editor(
        st.session_state.holdings_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )

    st.write("**Targets**")
    # Build a table-like editor for targets
    tickers = st.session_state.holdings_df["Ticker"].astype(str).tolist()
    tgt_vals = {t: float(st.session_state.targets.get(t, 0.0)) for t in tickers}
    cols = st.columns(3)
    parts = [("Begin", tickers[::3]), ("Mid", tickers[1::3]), ("End", tickers[2::3])]
    updated = {}
    for col, (_, bucket) in zip(cols, parts):
        with col:
            for t in bucket:
                updated[t] = st.number_input(f"Target for {t}", min_value=0.0, max_value=1.0, value=tgt_vals.get(t, 0.0), step=0.01, key=f"tgt_{t}")

    if st.button("Save Targets"):
        st.session_state.targets = updated
        save_local_json(LOCAL_TARGETS_PATH, updated)
        st.success("Targets saved.")

def section_settings():
    st.subheader("Settings — Google Sheets & Persistence")
    st.markdown("""
Configure Streamlit Secrets for Google Sheets to persist your holdings:

```toml
# .streamlit/secrets.toml
[gcp_service_account]
type = "service_account"
project_id = "YOUR_PROJECT_ID"
private_key_id = "YOUR_KEY_ID"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "YOUR_SA_EMAIL@YOUR_PROJECT_ID.iam.gserviceaccount.com"
client_id = "YOUR_CLIENT_ID"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
client_x509_cert_url = "..."

[gsheets]
sheet_id = "YOUR_SHEET_ID"
worksheet = "Holdings"
```
Then use the **Save to Google Sheets** button in the header.
    """)

    st.write("**Local Persistence**")
    st.code(f"holdings.json • targets.json • alerts.json", language="bash")

def section_help():
    st.subheader("Help")
    st.markdown("""
**What's inside this collated app?**

- **Live Updates** via `yfinance`
- **Allocation Chart** (Home)
- **Gold Tracking** (Live ticker or manual ounces & spot)
- **Rebalance Advisor** (suggest Qty adjustments to move toward targets)
- **Alerts System** (price/alloc thresholds)
- **Lag Compensation** (simple linear extrapolation when data is stale)
- **Google Sheets Integration** (optional)

**Deploy on Streamlit Cloud**
1. Upload these files to a public GitHub repo.
2. On Streamlit Cloud, set the app to run `app.py`.
3. In **Advanced settings**, add your **secrets** (copy from `.streamlit/secrets.toml.template`).
4. Deploy.

**Local Run**
```bash
pip install -r requirements.txt
streamlit run app.py
```
    """)

# -----------------------------
# Render
# -----------------------------

ui_header()

if section == "Home — Overview & Allocation":
    section_home()
elif section == "Live Tracker (with Lag Compensation)":
    section_live_tracker()
elif section == "Gold Tracking (Live + Manual)":
    section_gold_tracking()
elif section == "Rebalance Advisor":
    section_rebalance()
elif section == "Alerts":
    section_alerts()
elif section == "Data — Edit Holdings / Targets":
    section_data()
elif section == "Settings — Google Sheets & Persistence":
    section_settings()
else:
    section_help()
