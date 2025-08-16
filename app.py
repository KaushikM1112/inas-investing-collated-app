
import os, io, json, time, math, functools, typing as t
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:
    yf = None

GSPREAD_OK = True
try:
    import gspread
    from google.oauth2 import service_account
except Exception:
    GSPREAD_OK = False

st.set_page_config(page_title="Investment – Collated Final", layout="wide")
st.title("📈 Investment Dashboard – Collated Final (with Greedy Planner)")

DEFAULT_BASE_CURRENCY = "AUD"
FX_TICKER_AUDUSD = "AUDUSD=X"
GOLD_TICKER_USD = "GC=F"
BTC_TICKER_USD = "BTC-USD"

def fmt_money(x, cur="AUD"):
    try:
        return f"{cur} {x:,.2f}"
    except Exception:
        return f"{cur} {x}"

def safe_yf_download(ticker, period="1d", interval="1d"):
    if yf is None:
        return None
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, threads=False, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
        return None
    except Exception:
        return None

def get_last_price(ticker: str):
    df = safe_yf_download(ticker, period="5d", interval="1d")
    if df is None or df.empty:
        return (float('nan'), "no-data")
    p = float(df["Close"].dropna().iloc[-1])
    ts = df.index[-1].to_pydatetime()
    return (p, ts.strftime("%Y-%m-%d %H:%M UTC"))

def moving_average(series: pd.Series, window: int = 20) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < window:
        return float(s.mean()) if len(s) else float('nan')
    return float(s.tail(window).mean())

def drift_adjust(price: float, hours_since: float, drift_bps_per_hour: float = 2.0) -> float:
    if not np.isfinite(price) or not np.isfinite(hours_since):
        return price
    return price * (1.0 + (drift_bps_per_hour / 10000.0) * hours_since)

@st.cache_data(show_spinner=False)
def load_local_json():
    try:
        with open("holdings.json", "r") as f:
            return json.load(f)
    except Exception:
        try:
            with open("example_holdings.json", "r") as f:
                return json.load(f)
        except Exception:
            return {"base_currency": DEFAULT_BASE_CURRENCY, "targets": {}, "positions": []}

def save_local_json(data: dict):
    try:
        with open("holdings.json", "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False

def get_gsheet_client():
    if not GSPREAD_OK:
        return None
    try:
        secrets = st.secrets.get("gcp_service_account", None)
        sheet_key = st.secrets.get("sheet_id", None)
        if not secrets or not sheet_key:
            return None
        creds = service_account.Credentials.from_service_account_info(dict(secrets), scopes=["https://www.googleapis.com/auth/spreadsheets"])
        gc = gspread.authorize(creds)
        return gc, sheet_key
    except Exception:
        return None

def read_from_sheet():
    cli = get_gsheet_client()
    if not cli:
        return None
    gc, sheet_key = cli
    try:
        sh = gc.open_by_key(sheet_key)
        ws = sh.worksheet("holdings")
    except Exception:
        return None
    rows = ws.get_all_records()
    pos = []
    targets = {}
    base = DEFAULT_BASE_CURRENCY
    for r in rows:
        tck = str(r.get("Ticker","")).strip()
        if not tck:
            continue
        qty = float(r.get("Quantity", 0) or 0)
        cost = float(r.get("CostBasis_AUD", 0) or 0)
        tgt = r.get("TargetWeight", None)
        typ = r.get("Type", "")
        if tgt is not None and tgt != "":
            targets[tck] = float(tgt)
        pos.append({"ticker": tck, "qty": qty, "cost_aud": cost, "type": typ})
    return {"base_currency": base, "targets": targets, "positions": pos}

def write_to_sheet(data: dict):
    cli = get_gsheet_client()
    if not cli:
        return False
    gc, sheet_key = cli
    try:
        sh = gc.open_by_key(sheet_key)
        try:
            ws = sh.worksheet("holdings")
        except Exception:
            ws = sh.add_worksheet(title="holdings", rows=200, cols=10)
        rows = [["Ticker","Quantity","CostBasis_AUD","TargetWeight","Type"]]
        for p in data.get("positions", []):
            rows.append([p.get("ticker",""), p.get("qty",0), p.get("cost_aud",0), data.get("targets",{}).get(p.get("ticker",""), ""), p.get("type","")])
        ws.clear()
        ws.update("A1", rows)
        return True
    except Exception:
        return False

st.sidebar.header("Data Source")
use_sheets = st.sidebar.checkbox("Use Google Sheets (if configured in secrets)", value=False)
if st.sidebar.button("Load Holdings"):
    data = read_from_sheet() if use_sheets else load_local_json()
    if not data:
        st.error("Failed to load holdings from the selected source.")
    else:
        st.session_state["holdings"] = data
        st.success("Holdings loaded.")
if st.sidebar.button("Save Holdings"):
    data = st.session_state.get("holdings", None)
    if not data:
        st.warning("Nothing to save — load or edit first.")
    else:
        ok = write_to_sheet(data) if use_sheets else save_local_json(data)
        st.success("Saved." if ok else "Save failed.")

holdings = st.session_state.get("holdings", load_local_json())

st.subheader("🎯 Targets & Positions")
c1, c2 = st.columns([2, 3])
with c1:
    st.markdown("**Target Allocation (by ticker, %)**")
    targets = holdings.get("targets", {})
    editable_targets = [{"Ticker": k, "Target %": v} for k, v in targets.items()]
    tgt_df = st.data_editor(pd.DataFrame(editable_targets), num_rows="dynamic", use_container_width=True)
    new_targets = {row["Ticker"]: float(row["Target %"]) for _, row in tgt_df.dropna().iterrows() if str(row["Ticker"]).strip() != ""}
    s = sum(new_targets.values()) or 1.0
    new_targets = {k: v/s*100.0 for k, v in new_targets.items()}
with c2:
    st.markdown("**Positions**")
    pos = holdings.get("positions", [])
    pos_df = st.data_editor(pd.DataFrame(pos), num_rows="dynamic", use_container_width=True)
    new_positions = pos_df.fillna({"qty":0,"cost_aud":0,"type":""}).to_dict(orient="records")
holdings["targets"] = new_targets
holdings["positions"] = new_positions
st.session_state["holdings"] = holdings

st.subheader("💹 Live Prices & Valuation")
try:
    FX_TICKER_AUDUSD = "AUDUSD=X"
    GOLD_TICKER_USD = "GC=F"
    BTC_TICKER_USD = "BTC-USD"
    fx_price, fx_ts = get_last_price(FX_TICKER_AUDUSD)
    gold_usd, gold_ts = get_last_price(GOLD_TICKER_USD)
    btc_usd, btc_ts = get_last_price(BTC_TICKER_USD)
except Exception:
    fx_price, fx_ts, gold_usd, gold_ts, btc_usd, btc_ts = float('nan'), "no-data", float('nan'), "no-data", float('nan'), "no-data"

col_fx, col_opts = st.columns([3,2])
with col_fx:
    st.write(f"**AUDUSD** last: {fx_price if np.isfinite(fx_price) else 'n/a'} @ {fx_ts}")
    audusd = fx_price if np.isfinite(fx_price) and fx_price>0 else float('nan')
    gold_aud = gold_usd / audusd if np.isfinite(gold_usd) and np.isfinite(audusd) else float('nan')
    btc_aud = btc_usd / audusd if np.isfinite(btc_usd) and np.isfinite(audusd) else float('nan')
    st.write(f"**Gold USD**: {gold_usd if np.isfinite(gold_usd) else 'n/a'} | **Gold AUD est**: {gold_aud if np.isfinite(gold_aud) else 'n/a'}")
    st.write(f"**BTC USD**: {btc_usd if np.isfinite(btc_usd) else 'n/a'} | **BTC AUD est**: {btc_aud if np.isfinite(btc_aud) else 'n/a'}")
with col_opts:
    drift_on = st.checkbox("Apply drift model to stale quotes", value=True)
    drift_bps_per_hour = st.number_input("Drift (bps/hour)", 0.0, 20.0, 2.0, step=0.5)
    fee_bps = st.number_input("Trading fee (bps)", 0.0, 200.0, 7.0, step=1.0)
    slippage_bps = st.number_input("Slippage (bps)", 0.0, 200.0, 5.0, step=1.0)

tickers = [p["ticker"] for p in new_positions if str(p.get("ticker","")).strip()]
unique_tickers = sorted(set([t for t in tickers if t]))
prices = {}
timestamps = {}
for tck in unique_tickers:
    p, ts = get_last_price(tck)
    if drift_on and isinstance(ts, str) and "UTC" in ts:
        try:
            from datetime import datetime as dt
            ts_dt = dt.strptime(ts.replace(" UTC",""), "%Y-%m-%d %H:%M")
            hours = (dt.utcnow() - ts_dt).total_seconds()/3600.0
            p = drift_adjust(p, hours, drift_bps_per_hour)
        except Exception:
            pass
    prices[tck] = p
    timestamps[tck] = ts

rows = []
for p in new_positions:
    tck = p.get("ticker","").strip()
    if not tck:
        continue
    qty = float(p.get("qty",0) or 0)
    cost = float(p.get("cost_aud",0) or 0)
    typ = p.get("type","")
    px = prices.get(tck, float('nan'))
    mkt = qty * (px if np.isfinite(px) else 0.0)
    rows.append({"Ticker": tck, "Type": typ, "Qty": qty, "Price": px, "MarketValue": mkt, "Cost_AUD": cost})
val_df = pd.DataFrame(rows)
total_mv = float(val_df["MarketValue"].sum()) if not val_df.empty else 0.0
st.dataframe(val_df.fillna("n/a"), use_container_width=True)
st.metric("Portfolio Market Value", fmt_money(total_mv))

st.subheader("🧭 Allocation")
if not val_df.empty:
    alloc = val_df.groupby("Ticker")["MarketValue"].sum()
    alloc_pct = (alloc / max(total_mv, 1)) * 100.0
    alloc_df = pd.DataFrame({"Ticker": alloc.index, "Weight %": alloc_pct.values}).sort_values("Weight %", ascending=False)
    st.dataframe(alloc_df, use_container_width=True)
    fig, ax = plt.subplots()
    ax.pie(alloc_pct.values, labels=alloc_pct.index, autopct="%1.1f%%")
    ax.set_title("Portfolio Allocation")
    st.pyplot(fig)

st.subheader("🔧 Rebalance Advisor")
tgt_series = pd.Series(new_targets, dtype=float)
cur_series = alloc_pct if 'alloc_pct' in locals() else pd.Series(dtype=float)
combined = pd.DataFrame({"Target %": tgt_series, "Current %": cur_series}).fillna(0.0)
combined["Diff %"] = combined["Target %"] - combined["Current %"]
st.dataframe(combined.sort_values("Diff %", ascending=True), use_container_width=True)

st.subheader("🧮 Greedy Planner (fees, slippage, lot sizes, FX)")
cash_aud = st.number_input("Available cash (AUD)", 0.0, 1e9, 10000.0, step=100.0)
lot_size = st.number_input("Lot size (min units per trade)", 1, 10000, 1, step=1)
if st.button("Plan Buys"):
    import math
    desired_weights = combined["Target %"] / combined["Target %"].sum()
    desired_values = desired_weights * (total_mv + cash_aud)
    current_values = (cur_series / 100.0) * total_mv
    gap = (desired_values - current_values).fillna(0.0)
    plan = []
    prices_vec = {t: prices.get(t, float('nan')) for t in desired_weights.index}
    fees = fee_bps / 10000.0
    slip = slippage_bps / 10000.0
    remaining_cash = cash_aud
    safety = 20000
    while remaining_cash > 0 and safety > 0:
        safety -= 1
        if gap.empty or gap.max() <= 0:
            break
        tgt = gap.sort_values(ascending=False).index[0]
        px = prices_vec.get(tgt, float('nan'))
        if not np.isfinite(px) or px <= 0:
            gap[tgt] = 0
            continue
        unit_cost = px * (1 + fees + slip)
        qty = max(0, int(min(remaining_cash // unit_cost, math.ceil((gap[tgt] / unit_cost)))))
        qty = (qty // lot_size) * lot_size
        if qty <= 0:
            break
        spend = qty * unit_cost
        remaining_cash -= spend
        gap[tgt] -= spend
        plan.append({"Ticker": tgt, "Qty": qty, "EstPrice": px, "EstSpend": spend})
    plan_df = pd.DataFrame(plan)
    if plan_df.empty:
        st.info("No feasible buys with current cash/lot size/fees.")
    else:
        st.dataframe(plan_df, use_container_width=True)
        st.metric("Cash leftover", fmt_money(remaining_cash))
        csv_buf = io.StringIO(); plan_df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download plan CSV", data=csv_buf.getvalue(), file_name="buy_plan.csv", mime="text/csv")

st.subheader("🚨 Alerts")
alert_rows = []
ma_window = st.number_input("MA window (days)", 5, 200, 20)
for tck in sorted(set([p.get("ticker","") for p in new_positions if p.get("ticker","")])):
    df = safe_yf_download(tck, period="90d", interval="1d")
    if df is None or df.empty:
        continue
    last = float(df["Close"].dropna().iloc[-1])
    ma = moving_average(df["Close"], ma_window)
    cur_w = float(alloc_pct.get(tck, 0.0)) if 'alloc_pct' in locals() else 0.0
    tgt_w = new_targets.get(tck, 0.0)
    alert_rows.append({"Ticker": tck, "Last": last, f"MA{ma_window}": ma, "AboveMA": last>ma if np.isfinite(ma) else None, "AllocDrift %": cur_w - tgt_w})
if alert_rows:
    st.dataframe(pd.DataFrame(alert_rows), use_container_width=True)

st.subheader("🗂️ Import / Export")
cE1, cE2 = st.columns(2)
with cE1:
    if st.button("Export JSON"):
        buf = io.StringIO(); json.dump(holdings, buf, indent=2)
        st.download_button("⬇️ Download holdings.json", data=buf.getvalue(), file_name="holdings.json", mime="application/json")
with cE2:
    up = st.file_uploader("Import holdings.json", type=["json"])
    if up is not None:
        try:
            new = json.load(up)
            st.session_state["holdings"] = new
            st.success("Imported into session. Use sidebar 'Save Holdings' to persist.")
        except Exception as e:
            st.error(f"Import failed: {e}")
