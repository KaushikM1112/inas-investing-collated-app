# Inas Investing — Collated Streamlit App

A single-file Streamlit app that combines the key features from your Investment project:

- Live price updates (ETFs, crypto, gold) via `yfinance`
- Portfolio allocation chart on Home
- Gold tracking (live ticker + manual entry)
- Rebalance advisor (target vs current; qty suggestions)
- Alerts system (price and allocation thresholds)
- Time-lag compensation in Live Tracker (simple 10‑minute linear extrapolation when data is stale)
- Optional Google Sheets persistence (read/write holdings)
- Streamlit Cloud ready

## Quickstart (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push these files to a GitHub repo.
2. Create a new Streamlit Cloud app targeting `app.py`.
3. Add **App secrets** from `.streamlit/secrets.toml.template`.
4. Deploy.

## Configure Google Sheets (Optional)

1. Create a Google Sheet with headers `Ticker | Quantity | CostBasis_AUD | Notes`.
2. Create a Google Cloud **Service Account**, enable **Google Sheets API** and **Google Drive API**, and generate a JSON key.
3. Share the Sheet with the service account email as **Editor**.
4. Copy credentials into Streamlit Cloud **Secrets** using the template below.

## Files

- `app.py` — the main Streamlit app (single-file)
- `requirements.txt` — dependencies
- `.streamlit/secrets.toml.template` — secrets template for deployment
- `holdings.json` — example high-growth holdings with BTC & GOLD
- `README.md` — this guide
