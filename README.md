
# Investment Dashboard – Collated Final (with Greedy Planner)

Single-file Streamlit app merging the collated app + greedy planner.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Google Sheets (optional)
Create `.streamlit/secrets.toml`:
```toml
sheet_id = "YOUR_GOOGLE_SHEET_ID"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "...@...iam.gserviceaccount.com"
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"
```
