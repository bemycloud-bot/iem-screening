# Deploy to Streamlit Community Cloud

## 1) Push project to GitHub
- Ensure these files are in your repo root:
  - `streamlit_app.py`
  - `predict_disease_probabilities.py`
  - `requirements.txt`
  - `comparisons/best_smote_balanced_lr.joblib`
  - `comparisons/class_mapping.csv`

## 2) Create the app
- Go to https://share.streamlit.io
- Click **New app**
- Select your repository and branch
- Main file path: `streamlit_app.py`
- Deploy

## 3) Set webhook secret (important)
- In Streamlit app settings, open **Secrets**
- Add:

```toml
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
```

- Save and restart app

## 4) Verify app behavior
- Upload a CSV and run prediction
- Use **Send Suspected Cases to Discord**
- Discord message should include:
  - Patient-only suspected summary (LC/HC/IS excluded)
  - PNG attachment for suspected report (or HTML fallback if PNG generation fails)

## 5) Notes for privacy
- Do not upload identifiable patient data to public cloud unless policy allows.
- Keep webhook only in Secrets (not in code).
