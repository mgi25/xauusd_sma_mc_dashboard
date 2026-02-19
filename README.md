# XAUUSD SMA Strategy + Monte Carlo Robustness Dashboard (Streamlit)

## Idea (simple + marks-friendly)
1) Build a simple SMA strategy (e.g., SMA20 vs SMA50) on XAUUSD.
2) Backtest on historical prices to get strategy returns + equity curve.
3) Monte Carlo (bootstrap) robustness:
   - resample historical strategy returns many times
   - create many alternative equity curves
   - measure distribution of outcomes (final return, drawdown, Sharpe)

This answers: "Is the strategy stable, or did it look good by chance on one history?"

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Data
Upload your MT5 CSV. In the UI, select the date and close columns.
Sample file: `data/sample_xauusd_mt5.csv`
