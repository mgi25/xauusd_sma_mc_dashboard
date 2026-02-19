# MT5 -> Download XAUUSD data to CSV (run on your MT5 PC)
# pip install MetaTrader5 pandas

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

SYMBOL = "XAUUSD"   # try XAUUSDm if needed
TIMEFRAME = mt5.TIMEFRAME_D1
YEARS_BACK = 5
OUT_CSV = f"{SYMBOL}_D1_{YEARS_BACK}Y.csv"

if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

if not mt5.symbol_select(SYMBOL, True):
    mt5.shutdown()
    raise RuntimeError(f"symbol_select failed for {SYMBOL}. Try SYMBOL='XAUUSDm'.")

end = datetime.now()
start = end - timedelta(days=int(YEARS_BACK * 365.25))

rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start, end)
if rates is None or len(rates) == 0:
    mt5.shutdown()
    raise RuntimeError(f"No data returned. Error: {mt5.last_error()}")

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")
df = df.rename(columns={"time": "date"})
df = df[["date", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
df.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV, "rows:", len(df))
mt5.shutdown()
