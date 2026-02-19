# Download XAUUSD (Gold) data from MetaTrader 5 and save as CSV
# Requirements:
#   pip install MetaTrader5 pandas
# Run this on the same PC where MT5 terminal is installed and you are logged in.

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# ---------- SETTINGS ----------
SYMBOL_CANDIDATES = ["XAUUSD!", "XAUUSDm"]   # try both (brokers often use suffix like "m")
TIMEFRAME = mt5.TIMEFRAME_D1               # change to mt5.TIMEFRAME_H1 / M15 / M5 etc.
YEARS_BACK = 5                             # for H1, use smaller like 1
OUT_CSV = "XAUUSD_export.csv"
# -----------------------------

def pick_symbol(candidates):
    for sym in candidates:
        if mt5.symbol_select(sym, True):
            info = mt5.symbol_info(sym)
            if info is not None and info.visible:
                return sym
    return None

def main():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    symbol = pick_symbol(SYMBOL_CANDIDATES)
    if symbol is None:
        mt5.shutdown()
        raise RuntimeError("Could not select XAUUSD symbol. Try adding your broker symbol name to SYMBOL_CANDIDATES.")

    end = datetime.now()
    start = end - timedelta(days=int(YEARS_BACK * 365.25))

    rates = mt5.copy_rates_range(symbol, TIMEFRAME, start, end)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"No data returned for {symbol}. MT5 error: {err}")

    df = pd.DataFrame(rates)
    df["date"] = pd.to_datetime(df["time"], unit="s")
    df = df.drop(columns=["time"])

    # Reorder columns (keeps extra ones if present)
    preferred = ["date", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows to: {OUT_CSV}  (symbol={symbol})")

    mt5.shutdown()

if __name__ == "__main__":
    main()
