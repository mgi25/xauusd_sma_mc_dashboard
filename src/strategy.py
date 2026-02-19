from __future__ import annotations
import pandas as pd

def sma_signals(df: pd.DataFrame, short_window: int, long_window: int, mode: str) -> pd.DataFrame:
    if short_window >= long_window:
        raise ValueError("Short window must be < long window.")
    out = df.copy()
    out["sma_short"] = out["close"].rolling(int(short_window)).mean()
    out["sma_long"]  = out["close"].rolling(int(long_window)).mean()

    cond = out["sma_short"] > out["sma_long"]
    mode = mode.strip()
    if mode == "Long only":
        out["position"] = cond.astype(int)           # 1 or 0
    elif mode == "Long/Short":
        out["position"] = cond.astype(int).replace({0: -1})  # 1 or -1
    else:
        raise ValueError("Unknown mode")
    return out
