from __future__ import annotations
import numpy as np
import pandas as pd

def add_vol_regime(bt: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    out = bt.copy()
    out["roll_vol"] = out["ret"].rolling(int(vol_window)).std()
    thr = out["roll_vol"].median()
    out["vol_regime"] = np.where(out["roll_vol"] >= thr, "High Vol", "Low Vol")
    return out

def regime_summary(bt_reg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, d in bt_reg.groupby("vol_regime"):
        rows.append({
            "Regime": g,
            "Bars": int(len(d)),
            "Avg_Strategy_Return": float(d["strat_ret"].mean()),
            "Volatility": float(d["strat_ret"].std(ddof=1)),
            "Win_Rate": float((d["strat_ret"] > 0).mean()),
        })
    return pd.DataFrame(rows)
