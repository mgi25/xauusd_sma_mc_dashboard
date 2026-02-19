from __future__ import annotations
import numpy as np
import pandas as pd

def compute_backtest(df_sig: pd.DataFrame, cost_bps: float = 0.0) -> pd.DataFrame:
    out = df_sig.copy()
    out["ret"] = out["close"].pct_change()
    out["pos_prev"] = out["position"].shift(1)
    out["strat_ret_gross"] = out["pos_prev"] * out["ret"]

    cost_bps = float(cost_bps)
    out["turnover"] = (out["position"] - out["position"].shift(1)).abs()
    out["cost"] = (cost_bps / 10000.0) * out["turnover"]
    out["strat_ret"] = out["strat_ret_gross"] - out["cost"]

    out = out.dropna(subset=["ret", "strat_ret", "sma_short", "sma_long"]).reset_index(drop=True)

    out["equity"] = (1.0 + out["strat_ret"]).cumprod()
    out["bh_equity"] = (1.0 + out["ret"]).cumprod()
    return out

def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(dd.min())

def sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 10:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float((mu / sd) * np.sqrt(periods_per_year))

def summarize_bt(bt: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    total_return = float(bt["equity"].iloc[-1] - 1.0)
    bh_return = float(bt["bh_equity"].iloc[-1] - 1.0)
    mdd = max_drawdown(bt["equity"].to_numpy())
    mdd_bh = max_drawdown(bt["bh_equity"].to_numpy())
    shp = sharpe(bt["strat_ret"].to_numpy(), periods_per_year)
    shp_bh = sharpe(bt["ret"].to_numpy(), periods_per_year)
    win_rate = float((bt["strat_ret"] > 0).mean())

    return pd.DataFrame([{
        "Total_Return": total_return,
        "BuyHold_Return": bh_return,
        "Max_Drawdown": mdd,
        "BuyHold_Max_Drawdown": mdd_bh,
        "Sharpe": shp,
        "BuyHold_Sharpe": shp_bh,
        "Win_Rate": win_rate,
        "Bars": int(len(bt)),
    }])
