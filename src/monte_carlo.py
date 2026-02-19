from __future__ import annotations
import numpy as np
import pandas as pd
from .backtest import max_drawdown, sharpe

def iid_bootstrap_paths(returns: np.ndarray, n_sims: int, horizon: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    r = np.asarray(returns, dtype=float)
    idx = rng.integers(0, len(r), size=(horizon, n_sims))
    return r[idx]

def block_bootstrap_paths(returns: np.ndarray, n_sims: int, horizon: int, block: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    r = np.asarray(returns, dtype=float)
    n = len(r)
    block = int(block)
    k = int(np.ceil(horizon / block))
    starts = rng.integers(0, max(1, n - block + 1), size=(k, n_sims))
    out = np.zeros((k * block, n_sims), dtype=float)
    for i in range(k):
        for s in range(n_sims):
            st = starts[i, s]
            out[i*block:(i+1)*block, s] = r[st:st+block]
    return out[:horizon, :]

def returns_to_equity(sim_ret: np.ndarray) -> np.ndarray:
    sim_ret = np.asarray(sim_ret, dtype=float)
    return np.vstack([np.ones((1, sim_ret.shape[1])), np.cumprod(1.0 + sim_ret, axis=0)])

def summarize_sims(sim_ret: np.ndarray, periods_per_year: int = 252) -> pd.DataFrame:
    eq = returns_to_equity(sim_ret)
    finals = eq[-1, :] - 1.0
    mdds = np.array([max_drawdown(eq[:, i]) for i in range(eq.shape[1])], dtype=float)
    shps = np.array([sharpe(sim_ret[:, i], periods_per_year) for i in range(eq.shape[1])], dtype=float)
    return pd.DataFrame({"Final_Return": finals, "Max_Drawdown": mdds, "Sharpe": shps})

def percentile_bands(eq: np.ndarray, qs=(0.05, 0.5, 0.95)) -> pd.DataFrame:
    out = {}
    for q in qs:
        out[f"q{int(q*100)}"] = np.quantile(eq, q, axis=1)
    return pd.DataFrame(out)
