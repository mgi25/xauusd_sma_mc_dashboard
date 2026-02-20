from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
import time

from src.io_utils import load_csv_bytes, coerce_date_close
from src.strategy import sma_signals
from src.backtest import compute_backtest, summarize_bt
from src.monte_carlo import iid_bootstrap_paths, block_bootstrap_paths, returns_to_equity, summarize_sims, percentile_bands
from src.regimes import add_vol_regime, regime_summary
from src.plots import fig_to_png_bytes, plot_price_sma, plot_equity, plot_hist, plot_fan
from src.export import df_to_csv_bytes, dfs_to_excel_bytes, bundle_zip


def live_simulation_panel(sim_eq: np.ndarray, frame_delay_ms: int = 40, sample_paths: int = 12) -> None:
    """Animate a lightweight Monte Carlo replay for presentation/demo use."""
    sim_eq = np.asarray(sim_eq, dtype=float)
    n_steps, n_sims = sim_eq.shape
    sample_paths = max(3, min(int(sample_paths), n_sims))

    rng = np.random.default_rng(42)
    chosen = rng.choice(n_sims, size=sample_paths, replace=False)
    chosen_paths = sim_eq[:, chosen]

    metric_a, metric_b, metric_c = st.columns(3)
    chart_slot = st.empty()
    progress = st.progress(0)

    frame_idx = np.unique(np.linspace(2, n_steps, min(120, n_steps - 1), dtype=int))
    for i, t in enumerate(frame_idx):
        current = sim_eq[t - 1, :] - 1.0
        p_profit = float((current > 0).mean() * 100.0)

        metric_a.metric("Step", f"{t - 1}/{n_steps - 1}")
        metric_b.metric("Median return", f"{np.median(current):.2%}")
        metric_c.metric("P(final > 0)", f"{p_profit:.1f}%")

        chart_df = pd.DataFrame(
            {
                "P05": np.quantile(sim_eq[:t, :], 0.05, axis=1),
                "Median": np.quantile(sim_eq[:t, :], 0.50, axis=1),
                "P95": np.quantile(sim_eq[:t, :], 0.95, axis=1),
            },
            index=np.arange(t),
        )

        for j in range(sample_paths):
            chart_df[f"Path {j + 1}"] = chosen_paths[:t, j]

        chart_slot.line_chart(chart_df, use_container_width=True)
        progress.progress(int((i + 1) / len(frame_idx) * 100))

        if frame_delay_ms > 0:
            time.sleep(frame_delay_ms / 1000.0)

    st.caption("Replay complete — blue fan lines show confidence cone, sample paths show scenario variety.")

st.set_page_config(page_title="XAUUSD SMA + Monte Carlo", layout="wide")
st.title("XAUUSD SMA Strategy + Monte Carlo Robustness Test")

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.header("Strategy")
    short_w = st.number_input("SMA short window", min_value=2, value=20, step=1)
    long_w  = st.number_input("SMA long window", min_value=3, value=50, step=1)
    mode = st.selectbox("Trading mode", options=["Long only", "Long/Short"], index=1)
    cost_bps = st.number_input("Transaction cost (bps per position change)", min_value=0.0, value=0.0, step=0.1)

    st.header("Monte Carlo")
    mc_method = st.selectbox("Bootstrap method", options=["IID bootstrap", "Block bootstrap"], index=0)
    block = st.number_input("Block size (Block bootstrap)", min_value=2, value=5, step=1)
    n_sims = st.slider("Simulations", min_value=500, max_value=50000, value=5000, step=500)
    horizon_choice = st.selectbox("Simulation horizon", options=["Same as backtest length", "Custom"], index=0)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=7, step=1)
    st.markdown("---")
    st.header("Live demo")
    show_live = st.checkbox("Show live simulation playback", value=True)
    playback_ms = st.slider("Playback speed (ms/frame)", min_value=0, max_value=250, value=35, step=5)
    playback_paths = st.slider("Animated sample paths", min_value=3, max_value=20, value=10, step=1)

    run_btn = st.button("Run", type="primary", use_container_width=True)

st.markdown("""
**What this does:**  
1) Backtests an SMA strategy on gold (XAUUSD).  
2) Uses Monte Carlo bootstrap to generate many randomized outcomes from the strategy returns.  
3) Shows how stable (or unstable) the performance is.
""")

if uploaded is None:
    st.info("Upload your CSV to start. (Sample file is in `data/sample_xauusd_mt5.csv` in this project.)")
    st.stop()

df_raw = load_csv_bytes(uploaded.getvalue())
cols = list(df_raw.columns)
date_col = st.selectbox("Date column", options=cols, index=cols.index("date") if "date" in cols else 0)
close_col = st.selectbox("Close column", options=cols, index=cols.index("close") if "close" in cols else min(1, len(cols)-1))

df = coerce_date_close(df_raw, date_col, close_col)
st.subheader("Data preview")
st.dataframe(df.tail(10), use_container_width=True)

if not run_btn:
    st.warning("Set parameters and click Run.")
    st.stop()

df_sig = sma_signals(df, int(short_w), int(long_w), mode=mode)
bt = compute_backtest(df_sig, cost_bps=float(cost_bps))
summary = summarize_bt(bt)

st.subheader("Backtest Summary")
st.dataframe(summary, use_container_width=True)

bt_reg = add_vol_regime(bt, vol_window=20)
reg_sum = regime_summary(bt_reg)
st.subheader("Different Conditions: High vs Low Volatility")
st.dataframe(reg_sum, use_container_width=True)

c1, c2 = st.columns([1,1])
with c1:
    st.image(fig_to_png_bytes(plot_price_sma(df_sig.dropna())), caption="Price with SMAs", use_container_width=True)
with c2:
    st.image(fig_to_png_bytes(plot_equity(bt)), caption="Equity curve", use_container_width=True)

ret = bt["strat_ret"].to_numpy(dtype=float)

if horizon_choice == "Same as backtest length":
    horizon = len(ret)
else:
    horizon = st.slider("Custom horizon (bars)", min_value=50, max_value=min(2000, len(ret)), value=min(252, len(ret)), step=10)

if mc_method == "IID bootstrap":
    sim_ret = iid_bootstrap_paths(ret, int(n_sims), int(horizon), int(seed))
else:
    sim_ret = block_bootstrap_paths(ret, int(n_sims), int(horizon), int(block), int(seed))

sim_eq = returns_to_equity(sim_ret)
sim_metrics = summarize_sims(sim_ret)
bands = percentile_bands(sim_eq, qs=(0.05, 0.5, 0.95))

st.subheader("Monte Carlo Robustness")
d1, d2, d3 = st.columns([1,1,1])
with d1:
    st.image(fig_to_png_bytes(plot_hist(sim_metrics["Final_Return"].to_numpy(), "Final Return Distribution", "Final return")),
             caption="Final returns", use_container_width=True)
with d2:
    st.image(fig_to_png_bytes(plot_hist(sim_metrics["Max_Drawdown"].to_numpy(), "Max Drawdown Distribution", "Max drawdown")),
             caption="Max drawdowns", use_container_width=True)
with d3:
    st.image(fig_to_png_bytes(plot_fan(bands)), caption="Equity percentile bands", use_container_width=True)

if show_live:
    st.subheader("Live Monte Carlo Playback")
    st.write("A short animated replay of how return outcomes evolve over time.")
    live_simulation_panel(sim_eq, frame_delay_ms=int(playback_ms), sample_paths=int(playback_paths))

real_final = float(bt["equity"].iloc[-1] - 1.0)
p5, p50, p95 = [float(np.quantile(sim_metrics["Final_Return"], q)) for q in (0.05, 0.50, 0.95)]
st.markdown(f"**Real backtest final return:** `{real_final:.3f}`  \n**Monte Carlo final return percentiles:** 5%=`{p5:.3f}`, 50%=`{p50:.3f}`, 95%=`{p95:.3f}`")

st.subheader("Downloads")
files = {
    "tables/backtest_summary.csv": df_to_csv_bytes(summary),
    "tables/regime_summary.csv": df_to_csv_bytes(reg_sum),
    "tables/mc_metrics.csv": df_to_csv_bytes(sim_metrics),
    "tables/mc_bands.csv": df_to_csv_bytes(bands.reset_index().rename(columns={"index":"step"})),
}

excel_bytes = dfs_to_excel_bytes({
    "Backtest_Summary": summary,
    "Regimes": reg_sum,
    "MC_Metrics": sim_metrics,
    "MC_Bands": bands.reset_index().rename(columns={"index":"step"}),
})
files["tables/all_tables.xlsx"] = excel_bytes

files["plots/price_sma.png"] = fig_to_png_bytes(plot_price_sma(df_sig.dropna()))
files["plots/equity_curve.png"] = fig_to_png_bytes(plot_equity(bt))
files["plots/final_return_hist.png"] = fig_to_png_bytes(plot_hist(sim_metrics["Final_Return"].to_numpy(), "Final Return Distribution", "Final return"))
files["plots/max_drawdown_hist.png"] = fig_to_png_bytes(plot_hist(sim_metrics["Max_Drawdown"].to_numpy(), "Max Drawdown Distribution", "Max drawdown"))
files["plots/equity_bands.png"] = fig_to_png_bytes(plot_fan(bands))

zip_bytes = bundle_zip(files)

e1, e2, e3 = st.columns([1,1,1])
with e1:
    st.download_button("Backtest Summary CSV", data=files["tables/backtest_summary.csv"], file_name="xauusd_sma_backtest_summary.csv", mime="text/csv", use_container_width=True)
with e2:
    st.download_button("All Tables (Excel)", data=excel_bytes, file_name="xauusd_sma_mc_tables.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
with e3:
    st.download_button("Everything (ZIP outputs)", data=zip_bytes, file_name="xauusd_sma_mc_outputs.zip", mime="application/zip", use_container_width=True)

st.success("Done. Use screenshots of summary tables + graphs in your 4–5 page report.")
