from __future__ import annotations
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def plot_price_sma(df_sig: pd.DataFrame):
    fig = plt.figure()
    plt.plot(df_sig["date"], df_sig["close"], label="Close")
    plt.plot(df_sig["date"], df_sig["sma_short"], label="SMA short")
    plt.plot(df_sig["date"], df_sig["sma_long"], label="SMA long")
    plt.title("XAUUSD Close with SMAs")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    return fig

def plot_equity(bt: pd.DataFrame):
    fig = plt.figure()
    plt.plot(bt["date"], bt["equity"], label="Strategy")
    plt.plot(bt["date"], bt["bh_equity"], label="Buy & Hold")
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    return fig

def plot_hist(series: np.ndarray, title: str, xlabel: str):
    fig = plt.figure()
    plt.hist(series, bins=60)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    return fig

def plot_fan(bands: pd.DataFrame, title: str = "Monte Carlo Equity Bands"):
    fig = plt.figure()
    x = np.arange(len(bands))
    plt.plot(x, bands["q50"], label="Median")
    plt.plot(x, bands["q5"], label="5th pct")
    plt.plot(x, bands["q95"], label="95th pct")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    return fig
