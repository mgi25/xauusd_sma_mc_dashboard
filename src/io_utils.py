from __future__ import annotations
import io
import pandas as pd

def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}") from e
    if df.empty:
        raise ValueError("CSV is empty.")
    df.columns = [c.strip() for c in df.columns]
    return df

def coerce_date_close(df: pd.DataFrame, date_col: str, close_col: str) -> pd.DataFrame:
    out = df[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if len(out) < 60:
        raise ValueError("Need at least ~120 rows (recommended 252+).")
    return out
