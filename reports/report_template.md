# Mini Project Report: Monte Carlo Robustness of SMA Strategy on XAUUSD

## 1. Problem Description
We simulate and evaluate a simple Moving Average (SMA) trading strategy on gold (XAUUSD). Because backtesting uses only one historical sequence, we apply Monte Carlo simulation to test how strategy performance changes under random variations.

## 2. Why Simulation is Useful
Monte Carlo creates many alternative outcomes by resampling historical strategy returns. It estimates:
- distribution of final returns
- distribution of max drawdown (risk)
- stability of Sharpe ratio
This helps decide if results are robust or likely due to chance.

## 3. Assumptions
- Historical returns represent realistic future behavior.
- Strategy rules remain fixed (same SMA windows).
- Bootstrap resampling generates simulated scenarios:
  - IID bootstrap: returns sampled independently
  - Block bootstrap: returns sampled in blocks to preserve dependence
- Transaction cost is modeled as a fixed bps cost per position change (optional).
- Slippage, margin, and liquidity constraints are ignored (simplification).

## 4. Model Design (Algorithm)
1) Load XAUUSD close price data.
2) Compute SMA_short and SMA_long.
3) Generate position:
   - Long/Short: +1 if SMA_short > SMA_long else -1
   - (or Long only: 1 else 0)
4) Compute market returns and strategy returns (with optional costs).
5) Build equity curve.
6) Compute metrics: total return, max drawdown, Sharpe, win rate.
7) Monte Carlo:
   - Resample strategy returns many times.
   - Build simulated equity curves.
   - Compute distributions for return/drawdown/Sharpe.
8) Compare real backtest to Monte Carlo percentiles.
9) Test conditions:
   - Split by volatility (High vs Low volatility) and compare average returns/win rate.

## 5. Implementation
Python: NumPy, Pandas, Matplotlib, Streamlit UI.

## 6. Results and Analysis
Add screenshots:
- Price with SMA lines
- Equity curve
- Monte Carlo histogram of final returns
- Monte Carlo histogram of max drawdown
- Equity percentile bands (5th/50th/95th)
- Tables: Backtest summary, volatility regime summary

Interpretation:
- Wide distributions mean unstable performance.
- If real backtest is near the median, strategy is average.
- If real backtest is near top percentiles, it is stronger than typical outcomes.
- Drawdown distributions show worst-case risk.

## 7. Conclusion
Learning:
- Backtest results can be misleading without robustness testing.
- Monte Carlo quantifies uncertainty and risk.
Improvements:
- Add realistic slippage/spread, and trade sizing.
- Walk-forward testing.
- Compare multiple SMA settings and timeframes.
