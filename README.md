# Linear Regression for Predicting AAPL Daily Returns

Using OLS linear regression to predict AAPL's next-day return from lagged returns of correlated stocks and market indices. Trained on 2020-2023, tested out-of-sample on 2024.

## Why This Project

I wanted to see what happens when applying textbook linear regression to a problem where it probably shouldn't work that well. Daily stock returns are notoriously hard to predict, so instead of chasing a high R² the focus here is on doing the full workflow correctly and being honest about the results.

## What I Did

The model uses lagged daily returns from AAPL, AMZN, MSFT, QQQ, and S&P 500 along with AAPL's 5-day rolling volatility and rolling mean return as features. Everything is in returns space rather than price levels to avoid the spurious regression problem where non-stationary prices give a fake R² of 0.99.

I started with all 7 features, ran the full model, then dropped everything that wasn't significant at the 5% level. The reduced model kept SP500_lag1 and AAPL_vol_5d as the only significant predictors.

## Assumption Testing

All 5 OLS assumptions were formally tested rather than just eyeballed.

| Assumption | Test | Result |
|---|---|---|
| Linearity | Scatter plots | Weak but no nonlinearity |
| Homoscedasticity | Breusch-Pagan | Violated (p ≈ 0.00) |
| No Multicollinearity | VIF | Passed (VIF ≈ 1.0) |
| Normality of Residuals | Jarque-Bera | Violated (fat tails) |
| No Autocorrelation | Durbin-Watson | Passed (DW ≈ 2.0) |

Two of the five assumptions are violated, which is typical for financial returns data. Heteroscedasticity comes from volatility clustering, and fat tails are a known property of return distributions.

## Results

| Metric | Model | Naive Baseline |
|---|---|---|
| RMSE (%) | ~1.98 | ~1.99 |
| MAE (%) | ~1.40 | ~1.41 |

The model barely beats (or doesn't beat) a naive baseline that just predicts the mean training return every day. Out-of-sample R² is effectively zero. Directional accuracy is close to a coin flip.

This is the expected outcome. If a simple linear model on lagged returns could reliably predict daily stock moves, the signal would get arbitraged away almost immediately.

## Data

- **Ticker:** AAPL with AMZN, MSFT, QQQ, S&P 500 as cross-asset features
- **Training:** 2020-01-01 to 2023-12-31 (1001 observations)
- **Testing:** 2024-01-01 to 2024-12-31 (251 observations)
- **Source:** Yahoo Finance via `yfinance`

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook linear_regression_aapl_returns.ipynb
```

## Built With

- Python 3.10+
- [statsmodels](https://www.statsmodels.org/) for OLS regression and diagnostic tests
- [scikit-learn](https://scikit-learn.org/) for evaluation metrics
- [yfinance](https://github.com/ranaroussi/yfinance) for market data
- NumPy, pandas, matplotlib, seaborn
