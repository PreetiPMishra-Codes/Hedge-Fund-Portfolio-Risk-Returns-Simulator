import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

def calculate_rolling_volatility(portfolio_ret:pd.Series,window:int=30)->pd.Series:
    """
    Rolling 30-day annualized volatility: 'how risky has the portfolio
    been over the last month', recalculated for every day.
    This is our ML target — the metric whose trend we'll project.
    """
    rolling_std = portfolio_ret.rolling(window=window).std()
    rolling_vol_annualized = rolling_std * np.sqrt(252)
 
    # First `window - 1` values are NaN (not enough days yet to fill the window)
    rolling_vol_annualized = rolling_vol_annualized.dropna()
 
    return rolling_vol_annualized

def project_volatility_trend(rolling_vol: pd.Series, days_forward: int = 30):
    """
    Fits a Ridge regression on (day_index -> rolling_volatility) and
    extrapolates `days_forward` business days into the future.
 
    Ridge, not plain Linear Regression: it adds an L2 penalty that
    prevents the model from overreacting to noisy recent volatility
    spikes, giving a smoother, more conservative trendline.
    """
    # X = time step (0, 1, 2, ... n-1), reshaped to a column vector
    # because scikit-learn always expects 2D input: (n_samples, n_features)
    X = np.arange(len(rolling_vol)).reshape(-1, 1)
    y = rolling_vol.values
 
    model = Ridge(alpha=1.0)
    model.fit(X, y)
 
    # Build future X values: the next `days_forward` time steps
    future_X = np.arange(len(rolling_vol), len(rolling_vol) + days_forward).reshape(-1, 1)
    future_predictions = model.predict(future_X)
 
    # Build a matching future date index (business days only, like the market)
    last_date = rolling_vol.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=days_forward + 1)[1:]
 
    forecast = pd.Series(future_predictions, index=future_dates, name="Projected Volatility")
 
    return model, forecast
