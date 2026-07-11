# Calculate daily percentage returns
def calculate_daily_returns(prices:pd.DataFrame)->pd.DataFrame:
    """
    Converts a price series into a daily returns series.
    """
    daily_returns = prices.pct_change()

    # The very first row will always be NaN (no "previous day" to
    # compare against), so we drop it.
    daily_returns = daily_returns.dropna()

    return daily_returns
