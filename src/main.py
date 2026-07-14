# ----------------------------------------------------------------------
# MAIN execution block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Fetching data for: {TICKERS}")
    prices = fetch_price_data(TICKERS, START_DATE, END_DATE)

    print("\nRaw price data (first 5 rows):")
    print(prices.head())

    prices_clean = clean_price_data(prices)
    print(f"\nMissing values after cleaning:\n{prices_clean.isna().sum()}")

    returns = calculate_daily_returns(prices_clean)
    print("\nDaily returns (first 5 rows):")
    print(returns.head())

    print("\nSummary statistics of daily returns:")
    print(returns.describe())

    prices_clean.to_csv("clean_prices.csv")
    returns.to_csv("daily_returns.csv")
    print("\nSaved clean_prices.csv and daily_returns.csv")


   #equal weights, we r taking 25 % each in same order as tickers
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    print(f"\nWeights: {dict(zip(returns.columns, weights))}")
    print(f"Weights sum to: {weights.sum()}  (must equal 1.0)")
 
    portfolio_returns = calculate_portfolio_returns(returns, weights)
    print("\nPortfolio daily returns (first 5 rows):")
    print(portfolio_returns.head())
 
    portfolio_variance = calculate_portfolio_variance(returns, weights)
    portfolio_std = np.sqrt(portfolio_variance)
    print(f"\nPortfolio daily variance: {portfolio_variance:.8f}")
    print(f"Portfolio daily std dev (volatility): {portfolio_std:.6f}")
    print(f"Portfolio annualized volatility: {portfolio_std * np.sqrt(252):.4%}")


    var_95 = calculate_historical_var(portfolio_returns, confidence_level=0.95)
    var_99 = calculate_historical_var(portfolio_returns, confidence_level=0.99)
 
    print(f"\nHistorical VaR (95% confidence): {var_95:.4%}")
    print(f"Historical VaR (99% confidence): {var_99:.4%}")
 
    # Translate into dollar terms for a hypothetical $1,000,000 portfolio
    portfolio_value = 1_000_000
    print(f"\nOn a ${portfolio_value:,} portfolio:")
    print(f"  95% VaR -> On 95% of days, losses should not exceed "
          f"${abs(var_95) * portfolio_value:,.2f}")
    print(f"  99% VaR -> On 99% of days, losses should not exceed "
          f"${abs(var_99) * portfolio_value:,.2f}")

    
    rolling_vol = calculate_rolling_volatility(portfolio_returns, window=30)
    print("\nRolling 30-day annualized volatility (last 5 values):")
    print(rolling_vol.tail())
 
    model, forecast = project_volatility_trend(rolling_vol, days_forward=30)
    print(f"\nRidge model coefficient (slope): {model.coef_[0]:.8f}")
    print(f"Ridge model intercept: {model.intercept_:.6f}")
    trend_direction = "INCREASING" if model.coef_[0] > 0 else "DECREASING"
    print(f"Trend direction: risk is {trend_direction} over time")
 
    print("\nProjected volatility, next 30 business days (first 5):")
    print(forecast.head())
    print(f"\nCurrent volatility: {rolling_vol.iloc[-1]:.4%}")
    print(f"Projected volatility in 30 days: {forecast.iloc[-1]:.4%}")
