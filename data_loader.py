"""
Hedge Fund Portfolio Risk & Returns Simulator

Data Ingestion & Data cleaning 
calculate daily pct returns
"""

import yfinance as yf
import pandas as pd
import numpy as np


# Define the portfolio universe and time window

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
START_DATE = "2021-01-01"
END_DATE = "2026-07-11"  

    
# Download historical price data
def fetch_price_data(tickers,start,end):
    """
    Downloads adjusted close prices for a list of tickers.
    Returns a single DataFrame: rows = dates, columns = tickers.
    """
    raw_data = yf.download(tickers,start=start,end=end,auto_adjust=True)

    # auto_adjust=True already adjusts for splits/dividends and
    # collapses the price to a single 'Close' column per ticker.
    prices = raw_data["Close"]

    return prices


# Clean the data
def clean_price_data(prices:pd.DataFrame)->pd.DataFrame:
    """
    Handles missing data (holidays,delistings,API gaps).
    """
    # Forward-fill first: if a price is missing for a day, assume
    # it's still worth what it was worth the last known day.
    prices = prices.ffill()

    # Drop any leading rows that are still NaN
    prices = prices.dropna()

    return prices


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
