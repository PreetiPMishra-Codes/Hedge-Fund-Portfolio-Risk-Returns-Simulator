"""
Hedge Fund Portfolio Risk & Returns Simulator

Data Ingestion & Data cleaning 
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
