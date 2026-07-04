import yfinance as yf
import pandas as pd 
import numpy as np 
#portfolio universe
TICKERS=["AAPL","MSFT","GOOGL","AMZN"];
START_DATE="2020-07-01" #YYYY-MM-DD
END_DATE="2026-07-01" # 5 years we are taking long enough to track macroeconomic shifts
#yet short enuf to be relevant and have fast data processing

def fetch_price_data(tkr,start,end):
    """
    Downloads adjusted close prices for a list of tickers.
    Returns a single DataFrame with tickers as columns.
    """
    raw_data=yf.download(tkr,start=start,end=end,auto_adjust=True) #split and dividend-adjusted closing prices
    prices=raw_data["Close"] #turns to a single-layer dataframe
    #rows dates, cols tickers
    return prices

    #execution block
if __name__ == "__main__":
    prices = fetch_price_data(TICKERS, START_DATE, END_DATE)
    print(prices.head())
    print(prices.shape)
    print(prices.isna().sum())

    #data cleaning
def clean_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fills missing values (e.g. holidays mismatched across exchanges),
    then drops any remaining NaN rows (usually just the very start).
    """
    prices = prices.ffill()
    prices = prices.dropna()
    return prices

def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Converts price series into daily percentage returns.
    This is the foundation of almost all quant risk metrics.
    """
    returns = prices.pct_change().dropna()
    return returns
prices = fetch_price_data(TICKERS, START_DATE, END_DATE)
prices = clean_price_data(prices)
returns = calculate_daily_returns(prices)

print(returns.head())
print(returns.describe())