import numpy as np
import pandas as pd
#historical VAR
def calculate_historical_var(portfolio_ret:pd.Series,confidence_level:float=0.95)->float:
    """
    Historical VaR: looks at the ACTUAL distribution of past returns
    and finds the cutoff below which only (1 - confidence_level)% of
    days fell.
 
    confidence_level=0.95 -> "95% VaR": the loss threshold that
    historical returns breached on only 5% of days.
    """
    #if confidence level is 95% we want 5th percentile of days
    #the cutoff for worst 5% of days
    percentile_cutoff=(1-confidence_level)*100
    var=np.percentile(portfolio_ret,percentile_cutoff)
    return var
