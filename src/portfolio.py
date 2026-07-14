import numpy as np
import pandas as pd
def calculate_portfolio_returns(returns:pd.DataFrame,weights:np.ndarray)->pd.Series:
    """
    combines individual stock returns into a single portfolio return series using matrix multiplication
    returns: DataFrame of shape (n_days, n_assets)
    weights: 1D array of shape (n_assets,) that sums to 1.0
    """
    ret_matrix=returns.values #coverts dataframe to numpy array
    portfolio_ret=np.dot(ret_matrix,weights)# np.dot performs matrix-vector multiplication:
    # (n_days, n_assets) dot (n_assets,) -> (n_days,)
    # each day's portfolio return = sum(weight_i * return_i) across assets
    portfolio_ret=pd.Series(portfolio_ret,index=returns.index,name="Portfolio")
    #wrapping to pandas series again so we keep the DatetimeIndex
    return portfolio_ret

def calculate_portfolio_variance(returns:pd.DataFrame,weights:np.ndarray)->float:
    """
    uses covariance matrix to compute historical portfolio variance
    this is not only calcs individual asset risks but only calcs how they move together
    """
    
    # Covariance matrix: shape (n_assets, n_assets).
    # cov_matrix[i][j] = how asset i and asset j move together
    cov_matrix=returns.cov()
    # Portfolio variance formula:w^T*Cov*w
    # This is the textbook Markowitz portfolio variance equation
    portf_var=np.dot(weights.T,np.dot(cov_matrix,weights))
    return portf_var
