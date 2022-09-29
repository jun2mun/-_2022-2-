import numpy as np
from scipy import stats

class BlackScholes(object):
    """ 
    Class to calculate (European) call and put option prices through the Black-Scholes formula 
    without dividends
    
    :param S: Price of underlying stock
    :param K: Strike price
    :param T: Time till expiration (in years)
    :param r: Risk-free interest rate (0.05 indicates 5%)
    :param sigma: Volatility (standard deviation) of stock (0.15 indicates 15%)
    """
    @staticmethod
    def _d1(S,K,T,r,sigma):
        return (np.log(S/K) + (r+sigma**2/2)*T) / (sigma*np.sqrt(T))

    def _d2(S,K,T,r,sigma):
        return (np.log(S/K) + (r-sigma**2/2)*T) / (sigma*np.sqrt(T))

    def bscall(S, K, T, r, sig):
        d1 = (np.log(S/K)+(r+0.5*sig**2)*T)/(sig*np.sqrt(T))
        d2 = (np.log(S/K)+(r-0.5*sig**2)*T)/(sig*np.sqrt(T))
        return S*stats.norm.cdf(d1)-K*np.exp(-r*T)*stats.norm.cdf(d2)
        
    def bsput(S, K, T, r, sig):
        d1 = (np.log(S/K)+(r+0.5*sig**2)*T)/(sig*np.sqrt(T))
        d2 = (np.log(S/K)+(r-0.5*sig**2)*T)/(sig*np.sqrt(T))
        return K*np.exp(-r*T)*stats.norm.cdf(-d2)-S*stats.norm.cdf(-d1)
