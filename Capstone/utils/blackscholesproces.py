import numpy as np
from scipy.stats import norm

class BlackScholes:
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
    def _d1(S, K, T, r, sigma):
        return (1 / (sigma * np.sqrt(T))) * (np.log(S/K) + (r + sigma**2 / 2) * T)
    
    def _d2(self, S, K, T, r, sigma):
        return self._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    def call_price(self, S, K, T, r, sigma):
        """ Main method for calculating price of a call option """
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        return norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r*T)
    
    def put_price(self, S, K, T, r, sigma):
        """ Main method for calculating price of a put option """
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        return norm.cdf(-d2) * K * np.exp(-r*T) - norm.cdf(-d1) * S
    
    def call_in_the_money(self, S, K, T, r, sigma):
        """ 
        Calculate probability that call option will be in the money at
        maturity according to Black-Scholes.
        """
        d2 = self._d2(S, K, T, r, sigma)
        return norm.cdf(d2)
    
    def put_in_the_money(self, S, K, T, r, sigma):
        """ 
        Calculate probability that put option will be in the money at
        maturity according to Black-Scholes.
        """
        d2 = self._d2(S, K, T, r, sigma)
        return 1 - norm.cdf(d2)


    def call_implied_volatility(price, S, K, T, r):
        """ Calculate implied volatility of a call option up to 2 decimals of precision. """
        sigma = 0.0001
        while sigma < 1:
            d1 = BlackScholes()._d1(S, K, T, r, sigma)
            d2 = BlackScholes()._d2(S, K, T, r, sigma)
            price_implied = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            if price - price_implied < 0.0001:
                return sigma
            sigma += 0.0001
        return "Not Found"

    def put_implied_volatility(price, S, K, T, r):
        """ Calculate implied volatility of a put option up to 2 decimals of precision. """
        sigma = 0.0001
        while sigma < 1:
            call = BlackScholes().call_price(S, K, T, r, sigma)
            price_implied = K * np.exp(-r*T) - S + call
            if price - price_implied < 0.0001:
                return sigma
            sigma += 0.0001
        return "Not Found"

'''
S = 44 # Stock price
K = 45 # Strike price
T = 2/12 # Time till expiration (in years)
r = 0.06 # Risk-free interest rate
sigma = np.sqrt(0.2) # Volatility (standard deviation)

BlackScholes().call_price(S, K, T, r, sigma)
BlackScholes().put_price(S, K, T, r, sigma)

S = 36 # Stock price
K = 35 # Strike price
T = 2/12 # Time till expiration (in years)
r = 0.05 # Risk-free interest rate
option_price = 1.5 # Price of the option that is offered in the market

call_imp_vol = call_implied_volatility(option_price, S, K, T, r)
put_imp_vol = put_implied_volatility(option_price, S, K, T, r)

print(f"Implied volatility for call option = {round(call_imp_vol*100, 3)}%")
print(f"Implied volatility for put option = {round(put_imp_vol*100, 3)}%")
'''

# 여러 외부 변수의 변화에 대한 옵션 가격의 민감도
class BlackScholesGreeks:
    """ 
    Class to calculate (European) call and put option greeks.
    
    :param S: Price of underlying stock
    :param K: Strike price
    :param T: Time till expiration (in years)
    :param r: Risk-free interest rate (0.05 indicates 5%)
    :param sigma: Volatility (standard deviation) of stock (0.15 indicates 15%)
    :param q: The annual dividend yield of a stock
    """
    @staticmethod
    def _d1(S, K, T, r, sigma):
        return (1 / (sigma * np.sqrt(T))) * (np.log(S/K) + (r + sigma**2 / 2) * T)
    
    def _d2(self, S, K, T, r, sigma):
        return self._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    def call_delta(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        return norm.cdf(d1)
    
    def call_gamma(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def call_vega(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T)
    
    def call_theta(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        return - ((S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) - r * K * np.exp(-r*T) * norm.cdf(d2)
    
    def call_rho(self, S, K, T, r, sigma):
        d2 = self._d2(S, K, T, r, sigma)
        return K * T * np.exp(-r*T) * norm.cdf(d2)
    
    def call_lambda(self, S, K, T, r, sigma):
        delta = self.call_delta(S, K, T, r, sigma)
        return delta * (S / V)
    
    def call_epsilon(self, S, K, T, r, sigma, q):
        d1 = self._d1(S, K, T, r, sigma)
        return -S * T * np.exp(-q*T) * norm.cdf(d1)
    
    def put_delta(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        return norm.cdf(d1) - 1
    
    def put_gamma(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def put_vega(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T)
    
    def put_theta(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        return - ((S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) + r * K * np.exp(-r*T) * norm.cdf(-d2)
    
    def put_rho(self, S, K, T, r, sigma):
        d2 = self._d2(S, K, T, r, sigma)
        return - K * T * np.exp(-r*T) * norm.cdf(-d2)
    
    def put_lambda(self, S, K, T, r, sigma):
        delta = self.put_delta(S, K, T, r, sigma)
        return delta * (S / V)
    
    def put_epsilon(self, S, K, T, r, sigma, q):
        d1 = self._d1(S, K, T, r, sigma)
        return S * T * np.exp(-q*T) * norm.cdf(-d1)
    
    def vanna(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        return norm.pdf(d1) * d2 / sigma
    
    def charm(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        return -norm.pdf(d1) * ((2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)))
    
    def vomma(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) * ((d1 * d2) / sigma)
    
    def veta(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        return -S * norm.pdf(d1) * np.sqrt(T) * (r * d1 / (sigma * np.sqrt(T)) - (1 + d1 * d2) / (2 * T))
    
    def zomma(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        return norm.pdf(d1) * (d1*d2 - 1) / (S * sigma**2 * np.sqrt(T))
    
    def color(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        return norm.pdf(d1) / (2*S*T*sigma*np.sqrt(T)) * (2*T + 1 + ((2*r*T - d2*sigma*np.sqrt(T)) / sigma*np.sqrt(T)) * d1)
    
    def ultima(self, S, K, T, r, sigma):
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        vega = self.call_vega(S, K, T, r, sigma)
        return -vega/sigma**2 * (d1 * d2 * (1-d1*d2) + d1**2 + d2**2)

# 배당금
class BlackScholesDividend:
    """ 
    Class to calculate (European) call and put option prices through the Black-Scholes formula 
    with dividends
    
    :param S: Price of underlying stock
    :param K: Strike price
    :param T: Time till expiration (in years)
    :param r: Risk-free interest rate (0.05 indicates 5%)
    :param sigma: Volatility (standard deviation) of stock (0.15 indicates 15%)
    :param q: Annual dividend yield (0.05 indicates 5%)
    """
    @staticmethod
    def _d1(S, K, T, r, sigma, q):
        return (1 / (sigma * np.sqrt(T))) * (np.log(S * np.exp(-q*T) / K) + (r + sigma**2 / 2) * T)
    
    def _d2(self, S, K, T, r, sigma, q):
        return self._d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)
    
    def call_price(self, S, K, T, r, sigma, q):
        """ Main method for calculating price of a call option """
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        return norm.cdf(d1) * S * np.exp(-q*T) - norm.cdf(d2) * K * np.exp(-r*T)
    
    def put_price(self, S, K, T, r, sigma, q):
        """ Main method for calculating price of a put option """
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        return norm.cdf(-d2) * K * np.exp(-r*T) - norm.cdf(-d1) * S * np.exp(-q*T)
    
    def call_in_the_money(self, S, K, T, r, sigma, q):
        """ 
        Calculate probability that call option will be in the money at
        maturity according to Black-Scholes.
        """
        d2 = self._d2(S, K, T, r, sigma, q)
        return norm.cdf(d2)
    
    def put_in_the_money(self, S, K, T, r, sigma, q):
        """ 
        Calculate probability that put option will be in the money at
        maturity according to Black-Scholes.
        """
        d2 = self._d2(S, K, T, r, sigma, q)
        return 1 - norm.cdf(d2)