from blackscholesproces import BlackScholes

class OptionStrategies:
    @staticmethod
    def short_straddle(S, K, T, r, sigma):
        call = BlackScholes().call_price(S, K, T, r, sigma)
        put = BlackScholes().put_price(S, K, T, r, sigma)
        return - put - call
    
    @staticmethod
    def long_straddle(S, K, T, r, sigma):
        call = BlackScholes().call_price(S, K, T, r, sigma)
        put = BlackScholes().put_price(S, K, T, r, sigma)
        return put + call
    
    @staticmethod
    def short_strangle(S, K1, K2, T, r, sigma):
        assert K1 < K2, f"Please make sure that K1 < K2. Now K1={K1}, K2={K2}"
        put = BlackScholes().put_price(S, K1, T, r, sigma)
        call = BlackScholes().call_price(S, K2, T, r, sigma)
        return - put - call
    
    @staticmethod
    def long_strangle(S, K1, K2, T, r, sigma):
        assert K1 < K2, f"Please make sure that K1 < K2. Now K1={K1}, K2={K2}"
        put = BlackScholes().put_price(S, K1, T, r, sigma)
        call = BlackScholes().call_price(S, K2, T, r, sigma)
        return put + call
    
    @staticmethod
    def short_put_butterfly(S, K1, K2, K3, T, r, sigma):
        assert K1 < K2 < K3, f"Please make sure that K1 < K2 < K3. Now K1={K1}, K2={K2}, K3={K3}"
        put1 = BlackScholes().put_price(S, K1, T, r, sigma)
        put2 = BlackScholes().put_price(S, K2, T, r, sigma)
        put3 = BlackScholes().put_price(S, K3, T, r, sigma)
        return - put1 + 2 * put2 - put3
    
    @staticmethod
    def long_call_butterfly(S, K1, K2, K3, T, r, sigma):
        assert K1 < K2 < K3, f"Please make sure that K1 < K2 < K3. Now K1={K1}, K2={K2}, K3={K3}"
        call1 = BlackScholes().call_price(S, K1, T, r, sigma)
        call2 = BlackScholes().call_price(S, K2, T, r, sigma)
        call3 = BlackScholes().call_price(S, K3, T, r, sigma)
        return call1 - 2 * call2 + call3
    
    @staticmethod
    def short_iron_condor(S, K1, K2, K3, K4, T, r, sigma):
        assert K1 < K2 < K3 < K4, f"Please make sure that K1 < K2 < K3 < K4. Now K1={K1}, K2={K2}, K3={K3}, K4={K4}"
        put1 = BlackScholes().put_price(S, K1, T, r, sigma)
        put2 = BlackScholes().put_price(S, K2, T, r, sigma)
        call1 = BlackScholes().call_price(S, K3, T, r, sigma)
        call2 = BlackScholes().call_price(S, K4, T, r, sigma)
        return put1 - put2 - call1 + call2
    
    @staticmethod
    def long_iron_condor(S, K1, K2, K3, K4, T, r, sigma):
        assert K1 < K2 < K3 < K4, f"Please make sure that K1 < K2 < K3 < K4. Now K1={K1}, K2={K2}, K3={K3}, K4={K4}"
        put1 = BlackScholes().put_price(S, K1, T, r, sigma)
        put2 = BlackScholes().put_price(S, K2, T, r, sigma)
        call1 = BlackScholes().call_price(S, K3, T, r, sigma)
        call2 = BlackScholes().call_price(S, K4, T, r, sigma)
        return - put1 + put2 + call1 - call2

'''
import numpy as np
import matplotlib.pyplot as plt
T = 2/12 # Time till expiration (in years)
r = 0.05 # Risk-free interest rate
sigma = np.sqrt(1/68) # Volatility (standard deviation)
end_prices = np.arange(5, 65, 1) # Possible scenarios for underlying stock price at expiration.
print(end_prices)

K1 = 35 # Strike price
straddle_payoffs = [OptionStrategies.long_straddle(S, K1, T, r, sigma) for S in end_prices]
plt.plot(end_prices, straddle_payoffs)
plt.xlabel('End Stock prices S')
plt.ylabel('Payoff')
plt.title(f'T={round(T, 2)}, r={r}, sigma={round(sigma, 4)}')
plt.suptitle('Payoff function Black-Scholes using long straddle', weight='bold', fontsize=12)
plt.show()
'''