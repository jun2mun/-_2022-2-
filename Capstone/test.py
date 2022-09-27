import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm
from scipy.optimize import minimize


starting_capital = 1000 # PV
r = 0.05 # 5% interest
n = 50 # Years


def fv_simple(pv: float, r: float, n: int):
    return (1+r*n)*pv

def pv_simple(fv: float, r: float, n: int):
    return fv / (1 + r * n)
    
def r_simple(pv: float, fv: float, n: int):
    return (fv/pv - 1) /  n
    
fv = fv_simple(starting_capital, r, n)
pv = pv_simple(fv, r, n)
r_simple(pv, fv, n)

# 복리
def fv_compound(pv, r: float, n: int):
    return pv * (1+r)**n

def pv_compound(fv: float, r: float, n: int):
    return fv / (1 + r)**n

def r_compound(pv: float, fv: float, n: int):
    return (fv/pv)**(1/n) - 1

fv = fv_compound(starting_capital, r, n)
pv = pv_compound(fv, r, n)

r_compound(pv, fv, n)


m =12 # Monthly compounding // 1년에 이자를 여러번 받는 경우
def fv_interval(pv: float, r: float, n: int, m: int):
    return ((1 + r / m)**(m * n)) * pv

def pv_interval(fv: float, r: float, n: int, m: int):
    return fv / (1+r/m)**(m*n)

def r_interval(pv: float, fv: float, n: int, m: int):
    return ((fv / pv)**(1/(m*n)) - 1)* m


fv = fv_interval(starting_capital, r, n, m)
pv = pv_interval(fv, r, n, m)
r_interval(pv, fv, n, m)

# continuous
def fv_continuous(pv: float, r: float, n: int):
    return np.exp(r * n) * pv

def pv_continuous(fv: float, r: float, n: int):
    return fv / np.exp(r * n)

def r_continuous(pv: float, fv: float, n: int):
    return np.log(fv/pv) / n

A = 100. # Monthly added deposit
def fv_with_deposit(r: float, n: int, m: int, a: float):
    return (a / (r/m)) * ((1 + r/m)**(m*n) - 1)

def pv_with_deposit(r: float, n: int, m: int, a: float):
    return (a / (r/m)) * (1 - 1 / (1 + r / m)**(m*n))



var = 0.2 # Variance
sigma = np.sqrt(var) # Standard deviation
S = 44 # Stock price
K = 45 # Strike price
T = 2/12 # Time period (two months = two steps)
r = 0.06 # Risk-free interest rate
delta_t = 1/12 # time-steps of one month
R = np.exp(r * delta_t) # Discounted interest

u = np.exp(sigma*np.sqrt(delta_t)) # Up scenario
d = np.exp(-sigma*np.sqrt(delta_t)) # Down scenario

def price_binomial(R, Pu, Pd, u, d):
    """ Compute option price for a step """
    q = (R-d)/(u-d)
    return (1/R) * (q * Pu + (1-q)*Pd)

c_ud = np.maximum(0, u * d * S - K)
c_uu = np.maximum(0, u**2 * S - K)
c_dd = np.maximum(0, d**2 * S - K)

Cu = price_binomial(R, c_uu, c_ud, u, d)
print(f"Option price for 1-step up scenario: {Cu.round(2)}")
Cd = price_binomial(R, c_ud, c_dd, u, d)
print(f"Option price for 1-step down scenario: {Cd.round(2)}")
C = price_binomial(R, Cu, Cd, u, d)
print(f"Fair option price given all scenario's: {C.round(2)}")