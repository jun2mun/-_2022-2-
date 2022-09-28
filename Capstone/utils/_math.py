import numpy as np
import scipy.stats as stat

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True)


def europian_option(S,K,T,r,sigma,option_type):
    import numpy as np
    import scipy.stats as stat
    """
    블랙숄즈 모형 (유러피안 옵션)
    europtian_option(100,100,1,0.02,0.2,'call') 기초자산의 현재가격 100, 행사가격 100, 만기를 1년, 무위험이자율을 2% , 기초자산의 변성 20%
    => 이론가 8.916
    K는 행사가
    r은 무위험이자율
    sigma 기초자산의 연간 변동성
    T 만기 (3개월, 1년 등 만기까지의 기간)
    S 기초자산의 가격
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T)  / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # stat.norm.cdf(d2) => 만기 시 행사가격 X인 옵션이 행사될 확률
    #  
    if option_type == 'call': # 콜
        V = S * stat.norm.cdf(d1) - K * np.exp(-r * T) * stat.norm.cdf(d2)
    else: # 풋
        V = K * np.exp(-r * T) * stat.norm.cdf(-d2) - S * stat.norm.cdf(-d1)

    return V # 옵션의 가치()


def BS_CALL(S,K,T,r,sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T)  / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stat.norm.cdf(d1) - K * np.exp(-r * T) * stat.norm.cdf(d2)

def BS_PUT(S,K,T,r,sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T)  / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * stat.norm.cdf(-d2) - S * stat.norm.cdf(-d1)

# 배당금 지급 주식
def BS_CALLDIV(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return S*np.exp(-q*T) * stat.norm.cdf(d1) - K * np.exp(-r*T)* stat.norm.cdf(d2)

def BS_PUTDIV(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*stat.norm.cdf(-d2) - S*np.exp(-q*T)*stat.norm.cdf(-d1)

def binomial():
    import math


class BsOption:
    def __init__(self,S,K,T,r,sigma,q =0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    @staticmethod # 인스턴스를 통하지 않고 클래스에서 바로 호출할 수 있는 정적 메서드
    def N(x):
        return stat.norm.cdf(x)
    
    @property # get 함수 역할
    def params(self):
        return {'S': self.S, 
                'K': self.K, 
                'T': self.T, 
                'r':self.r,
                'q':self.q,
                'sigma':self.sigma}

    def d1(self):
        return (np.log(self.S/self.K) + (self.r -self.q + self.sigma**2/2)*self.T) \
                                / (self.sigma*np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.T)

    
    def _call_value(self):
        return self.S*np.exp(-self.q*self.T)*self.N(self.d1()) - \
                    self.K*np.exp(-self.r*self.T) * self.N(self.d2())
                    
    def _put_value(self):
        return self.K*np.exp(-self.r*self.T) * self.N(-self.d2()) -\
                self.S*np.exp(-self.q*self.T)*self.N(-self.d1())
    
    def price(self, type_ = 'C'):
        if type_ == 'C':
            return self._call_value()
        if type_ == 'P':
            return self._put_value() 
        if type_ == 'B':
            return  {'call': self._call_value(), 'put': self._put_value()}
        else:
            raise ValueError('Unrecognized type')

'''
import math

def combos(n, i):  # 조합
    return math.factorial(n) / (math.factorial(n-i)* math.factorial(i))

fair_value = 0
n = 4 # number of coin
for k in range(n+1):
    fair_value += combos(n,k)*0.5**k*0.5**(n-k) * k

print(fair_value)


'''


import numpy as np
import matplotlib.pyplot as plt
r = 0.0
sig = 0.2
T = 30/365

M = 100
N = 30

dt = T / N
rdt = r * dt
sigsdt = sig * np.sqrt(dt)


S0 = 100
np.random.seed(20220617)
S = np.empty((M,N+1))
rv = np.random.normal(r*dt,sigsdt,[M,N])

for i in range(M):
    S[i,0] = S0
    for j in range(N):
        S[i,j+1] = S[i,j] * (1 + rv[i,j])

m = 0
#for j in range(N+1):
    #print(S[m,j])

S_avg = []
Asian_payoff = []

K = 100

for i in range(M):
    S_avg.append(np.mean(S[i,:]))
    Asian_payoff.append(np.maximum(S_avg[i] - K,0)) # 0보다 큰 애들만 남김
    #print(Asian_payoff[i])


print(np.mean(Asian_payoff) * np.exp(-r*T))