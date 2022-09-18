from tkinter import N
import numpy as np
import scipy.stats as stats


def monte_carlo_paths(S_0,time_to_expiry,sigma,drift,seed,n_sims,\
    n_timesteps):
    print("monte_carlo_paths func")
    """
    Create random paths of a stock price following a brownian geomatric motion
        return:
    a (n_timesteps x n_sims x 1) matrix
    """
    ### np.random.randn : 가우시안 표준 정규 분포에서 난수 matrix array 생성
    ### np.cumprod : 각 원소들의 누적곱
    ### np.transpose : 전치행렬
    ### np.c_ 행렬 합치기 (왼쪽에서 오른쪽) [1,2] [3,4] -> [1,2,3,4]
    if seed > 0: # 난수
        np.random.seed(seed)
    stdnorm_random_variates = np.random.randn(n_sims, n_timesteps)
    # 주가 S
    S = S_0
    # ?
    dt = time_to_expiry / stdnorm_random_variates.shape[1]
    # 감마(드리프트)
    r = drift
    
    # 몬테카를로 구현?
    S_T = S * np.cumprod(np.exp((r-sigma**2/2)*dt+sigma*np.sqrt(dt)*\
        stdnorm_random_variates),axis=1)
    #print(S_T.shape) # (5000,30)
    #print(np.c_[np.ones(n_sims)*S_0,S_T])
    #print(np.c_[np.ones(n_sims)*S_0,S_T].shape) #(5000,31)
    #print(np.transpose(np.c_[np.ones(n_sims)*S_0,S_T]))
    #print(np.transpose(np.c_[np.ones(n_sims)*S_0,S_T]).shape) # (31,5000)

    # 전치행렬 이해
    return np.reshape(np.transpose(np.c_[np.ones(n_sims)*S_0,S_T]),\
        (n_timesteps+1,n_sims,1))


### 도우미 함수 ###
def BS_d1(S, dt, r, sigma, K):
    return (np.log(S/K) + (r+sigma**2/2)*dt) / (sigma*np.sqrt(dt))

def BlackScholes_price(S,T,r,sigma,K,t=0):
    dt = T-t
    Phi = stats.norm(loc=0,scale=1).cdf
    d1 = BS_d1(S,dt,r,sigma,K)
    d2 = d1 - sigma*np.sqrt(dt)
    return S*Phi(d1) - K*np.exp(-r*dt)*Phi(d2)

def BS_delta(S,T,r,sigma,K, t=0):
    dt = T-t
    d1 = BS_d1(S,dt,r,sigma,K)
    Phi = stats.norm(loc=0,scale=1).cdf
    return Phi(d1)