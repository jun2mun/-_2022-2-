import numpy as np
from scipy import stats

# 블랙숄즈에서 델타가 되는 값
# Call options delta : N(d1)
# PUT options delta : N(d1) -1
# delta의미 : S(주식)을 *delta 만큼 사라
def d1(S,K,sigma,T,r): # K는 행사가격
    """
    N(d1) 은 옵션 가격의 기초자산 가격 변화에 대한 민감도
    -> 기초자산(S) 가격이 변할 때 콜(풋) 프리미엄의 변화를 나타내는 측도
    """
    return  (np.log(S / K) + (r + 0.5 * sigma ** 2) * T)  / (sigma * np.sqrt(T))

def d2(S,K,sigma,T,r):
    """
    N(d2) 는 만기 시점에 기초자산 가격이 행사가격보다 높을 확률
    콜(풋) 옵션이 행사될 확률
    """
    return (np.log(S/K) + (r - 0.5 *sigma **2) * T ) / (sigma * np.sqrt(T))

def BS_Call(S,K,sigma,T,r):
    """
    가정 : 풋-콜 패리티
    """
    call_d1 = stats.norm.cdf(d1(S,K,sigma,T,r))
    call_d2 = stats.norm.cdf(d2(S,K,sigma,T,r))
    return S * call_d1 - K * np.exp(-r * T) * call_d2

def BS_PUT(S,K,sigma,T,r):
    """
    가정 : 풋-콜 패리티
    """
    put_d1 = stats.norm.cdf(-d1(S,K,sigma,T,r))
    put_d2 = stats.norm.cdf(-d2(S,K,sigma,T,r))
    return K * np.exp(-r * T) * put_d2 - S * put_d1


'''
print(f'd1(델타)의 값은 : {d1(S,K,sigma,T,r)}')
print(f'd2(옵션이 행사될 확률은 : {d2(S,K,sigma,T,r)}')
print(f'유럽형 콜옵션의 이론 가격은 : {BS_Call(S,K,sigma,T,r)}')
print(f'유럽형 풋옵션의 이론 가격은 : {BS_PUT(S,K,sigma,T,r)}')
'''

def monte_carlo_paths(S_0, time_to_expiry, sigma, drift, seed, n_sims, n_timesteps):
    """
    Create random paths of a underlying following a browian geometric motion
    
    input:
    
    S_0 = Spot at t_0
    time_to_experiy = end of the timeseries (last observed time)
    sigma = the volatiltiy (sigma in the geometric brownian motion)
    drift = drift of the process
    n_sims = number of paths to generate
    n_timesteps = numbers of aquidistant time steps 
    
    return:
    
    a (n_timesteps x n_sims x 1) matrix
    """
    if seed > 0:
            np.random.seed(seed)
    stdnorm_random_variates = np.random.randn(n_sims, n_timesteps)
    S = S_0
    dt = time_to_expiry / stdnorm_random_variates.shape[1]
    r = drift
    # See Advanced Monte Carlo methods for barrier and related exotic options by Emmanuel Gobet
    S_T = S * np.cumprod(np.exp((r-sigma**2/2)*dt+sigma*np.sqrt(dt)*stdnorm_random_variates), axis=1)
    return np.reshape(np.transpose(np.c_[np.ones(n_sims)*S_0, S_T]), (n_timesteps+1, n_sims, 1))