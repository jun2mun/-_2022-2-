### CRR 논문 참조 ###

# black scholes 모형 #
# Call의 delta -> N(d1)
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

r = 0.00
sig = 0.2 # 시그마 => 식의 변동성  ##중요!! ##
T = 30 / 365 # 365일중에 30일

M = 100 # 100개의 시나리오(시뮬레이션) -> m=99일 경우 99번째ㅐ 시나리오
N = 30 # 주어진 시간을 30개로 나누어 보겠다.

dt = T/N  # 하루
rdt = r*dt
sigsdt = sig * np.sqrt(dt)

''' -------------------------- '''
S0 = 100 # 현재 시점의 가격
np.random.seed(100)
S = np.empty([M,N+1]) # 빈 array 생성 -> 속도를 위해 // 제일 앞에 100을 넣어줄꺼기 때문에 N+1 
rv = np.random.normal(r*dt,sigsdt,[M,N]) #평균,표준편차(분산),샘플 사이즈
for i in range(M):
    S[i,0] = S0
    for j in range(N):
        S[i,j+1] = S[i,j] * (1+rv[i,j])

S_temp = S[:]

def first():
    m = 99
    K = 100 # 행사 가격

    hedge =  0 # 초기 0인 이유 : 주식을 안가지고 있다라고 설명
    cost = 0
    for j in range(N):
        d1 = (np.log(S[m,j]/K)+(r+0.5*sig**2)*(T-j*dt))/(sig*np.sqrt(T-j*dt)) # T-j => T-t
        delta = norm.cdf(d1) # call 옵션의 delta는 N(d1)
        #cost_1 = (delta -hedge) * S[m,j] # 주식을 (hedge-delta)개 만큼 사야됨(의미) * 주식 가격
        cost = (delta-hedge) * S[m,j] # 주식을 (hedge-delta)개 만큼 사야됨(의미) * 주식 가격

        print(S[m,j],delta,delta-hedge,cost) 
        # 기초자산의 순간순간 바뀌는 가격 | 델타 | 주식을 (delta-hedge)만큼 가지고 있어야함 | 누적되는 비용값
        '''
        print(S[m,j],delta,delta-hedge
        ex) 100.0 0.51 0.51
            99.24 0.45 -0.05 -> 주식의 가격 99.24 | 주식을 0.45개 만큼 가지고 있어야함 | 주식을 -0.05개만큼 사야됨
        '''

        '''
        print(S[m,j],delta,delta-hedge ,cost_1)
        ex) 100.0 0.51 0.51
            99.24 0.45 -0.05 -> 주식의 가격 99.24 | 주식을 0.45개 만큼 가지고 있어야함 | 주식을 -0.05개만큼 사야됨 | 비용
        '''

        # 100.0 0.51.... (기초자산(그날의) 가격 :100 , 그때 delta는 0.51..)
        # 목표는 초기 delta값을 유지하려고 하는것이 궁극적인 목표!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        hedge = delta
#first()

# 확인필요
def second():
    M = 100
    a = []
    K = 100 # 행사 가격
    #S = S_temp[:]

    for i in range(M):
        hedge =  0 # 초기 0인 이유 : 주식을 안가지고 있다라고 설명
        cost = 0
        for j in range(N):
            d1 = (np.log(S[i,j]/K)+(r+0.5*sig**2)*(T-j*dt))/(sig*np.sqrt(T-j*dt)) # T-j => T-t
            delta = norm.cdf(d1) # call 옵션의 delta는 N(d1)
            cost = (delta-hedge) * S[i,j] # 주식을 (hedge-delta)개 만큼 사야됨(의미) * 주식 가격
            hedge = delta
        if S[i,N] > K:
            cost = cost + (hedge-1) * S[i,N] + K
        
        else:
            cost = cost + (hedge-0) * S[i,N]

        a.append(cost)

    plt.hist(a,bins=20)
    plt.show()
    ### 궁극적인 목표는 cost - 옵션 가격을 뺐을 때 0이 되게하는것
    ### delta를 하나의 값으로 모이게할 수 있었다 -> JP Morgan
    
#second()