import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
print(f'평균 : {r*dt} , 분산 : sigsdt : {sigsdt} ,\n {[M,N]}')
rv = np.random.normal(r*dt,sigsdt,[M,N]) #평균,표준편차(분산),샘플 사이즈



''' -------------------------- '''

for i in range(M):
    S[i,0] = S0
    for j in range(N):
        S[i,j+1] = S[i,j] * (1+rv[i,j])

for i in range(M):
    plt.plot(S[i,:])

plt.show()

''' -------------------------- '''
m = 0
for i in range(N):
    print(S[m,j])
    # 100번째가 최종 주가