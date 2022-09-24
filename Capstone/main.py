from utils._math import europian_option
import numpy as np
'''
블랙숄즈 모형 (유러피안 옵션)
europtian_option(100,100,1,0.02,0.2,'call') 기초자산의 현재가격 100, 행사가격 100, 만기를 1년, 무위험이자율을 2% , 기초자산의 변성 20%
=> 이론가 8.916
K는 행사가
r은 무위험이자율
sigma 기초자산의 연간 변동성
T 만기 (3개월, 1년 등 만기까지의 기간)
S 기초자산의 가격
'''
# Parameter
K = 100
r = 0.01
sigma = 0.25
T = np.linspace(0,1,100) # 균일한 간격으로 채움
S = np.linspace(0,200,100)
T, S = np.meshgrid(T,S)
print(T)



#europian_option()