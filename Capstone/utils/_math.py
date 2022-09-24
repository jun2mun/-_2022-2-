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
    if option_type == 'call':
        V = S * stat.norm.cdf(d1) - K * np.exp(-r * T) * stat.norm.cdf(d2)
    else:
        V = K * np.exp(-r * T) * stat.norm.cdf(-d2) - S * stat.norm.cdf(-d1)

    return V # 옵션의 가치()


