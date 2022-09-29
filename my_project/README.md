# 여기는 매주 추가될 코드 내용 #


'''
cost 계산 예)
    '''
    for i in range(rest_period):
        S0 = LCID['Price'][i]
        print(S0,end="|")
        d1 = BlackScholes._d1(S0,K,30/365 - i * 1/365,0.0,0.2)
        d2 = BlackScholes._d2(S0,K,30/365 - i * 1/365,0.0,0.2)
        
        delta = stats.norm.cdf(d1)
        delta_2 = stats.norm.cdf(d2)
        cost += (hedge - delta) * S0
        # hedge - delta 만큼 주식을 사라
        hedge = delta
    print("\n ----------")
    if S0 > 11:
        cost += (hedge -1) * S0 + K # 돈이 들어옴 (+)
    else :
        cost += (hedge -0) * S0 
    print(S0,cost,hedge)
    # 돈이 들어왔다 나갔다 반복해서 14일후 결과 값이 나온다.
    '''
'''