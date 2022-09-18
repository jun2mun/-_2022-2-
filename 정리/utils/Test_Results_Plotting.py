import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.D_H_math import BS_delta, BlackScholes_price

def test_hedgin_strategy(deltas,paths,K,price,alpha,output=True):
    S_returns = paths[1:,:,0] -paths[:-1,:,0]
    hedge_pnl = np.sum(deltas * S_returns,axis=0)
    option_payoff = np.maximum(paths[-1:,:,0] -K , 0)
    replication_portfolio_pnls = -option_payoff + hedge_pnl + price
    mean_pnl = np.mean(replication_portfolio_pnls)
    cvar_pnl = -np.mean(np.sort(replication_portfolio_pnls)[:int((1-alpha)*replication_portfolio_pnls.shape[0])])
    if output:
        plt.hist(replication_portfolio_pnls)
        print('BS price at t0:',price)
        print('Mean Hedging PnL:', mean_pnl)
        print('Mean Hedging PnL:',mean_pnl)
        print('CVaR Hedging PnL:',cvar_pnl)
    return (mean_pnl,cvar_pnl,hedge_pnl,replication_portfolio_pnls,deltas)

def plot_deltas(paths,deltas_bs,deltas_rnn,times=[0,1,5,10,15,29]):
    fig = plt.figure(figsize=(10,6))
    for i,t in enumerate(times):
        plt.subplot(2,3,i+1)
        xs = paths[t,:,0]
        ys_bs = deltas_bs[t,:]
        ys_rnn = deltas_rnn[t,:]
        df = pd.DataFrame([xs,ys_bs,ys_rnn]).T
        #df = df.groupby(0, as_index=False).agg({1:np.mean,
        #                                          2: np.mean})
        plt.plot(df[0],df[1],df[0],df[2],linestyle='',marker='x')
        plt.legend(['BS_delta','RNN Delta'])
        plt.title('Delta at Time %i' %t)
        plt.xlabel('Spot')
        plt.ylabel('$\Delta$')
    plt.tight_layout()

def plot_strategy_pnl(portfolio_pnl_bs,portfolio_pnl_rnn):
    fig = plt.figure(figsize=(10,6))
    sns.boxplot(x=['Black-Scholes','RNN-LSTM-v1'], y=[portfolio_pnl_bs,portfolio_pnl_rnn])
    plt.title('Compare PnL Replication Strategy')
    plt.ylabel('PnL')

####### 블랙 숄즈 복제에 대한 헤징 오류 ######

def black_scholes_hedge_strategy(S_0,K,r,vol,T,paths,alpha,output):
    bs_price = BlackScholes_price(S_0,T,r,vol,K,0)
    times = np.zeros(paths.shape[0])
    times[1:] = T / (paths.shape[0]-1)
    bs_deltas = np.zeros((paths.shape[0]-1, paths.shape[1]))
    for i in range(paths.shape[0]-1):
        t = times[i]
        bs_deltas[i,:] = BS_delta(paths[i,:,0],T,r,vol,K,t)
    return test_hedgin_strategy(bs_deltas,paths,K,bs_price,alpha,output)

