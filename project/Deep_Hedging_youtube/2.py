from distutils.command.install_egg_info import to_filename
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def init():
    global r,sig,T,M,N,dt,rdt,sigsdt,S0,S,rv
    S0 = 1
    K = 1
    T = 30/365
    r = 0.0
    sig = 0.2

    M = 1000
    N = 100

    dt = T/N
    rdt = r*dt
    sigsdt = sig * np.sqrt(dt)

    ''' -------------------------- '''
    np.random.seed(1234)

    S = np.empty([M,N+1])
    rv = np.random.normal(r*dt,sigsdt,[M,N])

    for i in range(M):
        S[i,0] = S0
        for j in range(N):
            S[i,j+1] = S[i,j] * (1+rv[i,j])
init()
def first():
    a = []
    K = 100

    for i in range(M):
        cost = 0
        hedge = 0
        for j in range(N):
            d1 = (np.log(S[i,j]/K) + (r+0.5*sig**2)*(T-j*dt)/(sig*np.sqrt(T-j*dt)))
            delta = norm.cdf(d1)
            cost += (delta-hedge) * S[i,j] - 2.287150628044694 # payoff 빼주기
            hedge = delta
        
        cost -= hedge*S[i,N]
        a.append(cost)
    plt.plot(S[:,-1],a, marker='.', linestyle='none')
    plt.show()

def second():
    a = []
    K = 100

    for i in range(M):
        cost = 0
        price = S[i,0]
        for j in range(N):
            d1 = (np.log(S[i,j]/K) + (r+0.5*sig**2)*(T-j*dt)/(sig*np.sqrt(T-j*dt)))
            delta = norm.cdf(d1)
            cost += delta*(price-S[i,j+1])
            price = S[i,j+1]

        cost = cost + np.maximum(S[i,N]-K,0)
        a.append(cost)
    plt.plot(S[:,-1],a, marker='.', linestyle='none')
    plt.show()
'''
init()
first()

init()
second()
'''

import tensorflow as tf

my_input = []

hedge_cost = tf.keras.layers.Input(shape=(1,),name='hedge_cost')
my_input = my_input + [hedge_cost]

price = tf.keras.layers.Input(shape=(1,),name='price')
my_input = my_input + [price]

for j in range(3):
    delta = tf.keras.layers.Dense(1,name=str(j))(price) # output

    new_price = tf.keras.layers.Input(shape=(1,), name="S_"+str(j))
    my_input = my_input + [new_price]

    # tensorflow는 두개씩밖에 안되
    price_inc = tf.keras.layers.Subtract(name="price_inc"+str(j))([price,new_price]) # price - new_price
    cost = tf.keras.layers.Multiply(name='multiply'+str(j))([delta,price_inc]) # delta * price_inc
    hedge_cost = tf.keras.layers.Add(name='cost_'+str(j))([delta,hedge_cost]) # delta + hedge_cost
    price = new_price

model = tf.keras.Model(inputs=my_input,outputs=hedge_cost)
print(my_input)
tf.keras.utils.plot_model(model, to_file='model.png',show_shapes=False)