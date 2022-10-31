import tensorflow as tf
import numpy as np
import pandas as pd
print("텐서플로우 버전 :",tf.__version__)

def BlackScholes_model(hedge = 0, cost = 0, K = 23, T = 30/365, r= 0.0, sig = 0.2, N=13): #

    my_input = []

    premium = tf.keras.layers.Input(shape=(1,), name="premium")
    hedge_cost = tf.keras.layers.Input(shape=(1,), name='hedge_cost')
    price = tf.keras.layers.Input(shape=(1,), name="price")

    my_input = [premium] + [hedge_cost] + [price]

    N=14
    for j in range(N):

        delta = tf.keras.layers.Dense(32, activation='tanh')(price)
        delta = tf.keras.layers.Dense(1)(delta)

        new_price = tf.keras.layers.Input(shape=(1,), name='S'+str(j))
        my_input = my_input + [new_price]

        price_inc = tf.keras.layers.Subtract(name='price_inc_'+str(j))([price, new_price])
        cost = tf.keras.layers.Multiply(name="multiply_"+str(j))([delta, price_inc])
        hedge_cost = tf.keras.layers.Add(name='cost_'+str(j))([hedge_cost, cost])
        price = new_price

    payoff = tf.keras.layers.Lambda(lambda x : tf.math.maximum(x-K,0), name='payoff')(price)
    cum_cost = tf.keras.layers.Add(name="hedge_cost_plus_payoff")([hedge_cost, payoff])
    cum_cost = tf.keras.layers.Subtract(name="cum_cost-premium")([cum_cost, premium])

    model = tf.keras.Model(inputs=my_input, outputs=cum_cost)
    return model

def _execute(paths,strikes,riskaversion):
    pass    

