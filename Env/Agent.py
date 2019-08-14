#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:49:33 2019

@author: feifanhe
"""

import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras import utils as np_utils
from keras.layers import Input, Dense

import Env as stock

class Agent():
    # Target: 針對單一股票買賣
    # Input: 歷史股價
    # Output: 買/賣/不動
    def __init__(self, s_dim, a_dim, n, lr):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.n = n
        self.lr = lr
        self.__buildModel()
        
        # 儲存 n 個回合的 sample 用
        self.s_batch = np.empty(n, dtype = object)
        self.a_batch = np.empty(n, dtype = object)
        self.r_batch = np.empty(n, dtype = object)
        
        self.batch_counter = 0
        
    def __buildModel(self):
        # TODO: 建立 Agent(只要在input_layer跟output_layer之間加hidden layer就好)
        input_layer = Input(shape=(self.s_dim, ))
        h = Dense(16, activation='selu')(input_layer)
        h = Dense(8, activation='selu')(h)
        output_layer = Dense(self.a_dim, activation='softmax')(h)
        self.model = Model(inputs = input_layer, outputs = output_layer)
        self.model.compile(loss = 'categorical_crossentropy', 
                           optimizer = Adam(lr = self.lr))
        
    def sampleAction(self, s):
        # TODO: 讓 Agent sample 一個 action
        action = self.model.predict(s).reshape((self.a_dim, ))
        return np.random.choice(self.a_dim, p = action)
    
    def storeSample(self, s, a, r):
        # TODO: 儲存一個 episode 中所有的(s,a,r)pair
        self.s_batch[self.batch_counter] = np.vstack(s)
        self.a_batch[self.batch_counter] = np_utils.to_categorical(a, num_classes=self.a_dim)
        self.r_batch[self.batch_counter] = utilsToos.processReward(r)
        self.batch_counter += 1
        
    def fit(self):
        # TODO: 用 N 個trajectory 的更新 agent
        # self.model.fit(s, a*r...)
        # self.model.fit(s, a, sample_weight = r...)
        
        S = np.concatenate(self.s_batch)
        A = np.concatenate(self.a_batch)
        R = np.concatenate(self.r_batch)
        
        # normalize R
        R -= np.mean(R)
        R /= np.std(R) + K.epsilon()
        R += K.epsilon()
        
        self.model.fit(
                x = S,
                y = A,
                sample_weight = R,
                epochs = 1,
                verbose = 0
                )
        
        self.batch_counter = 0
        
if __name__ == '__main__':
    episode = 100
    n = 10
    learning_rate = 0.01
    
    env = stock.Env('./stockData/')