#!/usr/bin/env
# -*- coding:utf-8 -*-

import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x) )

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'Logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_derivative

        self.weights = []
        for i in range(1, len(layers)-1):
            # [0,1) * 2 - 1 => [-1,1) => * 0.25 => [-0.25,0.25)
            self.weights.append( (2*np.random.random((layers[i-1] + 1, layers[i] + 1 ))-1 ) * 0.25 )
            self.weights.append( (2*np.random.random((layers[i] + 1, layers[i+1] ))-1 ) * 0.25 )
        # for i in range(0, len(layers)-1):
        #     m = layers[i]  # 第i层节点数
        #     n = layers[i+1]  # 第i+1层节点数
        #     wm = m + 1
        #     wn = n + 1
        #     if i == len(layers)-2:
        #         wn = n
        #     weight = np.random.random((wm, wn)) * 2 - 1
        #     self.weights.append(0.25 * weight)


    def fit(self, X, y, learning_rate=0.2, epochs = 10000):
        X = np.atleast_2d(X)
        # temp = np.ones([X.shape[0], X.shape[1]+1])
        # temp[:,0:-1] = X
        # X = temp
        X = np.column_stack((X, np.ones(len(X))))
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            # 正向计算
            for l in range(len(self.weights)):
                a.append(self.activation( np.dot(a[l], self.weights[l])) )
            # 反向传播
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            # starting backprobagation
            layerNum = len(a) - 2
            for j in range(layerNum, 0, -1): # 倒数第二层开始
                deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]))
                # deltas.append(deltas[-(layerNum+1-j)].dot(self.weights[j].T) * self.activation_deriv(a[j]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a







