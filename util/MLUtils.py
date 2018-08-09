#-*- coding: utf-8 -*-
import numpy as np

#################################################### activation functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def relu(x):
    return max(0.0, x)


def relu_derivative(x):
    if x <= 0.0:
        r = 0.0
    else:
        r = 1.0

    return r


#################################################### softmax and loss functions

def softmax(X):
    exp_list = [np.exp(x) for x in X]
    s = sum(exp_list)

    return [x / s for x in exp_list]


def loss_for_softmax(V1, V2):
    return V1 - V2

