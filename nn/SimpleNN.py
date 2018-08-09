#-*- coding: utf-8 -*-
import random

import numpy as np

from util.MLUtils import sigmoid, softmax, loss_for_softmax, sigmoid_derivative


class SimpleNN:
    input_vectors = np.array([0])
    output_vectors = np.array([0])
    hidden_layers = []
    W1 = np.array([0]) # input layer와 hidden layer 사이의 가중치 매트릭스
    W2 = np.array([0]) # hidden layer와 output layer 사이의 가중치 매트릭스
    b1 = np.array([1.0]) # input layer의 bias항
    b2 = np.array([1.0]) # hidden layer의 bias항

    def __init__(self, input_vectors, output_vectors):
        self.input_vectors = input_vectors
        self.output_vectors = output_vectors

        self.initialize_weight_matrix()

    def initialize_weight_matrix(self):
        hidden_node_cnt = len(self.input_vectors[0])
        output_node_cnt = len(self.output_vectors[0])

        self.W1 = np.random.random((hidden_node_cnt + 1, hidden_node_cnt))
        self.W2 = np.random.random((hidden_node_cnt + 1, output_node_cnt))
        # self.W1 = np.zeros((hidden_node_cnt + 1, hidden_node_cnt))
        # self.W2 = np.zeros((hidden_node_cnt + 1, output_node_cnt))

    def train(self, learning_rate=0.01, epoch=100):
        for _ in range(epoch):
            forward_result_list = self.feedforward_for_all()
            self.backpropagation_for_all(forward_result_list, learning_rate)
            self.hidden_layers = []


    def feedforward_for_all(self):
        out_list = []

        for input_vector in self.input_vectors:
            out_list.append(self.feedforward(input_vector))

        return np.array(out_list)

    def feedforward(self, input_vector):
        hidden_layer_values = np.dot(np.append(input_vector, self.b1), self.W1)
        hidden_layer_values = np.array([sigmoid(x) for x in hidden_layer_values])

        self.hidden_layers.append(hidden_layer_values)

        output_layer_values = np.dot(np.append(hidden_layer_values, self.b2), self.W2)
        output_layer_values = np.array([sigmoid(x) for x in output_layer_values])

        return softmax(output_layer_values)

    def backpropagation_for_all(self, expected_output_list, learning_rate=0.01):
        for i, expected_output in enumerate(expected_output_list):
            self.backpropagation(expected_output, i, learning_rate)

    def backpropagation(self, expected_output, data_idx, learning_rate):
        output_delta = loss_for_softmax(expected_output, self.output_vectors[data_idx])
        hidden_delta = np.dot(output_delta, self.W2.T) * np.append(sigmoid_derivative(self.hidden_layers[data_idx]), self.b2)

        # learning part = Gradient Descent
        self.W2 -= learning_rate * np.dot(np.array([np.append(self.hidden_layers[data_idx], self.b2)]).T, np.array([output_delta]))
        self.W1 -= learning_rate * np.dot(np.array([np.append(self.input_vectors[data_idx], self.b1)]).T, np.array([hidden_delta[:-1]]))


if __name__ == "__main__":
    inputs = []
    outputs = []

    for _ in range(100):
        if random.random() >= 0.5:
            inputs.append([1, 0, 0])
            outputs.append([1, 0])
        else:
            inputs.append([0, 0, 1])
            outputs.append([0, 1])

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    nn = SimpleNN(inputs, outputs)
    nn.train()

    print(nn.feedforward(np.array([1, 0, 0])))
