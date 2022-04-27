"""

backprop_network.py

"""


import random

import numpy as np

import math


class Network(object):
    def __init__(self, sizes):

        """The list ``sizes`` contains the number of neurons in the

        respective layers of the network.  For example, if the list

        was [2, 3, 1] then it would be a three-layer network, with the

        first layer containing 2 neurons, the second layer 3 neurons,

        and the third layer 1 neuron.  The biases and weights for the

        network are initialized randomly, using a Gaussian

        distribution with mean 0, and variance 1.  Note that the first

        layer is assumed to be an input layer, and by convention we

        won't set any biases for those neurons, since biases are only

        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)

        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data):

        """Train the neural network using mini-batch stochastic

        gradient descent.  The ``training_data`` is a list of tuples

        ``(x, y)`` representing the training inputs and the desired

        outputs."""

        print("Initial test accuracy: {0}".format(self.one_label_accuracy(test_data)))

        n = len(training_data)

        ###### for section b ######
        test_accuracy = np.zeros(epochs)
        training_accuracy = np.zeros(epochs)
        training_loss = np.zeros(epochs)
        ##############################

        for j in range(epochs):

            random.shuffle(training_data)

            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:

                self.update_mini_batch(mini_batch, learning_rate)

            print(
                "Epoch {0} test accuracy: {1}".format(
                    j, self.one_label_accuracy(test_data)
                )
            )

            ###### for section b ######
            test_accuracy[j] = self.one_label_accuracy(test_data)
            training_accuracy[j] = self.one_hot_accuracy(training_data)
            training_loss[j] = self.loss(training_data)
            #############################

        ###### for section c ######
        print("Final test accuracy: {0}".format(self.one_label_accuracy(test_data)))
        ##############################
        return test_accuracy, training_accuracy, training_loss

    def update_mini_batch(self, mini_batch, learning_rate):

        """Update the network's weights and biases by applying

        stochastic gradient descent using backpropagation to a single mini batch.

        The ``mini_batch`` is a list of tuples ``(x, y)``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            w - (learning_rate / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]

        self.biases = [
            b - (learning_rate / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
        The function receives as input a 784 dimensional
        vector x and a one-hot vector y.
        The function should return a tuple of two lists (db, dw)
        as described in the assignment pdf
        """
        L = self.num_layers
        v, z = self.forward_pass(x)
        deltas = self.backward_pass(v, y)
        db = []
        dw = []
        # calculate backprop
        for l in range(L - 1):
            # db[l] = δ_l ◦ h′(v_l)
            # dw[l] = ∂ℓ / ∂W_l = (δ_l ◦ h′(v_l)) * transpose(z_l−1)
            delta_l = deltas[l]
            h_tag_v_l = relu_derivative(v[l])
            z_l1_T = np.transpose(z[l])
            db.append(delta_l * h_tag_v_l)
            dw.append((delta_l * h_tag_v_l) @ z_l1_T)

        return db, dw

    def forward_pass(self, x):
        """
        For each layer calculate the linear output v_L = W_L * z_L + b_L
        and non-linear activation z_l = σ(v_l)
        """
        L = self.num_layers
        v = []
        z = [np.copy(x)]
        # calculate forward pass
        for l in range(L - 1):
            # v_l+1 = W_l+1 * z_l + b_l+1)
            # z_l+1 = h(v_l+1)
            v_l1 = self.weights[l] @ z[-1] + self.biases[l]
            v.append(v_l1)
            z.append(relu(v_l1))

        return v, z

    def backward_pass(self, v, y):
        """
        Define δ_l = WT_l+1 * (h′(v_l+1) ◦ δl+1).
        Calculate recursively δ_L, δ_L-1, ...
        """
        L = self.num_layers
        delta = [self.loss_derivative_wr_output_activations(v[-1], y)]
        for i in reversed(range(L - 2)):
            delta.insert(
                0, np.transpose(self.weights[i + 1]) @ delta[0] * relu_derivative(v[i])
            )
        return delta

    def one_label_accuracy(self, data):

        """Return accuracy of network on data with numeric labels"""
        output_results = [
            (np.argmax(self.network_output_before_softmax(x)), y) for (x, y) in data
        ]

        return sum(int(x == y) for (x, y) in output_results) / float(len(data))

    def one_hot_accuracy(self, data):
        """Return accuracy of network on data with one-hot labels"""
        output_results = [
            (np.argmax(self.network_output_before_softmax(x)), np.argmax(y))
            for (x, y) in data
        ]

        return sum(int(x == y) for (x, y) in output_results) / float(len(data))

    def network_output_before_softmax(self, x):
        """Return the output of the network before softmax if ``x`` is input."""
        layer = 0
        for b, w in zip(self.biases, self.weights):
            if layer == len(self.weights) - 1:
                x = np.dot(w, x) + b
            else:
                x = relu(np.dot(w, x) + b)
            layer += 1

        return x

    def loss(self, data):
        """Return the loss of the network on the data"""
        loss_list = []
        for (x, y) in data:
            net_output_before_softmax = self.network_output_before_softmax(x)
            net_output_after_softmax = self.output_softmax(net_output_before_softmax)
            loss_list.append(
                np.dot(-np.log(net_output_after_softmax).transpose(), y).flatten()[0]
            )

        return sum(loss_list) / float(len(data))

    def output_softmax(self, output_activations):
        """Return output after softmax given output before softmax"""

        output_exp = np.exp(output_activations)

        return output_exp / output_exp.sum()

    def loss_derivative_wr_output_activations(self, output_activations, y):
        """Return derivative of loss with respect to the output activations before softmax"""
        return self.output_softmax(output_activations) - y


def relu(z):
    return z * (z > 0)


def relu_derivative(z):
    return 1.0 * (z > 0)
