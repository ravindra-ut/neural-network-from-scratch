import numpy as np
import random

class Network(object):

    def __init__(self, sizes):
        """
        :param sizes: list containing number of neurons per layer
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # random randn generates random numbers with gaussian
        # distribution with mean 0 and standard deviation 1.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        # a is input nd-array (n, 1)
        # z = w.a + b
        # activation = sigmoid(z)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size,
            learning_rate, test_data=None):
        # epochs are number of times to train the network
        if test_data:
            test_len = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print "Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data),
                                                  test_len)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, learning_rate):
        pass

    def backprop(self, x, y):
        pass

    def evaluate(self, test_data):
        pass

    def cost_derivative(self, output_activations, y):
        pass

def sigmoid(z):
    # sigmoid function
    return 1.0/(1.0 + np.exp(-z))

def sigmoid(z):
    # sigmoid derivative
    return sigmoid(z) * (1 - sigmoid(z))
