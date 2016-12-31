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
        change_in_b = [np.zeros(b.shape) for b in self.biases]
        change_in_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # for each training example calculate delta_b and delta_w
            delta_b, delta_w = self.backprop(x, y)
            change_in_b = [nb + dnb for nb, dnb in zip(change_in_b, delta_b)]
            change_in_w = [nw + dnw for nw, dnw in zip(change_in_w, delta_w)]
        # update weights and biases
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, change_in_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, change_in_b)]

    def backprop(self, x, y):
        # backprop for single training example
        change_in_b = [np.zeros(b.shape) for b in self.biases]
        change_in_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        # input layer is first activation a1
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        change_in_b[-1] = delta
        change_in_w[-1] = np.dot(delta, activations[-2].transpose())
        # -l is the lth layer from backward.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            change_in_b[-l] = delta
            change_in_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (change_in_b, change_in_w)

    def evaluate(self, test_data):
        # do a feedforward for each test sample
        # take maximum value as prediction by the network
        # match it with actual output
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

def sigmoid(z):
    # sigmoid function
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    # sigmoid derivative
    return sigmoid(z) * (1 - sigmoid(z))
