# m494s24 hw9
#
# Toy neural network closely based on the code in the book Neural
# Network Projects with Python, by James Loy.
#
# You will input four bits (thus four input nodes) and output one
# value. You may use an arbitrary number of hidden nodes. The task
# is to teach the network to count the number of ones (e.g., 0111
# should yield 3). With complete input,that is 16 patterns, this is
# an easy task, so you will try omitting some patterns and check
# whether the network generalized.
#
# m494s24   April 2024

import numpy as np
import matplotlib.pylab as plt

eta = xxx
seedy = xxx
NUM_HIDDEN = xxx

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class ToyNN:
    def __init__(self, x, y):
        self.input = x
        self.wts1 = np.random.rand(self.input.shape[1],NUM_HIDDEN)
        self.wts2 = np.random.rand(NUM_HIDDEN, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.wts1))
        self.output = sigmoid(np.dot(self.layer1, self.wts2))

    def backprop(self):
        diff_wts2 = np.dot(self.layer1.T, (2*(self.y - self.output) *
            sigmoid_derivative(self.output)))                                   
        diff_wts1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) *
            sigmoid_derivative(self.output), self.wts2.T) *
            sigmoid_derivative(self.layer1)))

        self.wts1 += eta*diff_wts1
        self.wts2 += eta*diff_wts2

    def check(self, input1):
        self.layer1 = sigmoid(np.dot(input1, self.wts1))
        self.output = sigmoid(np.dot(self.layer1, self.wts2))

if __name__ == "__main__":

    # Enter the set of input patterns here:
    X = np.array(XXX)

    # Enter the set of corresponding outputs
    y = np.array(XXX)

    np.random.seed(seedy)
    
    nn = ToyNN(X, y)

    # Training    
    for j in range(XXX):
        for i in range(XXX):
            nn.feedforward()
            nn.backprop()
            
        # Checking the output of patterns:
        nn.check(XXX)
        result1[j] = 4*nn.output
        nn.check(XXX)
        result2[j] = 4*nn.output 

        
# Display results, etc.
