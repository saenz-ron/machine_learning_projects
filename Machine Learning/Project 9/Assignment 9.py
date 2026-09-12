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
from numpy import random
import matplotlib.pylab as plt

# Tweak the learning rate if needed
eta = .11
seedy = 4132

# Can be any amount of hidden nodes but the fewer the better. For now lets do 6
NUM_HIDDEN = 6

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class ToyNN:
    def __init__(self, x, y):
        self.input = x
        # print(self.input.shape)
        self.wts1 = np.random.rand(self.input.shape[1], NUM_HIDDEN)
        self.wts2 = np.random.rand(NUM_HIDDEN, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        # print(y)
        # print(self.wts2)


    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.wts1))
        self.output = sigmoid(np.dot(self.layer1, self.wts2))
        # print(self.layer1)

    def backprop(self):
        diff_wts2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))                                   
        diff_wts1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.wts2.T) * sigmoid_derivative(self.layer1)))

        self.wts1 += eta*diff_wts1
        self.wts2 += eta*diff_wts2

    def check(self, input1):
        self.layer1 = sigmoid(np.dot(input1, self.wts1))
        self.output = sigmoid(np.dot(self.layer1, self.wts2))

if __name__ == "__main__":

    np.random.seed(seedy)

    # Enter the set of input patterns here:
    # IE every possible input node
    x = np.array([[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0], [0,0,1,1], [0,1,1,0], [1,1,0,0], [1,0,0,1], [1,0,1,0], [0,1,0,1], [0,1,1,1], [1,0,1,1], [1,1,0,1], [1,1,1,0], [1,1,1,1]])

    # Enter the set of corresponding outputs
    # IE every possible output node
    y = np.array([[0],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[3],[3],[3],[3],[4]])
    y = y/4
    
    
    nn1 = ToyNN(x, y)

    # Training
    # Replace 'XXX' with the amount of iterations you want
    for j in range(1000):
        nn1.feedforward()
        nn1.backprop()
    \
    # Display results, etc
    # The program should display all the parameters that you used: 
    # The learning rate, the number of hidden nodes, and seed of the PRNG, if you use seeding

    
    print("Neural Network Parameters:\nThe Learning Rate is:", eta)
    print("The Seed is:", seedy)
    print("The number of Hidden Nodes is:", NUM_HIDDEN)

    # Checking the output of patterns:
    nn1.check([1, 1, 0, 1])
    result1 = 4*nn1.output
    # Should be close to 3.0 if the network is trained

    nn1.check([0,1,0,0])
    result2 = 4*nn1.output
    # Should be close to 1.0 if the network is trained

    # Results without removing entries
    print("\nPredictions with all data being used in the training set")
    print("Prediction for [1,1,0,1] is:", result1)
    print("Prediction for [0,1,0,0] is:", result2)

    # Finding the relative error across the entire dataset   
    relative_err = 0
    for n in range(x.shape[0]):
        nn1.check(x[n])
        result = 4*nn1.output
        if n == 0:
            pass
        else:
            relative_err += np.abs((4*y[n] - result)/ (4*y[n])) *100

    average_relative_err = relative_err/(x.shape[0] -1)
    print("Average relative error across all data, but excluding target = 0 (%):", average_relative_err)

    # Removing an entry of data
    x = np.array([[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0], [0,0,1,1], [0,1,1,0], [1,1,0,0], [1,0,0,1], [1,0,1,0], [0,1,0,1], [1,0,1,1], [1,1,0,1], [1,1,1,0], [1,1,1,1]])

    y = np.array([[0],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[3],[3],[3],[4]])
    y = y/4

    # Training
    nn2 = ToyNN(x,y)    
    for j in range(1000):
        nn2.feedforward()
        nn2.backprop()

    # Checking the output of patterns:
    nn2.check([0,1,1,1])
    result3 = 4*nn2.output
    # Should be close to 3.0

    nn2.check([0,1,1,0])
    result4 = 4*nn2.output
    # Should be close to 2.0

    # Display these reults
    print("\nPredictions with 1 missing data entry ([0,1,1,1]):")
    print("Prediction for [0,1,1,1] is:", result3)
    print("Prediction for [0,1,1,0] is:", result4)

    # Finding the relative error across the entire dataset
    relative_err = 0
    for n in range(x.shape[0]):
        nn2.check(x[n])
        result = 4*nn2.output
        if n == 0:
            pass
        else:
            relative_err += np.abs((4*y[n] - result)/ (4*y[n])) *100

    average_relative_err = relative_err/(x.shape[0] -1)
    print(f"Average relative error across all data, but excluding target = 0 (%): {average_relative_err}")

    
    # Removing 2 entries of data
    x = np.array([[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0], [0,0,1,1], [0,1,1,0], [1,1,0,0], [1,0,0,1], [0,1,0,1], [0,1,1,1], [1,0,1,1], [1,1,0,1], [1,1,1,0]])

    y = np.array([[1],[1],[1],[1],[2],[2],[2],[2],[2],[3],[3],[3],[3],[4]])
    y = y/4

    # Training
    nn3 = ToyNN(x,y)    
    for j in range(1000):
        nn3.feedforward()
        nn3.backprop()

    # Checking the output of patterns:
    nn3.check([1,1,1,1])
    result5 = 4*nn3.output
    # Should be close to 4.0, but it's also the only in the dataset at 4.0

    nn3.check([1,0,1,0])
    result6 = 4*nn3.output
    # Should be close to 2.0

    # Display these reults
    print("\nPredictions with 2 missing data entries ([1,1,1,1] and [1,0,1,0]):")
    print("Prediction for [1,1,1,1] is:", result5)
    print("Prediction for [1,0,1,0] is:", result6)

    # Finding the relative error across the entire dataset
    relative_err = 0
    for n in range(x.shape[0]):
        nn3.check(x[n])
        result = 4*nn3.output
        if n == 0:
            pass
        else:
            relative_err += np.abs((4*y[n] - result)/ (4*y[n])) *100

    average_relative_err = relative_err/(x.shape[0] -1)
    print(f"Average relative error across all data , but excluding target = 0 (%): {average_relative_err}")