# allsmiles.py
#
# The script generates a dataset of noisy 5x5 images of three types:
# - happy smiley
# - sad smiley
# - poker (neutral)
#
# Then the Scikit-learn's MLPClassifier is used to divide the
# dataset into training and testing partitions and to train the
# MLP to classify the patterns. The results of the testing are
# displayed.
# You may also display some testing patterns (not required, but
# you may be interested in visualization) by uncommenting the
# last fragment of the scipt.
#
# The noise is implemented with thresholds of uniform PRNGs.
#
# This is a brutally barebones, painfully crude, and blindingly
# unsophisticated script. Setting noisiness is particularly clumsy.
#
# For m494s24
# May 2024
#

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import warnings
# For graphics:
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

size = 2000
test_perc = 0.3

# Set the thresholds of "noise", that is the degree of randomness
# in the "images". 0.4 - 0.6 for thresh1, and
# 1.0 for thresh2 seem to give stable/reasonable results.
# Play with the values! 
# 
thresh1 = 0.5
thresh2 = 1.0

# Construct the dataset:
smile = {6, 7, 8, 10, 14, 15, 19}
frown = {5, 9, 10, 14, 16, 17, 18}
poker = {11, 12, 13}
X = np.zeros((size, 25))
y = np.zeros((size,))
for i in range(size):
    X[i,:] = np.random.uniform(0, 1, 25)
    ran = np.random.rand()
    for j in range(25):
        if ran < 0.33:
            if j in smile:
                X[i, j] = np.random.uniform(low=thresh1, high=thresh2)
        elif ran < 0.67:
            if j in frown:
                X[i, j] = np.random.uniform(low=thresh1, high=thresh2)
        else:
            if j in poker:
                X[i, j] = np.random.uniform(low=thresh1, high=thresh2)
    if ran < 0.33:
        y[i] = 0
    elif ran < 0.67:
        y[i] = 1
    else:
        y[i] = 2
        
# Split data into train partition and test partition:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, \
    test_size=test_perc)

# Set parameters:
activation_type = 'relu'
solver_type = 'lbfgs'
max_iter_val = 500

# Play with this value (and, possibly, with the number of layers)
num_nodes = 10

# Construct the MLP Classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(num_nodes,),
    activation=activation_type,
    max_iter=max_iter_val,
    alpha=1e-4,
    solver=solver_type,
    verbose=0,
    learning_rate_init=0.2,
)

#mlp.fit(X_train, y_train)

# Fit the MLPClassifier catching a warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, \
        module="sklearn")
    mlp.fit(X_train, y_train)

# Obtain the confusion matrix
cm = confusion_matrix(y_test,mlp.predict(X_test))
print("Confusion matrix:\n")
print(cm)

# Print summary results
print("\nTraining set score: %f" % mlp.score(X_train, y_train))
print("\nTest set score: %f" % mlp.score(X_test, y_test))

# Uncomment the following fragment to see the "images"

# Just for fun, display the first three smiley "images" from the test
# partition
for k in range(3):
    fig = plt.figure()
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    currentAxis = plt.gca()
    for i in range(5):
        for j in range(5):
            col = (X_test[k,5*j+i])
            currentAxis.add_patch(Rectangle((2*i, 2*j), 2, 2, \
                facecolor=(col, col, col)))
    plt.show()

