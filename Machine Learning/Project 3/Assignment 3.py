# Author: Ronnie Saenz
# Assignment 3
# 2/26/23

'''Notes
The  goal of this assignment is to familiarize you with the practice of pseudo-random numbers generation. PRNGs 
are extensively used in machine learning for creating synthetic datasets to test various ML methods.
 
Generate M (the way to find M is explained below) pseudo-random numbers (PRNs) from the distribution U(0, 1) 
available in numpy, and find their sum. Repeat this N times, where N is a large number, for instance, 10000 or 100000. 
This way, you will obtain N numbers (sums). Obtain a histogram or a plot that illustrates their distribution. 
Compare it - preferably on the same plot - with the normal distribution, which you will obtain by showing the histogram 
(or plot) of N PRNs from the normal distribution generator in numpy, with appropriate parameters (mu and sigma). 
In the script comments, explain how you know what the appropriate parameters are.

 

M should be the sum of the last four digits of your USD ID #. 
I recommend using histograms, with options  bins=50  and  histtype='step'. 
I am sending you the code of  m494s24c9demo.py,  to help with the basics of PRN generation.

'''

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# Seed the generator
mySeed = 5

# m is the amount of PRNs that will be added together and 
# the sum of the last four digits of my USD ID
m = 6+8+5+7

# n is the amount of sums that will be added to the histogram
n = 10000

random.seed(mySeed)

for i in range(m):
    # The U(0, 1) PNRG:2
    r = random.rand()
    print('\nA PRN from U(0, 1): ', r)

# creates the spread for the prng and normal histogram
prng_data = np.random.randn(n)
# loc and scale are 0 and 1 because they are directly translated from U(0,1)
normal_data = np.random.normal(loc=0, scale=1, size=n)

# This function represents the PRNG histogram
plt.hist(prng_data, bins=100, histtype='step', color='blue', edgecolor='red')

# This function represents the normal histogram
plt.hist(normal_data, bins=100, histtype='step', color='blue', edgecolor='blue')

plt.title('PRNG vs Normal Function Histogram')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
