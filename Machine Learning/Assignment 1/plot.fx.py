# The script visualizes function f(x) Demo of the extremely
# basic use of matplotlib. Zillion options are available.
#
# Graph f(x) = exp(-0.1*x) * cos(x) for 0 < x < 50
#
# For m494s24, January 2024
#

import numpy as np
import matplotlib.pyplot as plt

# Define the domain of x and compute y
x = np.arange(0, 50, 0.5)
y = np.exp(-0.1*x) * np.cos(x)

plt.plot(x, y, '-', c='red')
plt.title('Graph of f(x)=exp(-0.1*x)*cos(x)')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
