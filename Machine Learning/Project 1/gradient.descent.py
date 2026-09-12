# gradient.descent.py
#
# An illustration of basic gradient descent in 2D
# (Not a professional program; just a simple demo)
#
# Looking for minimum of Rosenbrock function
# ( which is (1.0, 1.0) ), with starting point (0, 0).
#
# If the starting point is not close to the minimum,
# the parameters may need to be tweaked ("The Art of
# Tweaking" :)
#
# For m494s24    February 2024

import numpy as np
import matplotlib.pyplot as plt


# Rosenbrock function
def f(x, y):
    return (1-x)*(1-x)+5*(y-x*x)*(y-x*x)

def fx(x, y):
    return 2*(x-1)-20*(y-x*x)*x

def fy(x, y):
    return 10*(y-x*x)

# (Hyper-)parameters of gradient descent:
eta = 0.001         # learning rate
eps = 1e-6          # minimum step size
max_cnt = 10000     # max number of iterations

# Starting point
x = 0.0
y = 0.0

change = 1          # an artificial starting value

# The gradient descent loop
cnt = 0
while (cnt < max_cnt) and (change > eps):
    x_change = -eta*fx(x, y)
    y_change = -eta*fy(x, y)
    x += x_change
    y += y_change
    # see how big a step we made
    change = x_change + y_change
    cnt += 1
    # The following three lines are only for displaying
    # the descent process; they are not really needed

plt.plot(cnt, change, '-', c='red')
plt.title('Graph of f(x)=exp(-0.1*x)*cos(x)')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
