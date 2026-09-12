import numpy as np


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
    change = np.linalg.norm(np.array([x_change, y_change]))
    cnt += 1
    # The following three lines are only for displaying
    # the descent process; they are not really needed
    v = f(x, y)
    print(x, y, v, change)
    if cnt % 500 == 0:
        print(x, y, v, change)
