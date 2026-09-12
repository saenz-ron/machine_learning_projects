# The script visualizes function f(x,y), that is a surface in 3D
#
# Adapted from https://juejung.github.io/ for its simplicity
#
# For m494s24, January 2024
#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# For more advanced plots you might need the seaborn library
# import seaborn as sns

# Set the figure size and projection type
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Define grids in x and y dimension
xv = np.arange(-3, 3, 0.05)
yv = np.arange(-3, 3, 0.05)

# Span meshgrid over entire x/y plane
X, Y = np.meshgrid(xv, yv)

# Evaluate function at each point in the x/y plane
sq = X**2.0 + Y**2.0
Z = np.exp(sq/50.0)*np.cos(0.6*sq)

# Plot the result
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, \
    cmap = plt.cm.jet, linewidth=0, antialiased=False)
ax.set_title( \
    'Graph of f(x, y) = exp[-0.02*(x^2+y^2)]*cos(0.6*(x^2+y^2))')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.show()
