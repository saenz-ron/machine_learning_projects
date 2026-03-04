# Visualization of f(x,y) with a 3D scatter plot
# Visualization of the gradient of f(x,y)
# Adapted from Kneusel, Ch.7
#
# f(x,y) = x^2+xy+y^2
# 
# For m494s24, January 2024

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt

# Create the grid
x = np.linspace(-1.0,1.0,50)
y = np.linspace(-1.0,1.0,50)
xx = []
yy = []
zz = []

# Compute points for scatter plot
for i in range(50):
    for j in range(50):
        xx.append(x[i])
        yy.append(y[j])
        zz.append(x[i]*x[i]+x[i]*y[j]+y[j]*y[j])
x = np.array(xx)
y = np.array(yy)
z = np.array(zz)

# Plot the surface with the use of scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x, y, z, marker='.', s=2, color='b')
ax.scatter(x, y, z, marker='o', s=10, c=z, cmap='rainbow')
ax.view_init(30, 50)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
ax.set_title("Scatter 3D graph of f(x, y) = x^2 + xy + y^2")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
# If you want to save graph as an image, uncomment the next line 
# plt.savefig("3Dscatter.graph.jpg", dpi=300)

# Use quiver plot in 2D to visualize the gradient of f(x, y)
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.linspace(-1.0,1.0,20)
y = np.linspace(-1.0,1.0,20)
xv, yv = np.meshgrid(x, y, indexing='ij', sparse=False)
dx = 2*xv + yv
dy = 2*yv + xv
ax.quiver(xv, yv, dx, dy, color='b')
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Gradient field of f(x, y) = x^2 + xy + y^2")
plt.axis('equal')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
# If you want to save graph as an image, uncomment the next line
# plt.savefig("2Dgradient.jpg", dpi=300)
plt.show()

