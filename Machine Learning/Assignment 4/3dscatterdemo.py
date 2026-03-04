# Simplest 3D scatterplot demo
#
# m494s24, February 2024
#

import numpy as np
import matplotlib.pyplot as plt

# Number of points in the cloud
n = 100

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Generate and display 3D cloud of random points in U(0, 1) cube
x = np.random.rand(n)
y = np.random.rand(n)
z = np.random.rand(n)
ax.scatter(x, y, z, marker='o', color='magenta')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Cube of U(0, 1) points')
plt.show()
