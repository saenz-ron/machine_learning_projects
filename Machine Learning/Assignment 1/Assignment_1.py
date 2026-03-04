import numpy as np
import matplotlib.pyplot as plt

# Defines the function f(x,y,z) = x^2+y^2+z^2-x-y+z
def f(x, y, z):
    return x**2 + y**2 + z**2 - x - y + z

# Defines the gradient of the function
def grad_f(x, y, z):
    df_dx = 2*x - 1
    df_dy = 2*y - 1
    df_dz = 2*z + 1
    return np.array([df_dx, df_dy, df_dz])

# The Gradient Descent function
def grad_descent(eta, max_iter):
    # Initial guess
    x = np.random.uniform(-10, 10)
    y = np.random.uniform(-10, 10)
    z = np.random.uniform(-10, 10)
    
    # Stores the values for plotting
    iterations = []
    changes = []
    
    for i in range(max_iter):
        # Computes the gradient
        grad = grad_f(x, y, z)
        
        x_2, y_2, z_2 = x, y, z
        
        # Updates the parameters
        x -= eta * grad[0]
        y -= eta * grad[1]
        z -= eta * grad[2]
        
        # Stores the iteration and change
        iterations.append(i)
        changes.append(np.sqrt((x - x_2)**2 + (y - y_2)**2 + (z - z_2)**2))
        
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return x, y, z, iterations, changes


max_iter = 1000
etas = np.linspace(0.001, 0.1, 100)
converged_etas = []

for eta in etas:
    x_min, y_min, z_min, iterations, changes = grad_descent(eta, max_iter)
    converged_etas.append(eta)

# Plot the curve illustrating the change of values with respect to the iteration number
plt.figure(figsize=(10, 6))
plt.plot(iterations, changes, c='red')
plt.xlabel('Iteration')
plt.ylabel('Change in values')
plt.title('Change of Values between Consecutive Iterations')
plt.show()

# Print the largest value of step size for which the method converges
largest_eta = max(converged_etas)
print("The largest step size (eta) for which the method converges is:", largest_eta)
