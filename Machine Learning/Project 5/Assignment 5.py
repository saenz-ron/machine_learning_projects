import numpy as np
import matplotlib.pyplot as plt

a = 7
data_set = 100

b = int(input("Enter a number: "))

#  Generate a dataset 
x_axis = np.random.uniform(0,5,data_set)
y_axis = a * np.e**(x_axis*b)

# Add substantial Gaussian noise to y_axis
gauss_y_axis = np.random.normal(0, np.mean(y_axis), data_set)
y_axis += abs(gauss_y_axis)

# Display
plt.figure(figsize=(10, 6))
plt.scatter(x_axis, y_axis, c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Assignment 5 plot')
plt.show()

# Linearize
new_y = np.log(y_axis)

# Compare with original values
sumxlny = np.sum(x_axis * new_y)
sumx = np.sum(x_axis)
sumxsq = np.sum(x_axis**2)
sumlny = np.sum(new_y)

a_numerator =  (data_set * sumxlny) - (sumx * sumlny)
a_denominator = (data_set * sumxsq) - (sumx ** 2)
new_a = np.e**(a_numerator / a_denominator)
new_b = (sumlny - new_a * sumx) / data_set

print("The original value for 'a' : " + str(a))
print("The predicted value for 'a' : " + str(new_a))
print("The original value for 'b' : " + str(b))
print("The predicted value for 'b' : " + str(new_b))

