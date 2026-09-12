# A rudimentary sympy demo
#
# For more info, see:
# https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html
#
# For m494s24  January 2024

from sympy import *

x, y, z = symbols('x y z')

# Differentation:
print('The derivative of cos(x) is', diff(cos(x), x))
print('The derivative of e^(x^2) is', diff(exp(x**2), x))
print('The second derivative of e^(x^2) is', diff(exp(x**2), x, x))

f = x**2 * y**2 * z**2
print('The fx partial for f = (x^2)*(y^2)*(z^2) is', diff(f, x))
print('The fxy mixed partial for f = (x^2)*(y^2)*(z^2) is', \
      diff(f, x, y))
print('The fxyz mixed partial for f = (x^2)*(y^2)*(z^2) is', \
      diff(f, x, y, z))

# Integration:
print('The antiderivative of cos(x) is', integrate(cos(x), x))
print('The integral of e^(-x) from 0 to infinity is ', \
      integrate(exp(-x), (x, 0, oo)))
print('The double int of e^[-x^2-y^2] from -inf to inf on x, y is ', \
      integrate(exp(-x**2 - y**2), (x, -oo, oo), (y, -oo, oo)))

# Series expansion:
fun = exp(sin(x))
print('The sixth Maclaurin polynomial for e^(sin(x)) is \n', \
      fun.series(x, 0, 6))

# And much, much more...

# The above script produces:
# The derivative of cos(x) is -sin(x)
# The derivative of e^(x^2) is 2*x*exp(x**2)
# The second derivative of e^(x^2) is 2*(2*x**2 + 1)*exp(x**2)
# The fx partial for f = (x^2)*(y^2)*(z^2) is 2*x*y**2*z**2
# The fxy mixed partial for f = (x^2)*(y^2)*(z^2) is 4*x*y*z**2
# The fxyz mixed partial for f = (x^2)*(y^2)*(z^2) is 8*x*y*z
# The antiderivative of cos(x) is sin(x)
# The integral of e^(-x) from 0 to infinity is  1
# The double int of e^[-x^2-y^2] from -inf to inf on x, y is  pi
# The sixth Maclaurin polynomial for e^(sin(x)) is 
# 1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)
