import math
def simpson_rule(f,a,b,a_0):
    return abs(b-a)/6 * (f(a, a_0)+ 4 * f((a + b) / 2, a_0) + f(b, a_0))
def integral(x, a_0):
    return (1 + a_0 * (1 - math.cos(x)) ** 2) / (1 + a_0 * math.sin(x) ** 2) /(1 +  2 * a_0 * (1 - math.cos(x))) ** 0.5
def simpson_adaptive(f,a,b,eps,a_0):
    """
    #calculate the intrgral of function using the adpative simpson method
    #vary the parameter a_0, the intergation is nearly the same
    >>> simpson_adaptive(integral,0.0,math.pi,1e-10,1.0)
    3.1415926535897514

    >>> simpson_adaptive(integral,0.0,math.pi,1e-10,3.0)
    3.1415926535897714

    :param f: function to integrate
    :param a: starting point
    :param b: end point
    :param eps: tolerant error
    :return: the integration
    """
    c = (a + b) / 2
    left = simpson_rule(f,a,c,a_0)
    right = simpson_rule(f,c,b,a_0)
    mid = simpson_rule(f,a,b,a_0)
    if abs(left + right - mid) <= 15 * eps:  #whether the error can be accepted
        return left + right + (left + right - mid) / 15
    return simpson_adaptive(f,a,c,eps/2,a_0) + simpson_adaptive(f,c,b,eps/2,a_0) #loop to get a tolerant error
    pass

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
t = np.arange(0.0,100.0,1.0)
y = []
plt.xscale('linear')
plt.yscale('linear')
ax.set_xlabel('the value of a0')
ax.set_ylabel('integral')
for i in range(100):
    y.append(simpson_adaptive(integral,0.0,math.pi,1e-10,i))
y = np.array(y)
plt.plot(t,y,'r')
plt.show()

