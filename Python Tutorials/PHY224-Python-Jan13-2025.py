# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:14:38 2025

@author: ruich
"""

import numpy as np
a=1
b = np.array([3.0,2.3,1.0])
c = np.array([3,.3,.03])
d = np.array([[2,4],[4,6],[7,8]])
f = [9,90,900]

c=np.reshape(c,(3,1))
print((c,(3,1)))

t= np.array([0. , 0.1155, 0.2287, 0.3404, 0.4475, 0.5546, 0.6607, 0.7753, 0.8871, 1. ])
y= np.array([ 0. , 0.1655, 0.2009, 0.1124, -0.0873, -0.3996, -0.8197, -1.3977, -2.0856, -2.905 ])

print('mean of all values in y =', np.mean(y))

print('sample standard deviation of values in y =', np.std(y,ddof=1))

dydt=np.diff(y)/np.diff(t)

the_integral_of_energy = 0 
j = 0 
for j in range(9): 
    delta_t = t[j+1]-t[j]
    the_integral_of_energy = the_integral_of_energy + 0.5 * (y[j]**2)*delta_t
    j=j+1
    
print("The integral of energy is ", the_integral_of_energy)