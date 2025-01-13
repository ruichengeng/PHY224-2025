# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:01:29 2025

@author: ruich
"""

import scipy as sp
import math

data=sp.stats.norm.rvs(3,1,100) #1000 random number generated with normal distribution

mean, var = sp.stats.norm.stats(moments='mv')
stds=math.sqrt(var)

print("Mean is ", mean)
print("Standard Deviation is ", stds)