# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:08:54 2025

Turki Almansoori
Rui (Richard) Chen Geng Li

Code created for PHY224 Free Choice Lab: Electron Mass Charge Ratio
Prof. Sergio De La Barrera
Due April 4th, 2025
"""

#Necessary modules
import numpy as np 
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 


cc_current, cc_voltage, cc_ps_volt, cc_diameter = np.loadtxt("Constant_Current_data.csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)

plt.plot(cc_voltage, (cc_diameter/2.0))