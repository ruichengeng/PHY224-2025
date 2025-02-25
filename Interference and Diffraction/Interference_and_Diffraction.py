# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:34:14 2025

Interference and Diffraction

Turki Almansoori
Rui (Richard) Chen Geng Li

Code created for PHY224 Free Choice Lab: Interference and Diffraction
Prof. Sergio De La Barrera
Due March 3rd, 2025
"""

#Necessary modules
import numpy as np 
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 

position, intensity = np.loadtxt("single_slit_data_pos_100x.txt", 
                                                  delimiter = '\t', 
                                                  skiprows=2, unpack=True)


max_index = 0   
max_intensity = 0.0

for i in range(len(intensity)):
    if intensity[i] > max_intensity:
        max_index = i
        max_intensity = intensity[i]

def id_model(x_val, a, b, c, d):
    return a*np.sin(b*x_val+c)/(x_val+d)
    

# def id_model(x_val, a, b, c):
#     return a*np.sin(b*(x_val-position[max_index])/(c*(x_val-position[max_index])))

popt, pcov = curve_fit(id_model, position, intensity, maxfev = 100000)
    
plt.plot(position, intensity)
plt.plot(position, id_model(position, popt[0], popt[1], popt[2], popt[3]), color = "red")
# plt.plot(position, id_model(position, popt[0], popt[1], popt[2]), color = "red")