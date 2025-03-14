# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:12:00 2025

Turki Almansoori
Rui (Richard) Chen Geng Li

Code created for PHY224 Free Choice Lab: Interferometer
Prof. Sergio De La Barrera
Due March 23rd, 2025
"""

#Necessary modules
import numpy as np 
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 

#Data imports
#calibration interferometer data for python
reading, dN, reading_unc, dN_unc = np.loadtxt("calibration interferometer data for python.csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)

#Prediction Model
#Wavelength
def deltaN(x_val, a):
    return x_val*(2.0/a)

#Thermal Coefficient of Aluminium


dN_Per_dx = np.zeros(dN.size)

for n in range(1, len(dN)):
    dN_Per_dx[n]=dN[n]+dN_Per_dx[n-1]
    
#Curve_fit


#Plotting
plt.errorbar(reading, dN_Per_dx, xerr=reading_unc, yerr=dN_unc)



###################### Index of Refraction ################################

#Reading Data


#Variables
gamma = 650e-9 #Temp for the wavelength

#Prediction Model
#Index of Refraction
def index_refraction(x_val, a, b):
    #a is t for thickness
    #b is theta
    return (((x_val*gamma/(2.0*a))+np.cos(b)-1)**2 + np.sin(b)**2)/(2.0*(1.0-np.cos(b)-(x_val*gamma/(2.0*a))))




###################### Thermal Expansion of Aluminium ################################

#Reading Data


#Variables
L0=1.0 #Temp, length at base temperature

#Prediction Model
#Thermal Expansion
def thermal_expansion(x_val, a):
    return L0*np.exp()**(a*x_val)