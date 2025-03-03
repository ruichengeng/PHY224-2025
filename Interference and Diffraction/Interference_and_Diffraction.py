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

position, intensity = np.loadtxt("double_slit_data_100x.txt", 
                                                  delimiter = '\t', 
                                                  skiprows=2, unpack=True)


count = int(position.size/3)
position = position[count:-count]
intensity = intensity[count:-count]

max_index = 0   
max_intensity = 0.0

wavelength = 650e-9    # 650 nm in meters

d = 0.25e-3  # Slit separation: 0.25 mm in meters
w = 0.04e-3  # Slit width: 0.04 mm in meters

for i in range(len(intensity)):
    if intensity[i] > max_intensity:
        max_index = i
        max_intensity = intensity[i]

def id_model(x_val, a, b, c):
    #return a*np.sin(b*x_val+c)/(x_val+d)
    return a*np.sin(b*x_val)/(c*x_val)

# def temp_model(x_val, a, b, c, d, e):
#     return a*((np.sin(b*x_val)/(d*x_val))**2)*(np.cos(e*x_val)**2)
# def temp_model(x_val, a, b):
#     return max_intensity*((np.sin(a*(x_val-position[max_index]))/(b*(x_val-position[max_index])))**2)
    

def temp_model(x_val, a, b):
    phi = np.sin(b)*np.pi*a/wavelength
    return max_intensity*((np.sin(phi)/(phi))**2)

def double_slit_model(x_val, I0, k):
    theta = np.arcsin((x_val - position[max_index]) * k)
    
    phi = np.pi * w * np.sin(theta) / wavelength
    phi = np.where(phi == 0, 1e-9, phi)    
    beta = np.pi * d * np.sin(theta) / wavelength

    single_slit = (np.sinc(phi / np.pi)) ** 2
    double_slit = np.cos(beta) ** 2

    return I0 * single_slit * double_slit


# def id_model(x_val, a, b, c):
#     return a*np.sin(b*(x_val-position[max_index])/(c*(x_val-position[max_index])))

# popt, pcov = curve_fit(id_model, position, intensity, maxfev = 100000)

# popt, pcov = curve_fit(temp_model, position, intensity, p0=(400.0, 400.0), maxfev = 100000)
#popt, pcov = curve_fit(temp_model, position, intensity, p0=(0.0004, 1.0), maxfev = 100000)

popt, pcov = curve_fit(double_slit_model, position, intensity, p0=(max_intensity, 1.0), maxfev = 10000)

plt.figure(figsize=(15, 8))
plt.errorbar(position, intensity, fmt='o', color = "red", label = "Measured Data")
# plt.plot(position, id_model(position, popt[0], popt[1], popt[2], popt[3]), color = "red")
#plt.plot(position, temp_model(position, popt[0], popt[1], popt[2], popt[3], popt[4]), color = "red")

# plt.plot(position, temp_model(position, popt[0], popt[1]), color = "red")
# plt.plot(position, max_intensity*(np.sin(400*(position-position[max_index]))/(400*(position-position[max_index])))**2)

plt.plot(position, double_slit_model(position, *popt), color = "blue", label = "Fitted Data")

# plt.plot(position, id_model(position, popt[0], popt[1], popt[2]), color = "red")
plt.xlabel("Position (m)")
plt.ylabel("Intensity (V)")
plt.legend()
plt.show()