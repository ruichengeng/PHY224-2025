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

# position, intensity = np.loadtxt("double_slit_a0.04_d0.5_100x.txt", 
#                                                   delimiter = '\t', 
#                                                   skiprows=2, unpack=True)
position, intensity = np.loadtxt("double_slit_data_100x.txt", 
                                                  delimiter = '\t', 
                                                  skiprows=2, unpack=True)

# position= position[int(position.size/2):]
# intensity = intensity[int(intensity.size/2):]
# count = int(position.size/3)
# position = position[:-count]
# intensity = intensity[:-count]

count = int(position.size/3)
position = position[count:-count]
intensity = intensity[count:-count]

max_index = 0   
max_intensity = 0.0
max_position = 0.0

wavelength = 650e-9    # 650 nm in meters
wheel_sensor_dist = 0.515 #distance between the light sensor and the slit wheel

a = 0.04e-3  # Slit width: 0.04 mm in meters
d = 0.25e-3  # Slit separation: 0.25 mm in meters
# d = 0.5e-3  # Slit separation: 0.25 mm in meters
# a = 0.04e-3  # Slit width: 0.04 mm in meters

for i in range(len(intensity)):
    if intensity[i] > max_intensity:
        max_index = i
        max_intensity = intensity[i]
        max_position = position[i]

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

# def double_slit_model(x_val, I0, k):
#     theta = np.arcsin((x_val - position[max_index]) * k)
    
#     phi = np.pi * w * np.sin(theta) / wavelength
#     phi = np.where(phi == 0, 1e-9, phi)    
#     beta = np.pi * d * np.sin(theta) / wavelength

#     single_slit = (np.sinc(phi / np.pi)) ** 2
#     double_slit = np.cos(beta) ** 2

#     return I0 * single_slit * double_slit

def double_slit_model(x_val, a, b, c, d):
    #theta = np.arcsin(wavelength / a)
    sin_theta = (x_val - max_position)/(np.sqrt(wheel_sensor_dist**2 + (x_val - max_position)**2))
    
    phi = np.pi * a *sin_theta / (wavelength)
    beta = np.pi * b * sin_theta / wavelength

    single_slit = (np.sinc(phi/np.pi)) ** 2
    double_slit = (np.cos(beta)) ** 2
    offset_slit = (np.sinc(phi/np.pi)) ** 2

    return d*max_intensity * (single_slit * double_slit) - c* offset_slit

# def id_model(x_val, a, b, c):
#     return a*np.sin(b*(x_val-position[max_index])/(c*(x_val-position[max_index])))

# popt, pcov = curve_fit(id_model, position, intensity, maxfev = 100000)

# popt, pcov = curve_fit(temp_model, position, intensity, p0=(400.0, 400.0), maxfev = 100000)
#popt, pcov = curve_fit(temp_model, position, intensity, p0=(0.0004, 1.0), maxfev = 100000)

popt, pcov = curve_fit(double_slit_model, position, intensity, p0=(a, d, max_intensity, 0.5))


print("Number of fringes: ", popt[1]/popt[0])



print("a value is: ", wavelength/(max_position*popt[1]))

plt.figure(figsize=(15, 8))
plt.errorbar(position, intensity, fmt='o', color = "red", label = "Measured Data", markersize=1)
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


double_slit_Prediction = double_slit_model(position, *popt)

residual = np.abs(intensity - double_slit_Prediction)
residual_pos = position[residual>=0.0]
residual = residual[residual>=0.0]

#Version 2, finding the peaks
peaks, properties = scipy.signal.find_peaks(residual)
residual = residual[peaks]
residual_pos = residual_pos[peaks]



def residual_sin_model(x_val, I0, k, is_Single = False):
    theta = np.arcsin((x_val - position[max_index]) * k)
    
    phi = np.pi * a * np.sin(theta) / wavelength
    phi = np.where(phi == 0, 1e-9, phi)    
    beta = np.pi * d * np.sin(theta) / wavelength

    single_slit = (np.sinc(phi / np.pi)) ** 2
    double_slit = np.cos(beta) ** 2
    if (is_Single):
        return I0 * single_slit
    else:
        return I0 * single_slit * double_slit

res_popt, res_pcov = curve_fit(residual_sin_model, residual_pos, residual)

residual_model_data = residual_sin_model(residual_pos, *res_popt[:-1], True)

plt.figure(figsize=(15, 8))
plt.plot(residual_pos, residual_model_data, color = "green")

plt.plot(position, np.zeros(position.size), color = "blue")
plt.plot(residual_pos, residual, color = "red")



plt.show()

# plt.plot(position, id_model(position, popt[0], popt[1], popt[2], popt[3]), color = "red")
#plt.plot(position, temp_model(position, popt[0], popt[1], popt[2], popt[3], popt[4]), color = "red")

# plt.plot(position, temp_model(position, popt[0], popt[1]), color = "red")
# plt.plot(position, max_intensity*(np.sin(400*(position-position[max_index]))/(400*(position-position[max_index])))**2)

new_measured_data = intensity

for r in range(len(position)):
    if position[r] >= residual_pos[0] and position[r] <= residual_pos[-1]:
        new_measured_data[r]-= residual_sin_model(position[r], *res_popt[:-1], True)

# plt.figure(figsize=(15, 8))
# # plt.errorbar(position, new_measured_data, fmt='o', color = "red", label = "Measured Data")
# plt.errorbar(position, intensity, fmt='o', color = "red", label = "Measured Data")

# plt.plot(position, double_slit_model(position, *popt)-residual_sin_model(position, *res_popt[:-1], True), color = "blue", label = "Fitted Data")

# # plt.plot(position, id_model(position, popt[0], popt[1], popt[2]), color = "red")
# plt.xlabel("Position (m)")
# plt.ylabel("Intensity (V)")
# plt.legend()

# plt.show()





max_residual_intensity = 0.0
max_residual_pos = 0.0
max_residual_index = 0
for a in range(len(residual_model_data)):
    if residual_model_data[a] > max_residual_intensity:
        max_residual_intensity = residual_model_data[a]
        max_residual_pos = residual_pos[a]
        max_residual_index = a

first_min_residual_intensity = 100.0
first_min_residual_pos = 0.0

for j in range(max_residual_index, max_residual_index+30):
    if residual_model_data[j] < first_min_residual_intensity:
        first_min_residual_intensity = residual_model_data[j]
        first_min_residual_pos = residual_pos[j]


central_to_min_dist = np.abs(max_residual_pos - first_min_residual_pos)

residual_slit_width = wavelength / (central_to_min_dist/(np.sqrt(wheel_sensor_dist**2 + central_to_min_dist**2)))

print("Slit width is ", residual_slit_width)