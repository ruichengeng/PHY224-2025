# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:34:14 2025

Interference and Diffraction

Turki Almansoori
Rui (Richard) Chen Geng Li

Code created for PHY224 Free Choice Lab: Interference and Diffraction
Prof. Sergio De La Barrera
Due March 9th, 2025
"""

#Necessary modules
import numpy as np 
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 


#Begin by reading all of the data files corresponding to our different config setups
#Then assign the theoretical/manufacturer intended values of slit distance and separation

#NOTE: Uncertainty for the intensity was obtained via estimation of the data using cursor,
#since no manufacturer data regarding reading accuracy nor resolution.

#Configuration 1
position_config1, intensity_config1 = np.loadtxt("double_slit_data_100x.txt", 
                                                  delimiter = '\t', 
                                                  skiprows=2, unpack=True)

position_config1 = position_config1[intensity_config1>=0.0]
intensity_config1 = intensity_config1[intensity_config1>=0.0]
pos1_unc = np.ones(position_config1.size)*0.00006
intensity1_unc = np.ones(position_config1.size)*0.002
a1 = 0.04e-3  # Slit width: 0.04 mm in meters
d1 = 0.25e-3  # Slit separation: 0.25 mm in meters

#Configuration 2
position_config2, intensity_config2 = np.loadtxt("double_slit_a0.04_d0.5_100x.txt", 
                                                  delimiter = '\t', 
                                                  skiprows=2, unpack=True)

position_config2 = position_config2[intensity_config2>=0.0]
intensity_config2 = intensity_config2[intensity_config2>=0.0]
pos2_unc = np.ones(position_config2.size)*0.00006
intensity2_unc = np.ones(position_config2.size)*0.001
a2 = 0.04e-3  # Slit width: 0.04 mm in meters
d2 = 0.5e-3  # Slit separation: 0.25 mm in meters

#Configuration 3
position_config3, intensity_config3 = np.loadtxt("double_slit_a0.08_d0.25_10x.txt", 
                                                  delimiter = '\t', 
                                                  skiprows=2, unpack=True)

position_config3 = position_config3[intensity_config3>=0.0]
intensity_config3 = intensity_config3[intensity_config3>=0.0]
pos3_unc = np.ones(position_config3.size)*0.00011
intensity3_unc = np.ones(position_config3.size)*0.002
a3 = 0.08e-3  # Slit width: 0.04 mm in meters
d3 = 0.25e-3  # Slit separation: 0.25 mm in meters

#Some constants throughout the entire lab
wavelength = 650e-9    # 650 nm in meters
wheel_sensor_dist = 0.515 #distance between the light sensor and the slit wheel

#Main Function, to be repeated a for each of our dataset
def Run_Slits_Models(position, intensity, pos_unc, intensity_unc, a, d):
    
    #Calculation of the intensity's maximum, it's position and the corresponding index.
    max_index = 0   
    max_intensity = 0.0
    max_position = 0.0
    
    for i in range(len(intensity)):
        if intensity[i] > max_intensity:
            max_index = i
            max_intensity = intensity[i]
            max_position = position[i]
    
    print("Max Position = ", max_position, "m")
    print("Max Intensity = ", max_intensity, "V")
    
    #Prediction Model
    def double_slit_model(x_val, a, b, c, d):
        #theta = np.arcsin(wavelength / a)
        sin_theta = (x_val - max_position)/(np.sqrt(wheel_sensor_dist**2 + (x_val - max_position)**2))

        phi = np.pi * a *sin_theta / (wavelength)
        beta = np.pi * b * sin_theta / wavelength

        single_slit = (np.sinc(phi/np.pi)) ** 2
        double_slit = (np.cos(beta)) ** 2
        offset_slit = (np.sinc(phi/np.pi)) ** 2

        return d*max_intensity * (single_slit * double_slit) - c* offset_slit
    
    
    #Curve Fitting to the measurement data based on the model above
    popt, pcov = curve_fit(double_slit_model, position, intensity, p0=(a, d, max_intensity, 0.5), sigma=intensity_unc, absolute_sigma = True)
    
    #Printing out the fitted parameters and their uncertainties
    print("Number of predicted fringes: ", popt[1]/popt[0])
    print("a value (predicted slit width) is: ", popt[0], " ± ", np.sqrt(pcov[0][0]))
    print("b value (predicted slit separation) is: ", popt[1], " ± ", np.sqrt(pcov[1][1]))
    print("c value (predicted offset scalar value) is: ", popt[2], " ± ", np.sqrt(pcov[2][2]))
    print("d value (predicted maximum intensity scalar) is: ", popt[3], " ± ", np.sqrt(pcov[3][3]))
    
    #Plotting the first figure over the entire domain
    plt.figure(figsize=(12, 6))
    plt.errorbar(position, intensity, xerr=pos_unc, yerr=intensity_unc, fmt='o', color = "red", label = "Measured Data", markersize=1)
    plt.plot(position, double_slit_model(position, *popt), color = "blue", label = "Model Data")
    
    plt.xlabel("Position (m)")
    plt.ylabel("Intensity (V)")
    config_text = 'a=' + str(a*1000.0) + 'mm, d=' + str(d*1000.0) + 'mm'
    plt.title("Double Slit Measurement Data versus Prediction Model Data. Configuration: "+ config_text)
    plt.legend()
    
    plt.show()
    
    #Zooming in to the central region
    plt.figure(figsize=(12, 6))
    plt.errorbar(position, intensity, xerr=pos_unc, yerr=intensity_unc, fmt='o', color = "red", label = "Measured Data", markersize=1)
    plt.plot(position, double_slit_model(position, *popt), color = "blue", label = "Model Data")
    plt.xlim(position[max_index-195], position[max_index+195])
    plt.xlabel("Position (m)")
    plt.ylabel("Intensity (V)")
    config_text = 'a=' + str(a*1000.0) + 'mm, d=' + str(d*1000.0) + 'mm'
    plt.title("Double Slit Measurement Data versus Prediction Model Data. Configuration: "+ config_text)
    plt.legend()
    
    plt.show()
    
    #Calculation of the residuals
    #Prediction values
    double_slit_Prediction = double_slit_model(position, *popt)
    #Definition of residual = difference between measured and prediction data
    residual = intensity - double_slit_Prediction
    
    #Plotting the residuals
    plt.figure(figsize=(12, 6))    
    plt.plot(position, np.zeros(position.size), color = "blue", label = "Zero Residual Line")
    plt.errorbar(position, residual, xerr=pos_unc, yerr=intensity_unc, fmt ='o', color = "red", label = "Model Residual", markersize=1)
    plt.title("Residual Plot of Prediction Data versus Measured Data")
    plt.xlabel("Position (m)")
    plt.ylabel("Error in Intensity (V)")
    plt.legend()
    
    plt.show()
    
    #Calculation of the reduced chi square values
    #First obtain the standard deviation, so that we can then obtain the variance of position.
    position_std = np.sqrt(np.sum((position-np.mean(position))**2)/position.size)
    reduced_chi2 = np.sum((intensity-double_slit_Prediction)**2/position_std**2) /(position.size - popt.size)
    print ("The Reduced Chi Square Value is: ", reduced_chi2)
    
    

#Running the main function for each of our dataset.
Run_Slits_Models(position_config1, intensity_config1, pos1_unc, intensity1_unc, a1, d1)
Run_Slits_Models(position_config2, intensity_config2, pos2_unc, intensity2_unc, a2, d2)
Run_Slits_Models(position_config3, intensity_config3, pos3_unc, intensity3_unc, a3, d3)

