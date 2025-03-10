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
#Configuration 1
position_config1, intensity_config1 = np.loadtxt("double_slit_data_100x.txt", 
                                                  delimiter = '\t', 
                                                  skiprows=2, unpack=True)

position_config1 = position_config1[intensity_config1>=0.0]
intensity_config1 = intensity_config1[intensity_config1>=0.0]
pos1_unc = np.ones(position_config1.size)*0.00006
intensity1_unc = np.ones(position_config1.size)*0.005
a1 = 0.04e-3  # Slit width: 0.04 mm in meters
d1 = 0.25e-3  # Slit separation: 0.25 mm in meters

#Configuration 2
position_config2, intensity_config2 = np.loadtxt("double_slit_a0.04_d0.5_100x.txt", 
                                                  delimiter = '\t', 
                                                  skiprows=2, unpack=True)

position_config2 = position_config2[intensity_config2>=0.0]
intensity_config2 = intensity_config2[intensity_config2>=0.0]
pos2_unc = np.ones(position_config2.size)*0.00006
intensity2_unc = np.ones(position_config2.size)*0.005
a2 = 0.04e-3  # Slit width: 0.04 mm in meters
d2 = 0.5e-3  # Slit separation: 0.25 mm in meters

#Configuration 3
position_config3, intensity_config3 = np.loadtxt("double_slit_a0.08_d0.25_10x.txt", 
                                                  delimiter = '\t', 
                                                  skiprows=2, unpack=True)

position_config3 = position_config3[intensity_config3>=0.0]
intensity_config3 = intensity_config3[intensity_config3>=0.0]
pos3_unc = np.ones(position_config3.size)*0.00011
intensity3_unc = np.ones(position_config3.size)*0.01
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
    
    #Refining the dataset, so that the curve fit will be performed to only the "middle" region.
    uncleaned_position_size = position.size
    # count = int(position.size/3)
    # position = position[count:-count]
    # intensity = intensity[count:-count]
    # pos_unc = pos_unc[count:-count]
    # intensity_unc = intensity_unc[count:-count]
    
    for i in range(len(intensity)):
        if intensity[i] > max_intensity:
            max_index = i
            max_intensity = intensity[i]
            max_position = position[i]
    
    #Prediction Models
    def double_slit_model(x_val, a, b, c, d):
        #theta = np.arcsin(wavelength / a)
        sin_theta = (x_val - max_position)/(np.sqrt(wheel_sensor_dist**2 + (x_val - max_position)**2))

        phi = np.pi * a *sin_theta / (wavelength)
        beta = np.pi * b * sin_theta / wavelength

        single_slit = (np.sinc(phi/np.pi)) ** 2
        double_slit = (np.cos(beta)) ** 2
        offset_slit = (np.sinc(phi/np.pi)) ** 2

        return d*max_intensity * (single_slit * double_slit) - c* offset_slit
    
    
    
    popt, pcov = curve_fit(double_slit_model, position, intensity, p0=(a, d, max_intensity, 0.5), sigma=intensity_unc, absolute_sigma = True)
    
    
    print("Number of predicted fringes: ", popt[1]/popt[0])
    print("a value (predicted slit width) is: ", popt[0], " ± ", np.sqrt(pcov[0][0]))
    print("b value (predicted slit separation) is: ", popt[1], " ± ", np.sqrt(pcov[1][1]))
    print("c value (predicted offset scalar value) is: ", popt[2], " ± ", np.sqrt(pcov[2][2]))
    print("d value (predicted maximum intensity scalar) is: ", popt[3], " ± ", np.sqrt(pcov[3][3]))
    
    plt.figure(figsize=(15, 8))
    plt.errorbar(position, intensity, xerr=pos_unc, yerr=intensity_unc, fmt='o', color = "red", label = "Measured Data", markersize=1)
    plt.plot(position, double_slit_model(position, *popt), color = "blue", label = "Model Data")
    
    plt.xlabel("Position (m)")
    plt.ylabel("Intensity (V)")
    config_text = 'a=' + str(a*1000.0) + 'mm, d=' + str(d*1000.0) + 'mm'
    plt.title("Double Slit Measurement Data versus Prediction Model Data. Configuration: "+ config_text)
    plt.legend()
    
    plt.show()
    
    #Zoom
    plt.figure(figsize=(15, 8))
    plt.errorbar(position, intensity, xerr=pos_unc, yerr=intensity_unc, fmt='o', color = "red", label = "Measured Data", markersize=1)
    plt.plot(position, double_slit_model(position, *popt), color = "blue", label = "Model Data")
    
    plt.xlabel("Position (m)")
    plt.xticks(np.arange(position[int(position.size/3):2*int(position.size/3)], step = 0.01))
    plt.ylabel("Intensity (V)")
    config_text = 'a=' + str(a*1000.0) + 'mm, d=' + str(d*1000.0) + 'mm'
    plt.title("Double Slit Measurement Data versus Prediction Model Data. Configuration: "+ config_text)
    plt.legend()
    
    plt.show()
    
    
    double_slit_Prediction = double_slit_model(position, *popt)
    
    residual = np.abs(intensity - double_slit_Prediction)    
    
    #Reduced Chi square
    chi2=np.sum( (intensity - double_slit_Prediction)**2 / (np.abs(pos_unc/position))**2 )
    reduced_chi2 = chi2/(uncleaned_position_size - popt.size)
    print("The reduced chi square value is: ", reduced_chi2)
    
    # residual_pos_unc = pos_unc[residual>=0.0]
    # residual_int_unc = intensity_unc[residual>=0.0]
    # residual_pos = position[residual>=0.0]
    # residual = residual[residual>=0.0]
    
    #Version 2, finding the peaks
    peaks, properties = scipy.signal.find_peaks(residual)
    residual_pos = position[peaks]
    residual_pos_unc = pos_unc[peaks]
    residual_int_unc = intensity_unc[peaks]
    residual = residual[peaks]
    
    plt.figure(figsize=(15, 8))    
    plt.plot(position, np.zeros(position.size), color = "blue", label = "Zero Residual Line")
    plt.errorbar(residual_pos, residual, xerr=residual_pos_unc, yerr=residual_int_unc, fmt ='o', color = "red", label = "Model Residual")
    plt.title("Residual Plot of Prediction Data versus Measured Data")
    plt.xlabel("Position (m)")
    plt.ylabel("Error in Intensity (V)")
    plt.legend()
    
    plt.show()
    
    ###############################
    
    intensity_std = np.sqrt(np.sum((position-np.mean(position))**2)/position.size)
    #intensity_std = np.sqrt(np.sum((intensity-np.mean(intensity))**2)/intensity.size)

    # defining a function to calculate chi^2_red
    def chi2r(measured, predicted, errors, num_of_parameters):
        return np.sum((measured-predicted)**2/errors**2) / \
               (measured.size - num_of_parameters)
    
    print('Reduced Chi Squared of Nonlinear Curve Fit is:',
          chi2r(intensity, double_slit_model(position, *popt), intensity_std, popt.size))
    
    
    
Run_Slits_Models(position_config1, intensity_config1, pos1_unc, intensity1_unc, a1, d1)
Run_Slits_Models(position_config2, intensity_config2, pos2_unc, intensity2_unc, a2, d2)
Run_Slits_Models(position_config3, intensity_config3, pos3_unc, intensity3_unc, a3, d3)

