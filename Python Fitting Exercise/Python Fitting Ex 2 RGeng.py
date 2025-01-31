# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 04:31:59 2025

Created by Rui (Richard) Chen Geng Li
Student number: 1003048454

Code created for PHY224 Fitting Exercise 2
Prof. Sergio De La Barrera
Ta: Weigeng Peng
Due January 30th, 2025
"""

#Note:
#The structure of this code is ordered in a similar fassion as the bullet point listed in the IntroFitting.pdf file
#With rare exceptions such as that at the very end of this script, there is a distinguishable section of scratches for testing out ideas, unused models and possibly saving them for future uses. Please disregard that scratch section for this exercise.


#Necessary modules
import numpy as np #Useful helper functions that can convert our data into formats acceptable to the curve fitting components.
from scipy.optimize import curve_fit #The main component of this fitting exercise
import matplotlib.pyplot as plt #Handles the plotting aspect of this lab

#Imports and reads the data from file named co2_annmean_mlo for the annual average of daily measurement of atmospheric co2 from 1960 to 2023 from the top of Mauna Loa in Hawaii.
#Below: We skips 44 rows so that the first data entry is from 1959 aka the first row containing data
#We also separated the data category based on separation character ',' 
#Then we unpacked these into different arrays of data according to the type (i.e. years separated from co2 level separated from the uncertainty)
#year is an array with each element as the year in unit of year
#mean is an array with each element as the mean of the co2 level in a given year in unit of ppm
#unc is an array with each element as the uncertainty of the mean co2 level in a given year, also in unit of ppm
year, month, dec_date, mean, deseason, ndays, std, unc = np.loadtxt("co2_mm_mlo.csv", delimiter = ',', skiprows=41, unpack=True)



def p_model_1(x_val, A, B, C, D):
    return A+B*x_val+C*np.sin(2*np.pi*x_val-D)

def p_model_2(x_val, A, B, C, D, E, F):
    return A+B*x_val+C*np.sin(2*np.pi*x_val-D) + E*x_val**2



#Fixes the 0s in the uncertainties
for u in range(len(unc)):
    if unc[u]==0:
        unc[u]=1e-10
        




p0l = [2, 2, 0, 0.001, 0.5, 300]
popt, pcov = curve_fit(p_model_2, dec_date, mean, p0=p0l, sigma=unc, absolute_sigma=True)


plt.figure(figsize = (8, 16))
#plt.xticks(np.arange(1957, 2023, step = 1))
plt.plot(year, mean)
#plt.plot(dec_date, p_model_1(dec_date, popt[0], popt[1], popt[2], popt[3]))
plt.plot(dec_date, p_model_2(dec_date, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]))
































# import pandas as pd
# import numpy as np
# import scipy.optimize as opt
# import matplotlib.pyplot as plt

# # Load the dataset
# file_path = "co2_mm_mlo.csv"  # Update with the correct path if needed
# df = pd.read_csv(file_path, comment='#', skip_blank_lines=True)

# # Extract relevant data
# x = df["decimal date"].values
# y = df["average"].values

# # Remove missing values (-9.99 is used as a placeholder for missing data)
# mask = y > 0
# x = x[mask]
# y = y[mask]

# # Define a periodic model: sinusoidal function with quadratic trend
# def periodic_model(t, A, B, C, D, E, F):
#     return A * np.sin(B * t + C) + D * t**2 + E * t + F

# # Initial parameter guess: amplitude, frequency, phase, quadratic trend, linear trend, and offset
# guess = [2, 2 * np.pi / 1, 0, 0.001, 0.5, 300]  # Assume yearly seasonality

# # Perform curve fitting
# params, params_covariance = opt.curve_fit(periodic_model, x, y, p0=guess)

# # Generate fitted curve
# x_fit = np.linspace(min(x), max(x), 500)
# y_fit = periodic_model(x_fit, *params)

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.scatter(x, y, s=5, label="Data", color="blue", alpha=0.5)
# plt.plot(x_fit, y_fit, label="Fitted Curve", color="red")
# plt.xlabel("Year")
# plt.ylabel("CO₂ Concentration (ppm)")
# plt.title("CO₂ Concentration Over Time with Periodic Fit and Quadratic Trend")
# plt.legend()
# plt.show()

# # Print fitted parameters
# print("Fitted Parameters:", params)

