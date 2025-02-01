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

#Imports and reads the data from file named co2_mm_mlo.csv for the annual average of daily measurement of atmospheric co2 from 1960 to 2023 from the top of Mauna Loa in Hawaii.
#Same as we did for EX1, but now the data has many more entries and will have more column variables.
year, month, dec_date, mean, deseason, ndays, std, unc = np.loadtxt("co2_mm_mlo.csv", delimiter = ',', skiprows=41, unpack=True)


#Fixes the 0s in the uncertainties
for u in range(len(unc)):
    if unc[u]==0:
        unc[u]=.5
    elif unc[u]<0:
        unc[u]=.5


####################################################################################
#Defining the periodic model to be used for our curve fitting exercise

#Quadratic from EX1, since it fitted pretty nicely
def quadratic_model(x_val, A, B):
    return A*(x_val-1959)**2+B*(x_val-1959)+mean[0] 

#Same format as the lecture slide's but we are also implementing the quadratic model into it.
#Periodic component is of the format Ct*sin(2pi*D*t + E)
def periodic_model(x_val, A, B, C, D, E):
    return A*(x_val-1959)**2+B*(x_val-1959)+mean[0] + (C*x_val)*np.sin(2*D*np.pi*x_val-E)


#Curve fitting for the models' parameters
#quad_popt, quad_pcov = curve_fit(quadratic_model, dec_date, mean, p0=(0.01348, 0.77451), sigma = unc, absolute_sigma=True)
quad_popt, quad_pcov = curve_fit(quadratic_model, dec_date, mean, sigma = unc, absolute_sigma=True)

period_popt, period_pcov = curve_fit(periodic_model, dec_date, mean, p0=(quad_popt[0], quad_popt[1], 1, 1, 1), sigma = unc, absolute_sigma=True)


###############################################################################################
#Residual Calculation

periodic_model_data = dec_date*0
periodic_residual = dec_date*0

for p in range(len(dec_date)):
    periodic_model_data[p] = periodic_model(dec_date[p], period_popt[0], period_popt[1], period_popt[2], period_popt[3], period_popt[4])
    periodic_residual[p] = mean[p] - periodic_model_data[p]

#################################################################################################
per_chi2=np.sum( (mean - periodic_model_data)**2 / unc**2 )
per_reduced_chi2 = per_chi2/(mean.size - 2)

print("Periodic Chi squared ", per_chi2)
print("Periodic Chi reduced squared ", per_reduced_chi2)        


#################################################################################################
#Plotting the dataset and the fitted model

plt.figure(figsize = (8, 16))

#First subplot corresponding to the original data set and the periodic model's fitting.
plt.subplot(2, 1, 1)
plt.errorbar(dec_date, mean, yerr=unc, fmt='o', capsize=0, ecolor = "red", label = "Data", marker = ".", markersize = 1)
plt.plot(dec_date, periodic_model(dec_date, period_popt[0], period_popt[1], period_popt[2], period_popt[3], period_popt[4]), label = "Periodic Model Curve Fit", color="blue")
plt.xlabel("Year")
plt.ylabel(r'$CO_2$ Level (in unit of ppm)')
plt.xticks(np.arange(1958, 2024, step = 5))
plt.legend()
plt.title("Mean CO$_2$ level with periodic model curve fitting")

#Second subplot for the residuals, with a newly defined variable lin_zero_err as the line where the residual is 0.
zero_residual_line = np.zeros(len(dec_date))
plt.subplot(2, 1, 2)
plt.plot(dec_date, zero_residual_line, label="Zero residual line")
plt.errorbar(dec_date, periodic_residual, yerr=unc, fmt='o', capsize=0, ecolor = "red", label = "Residual of the periodic model versus actual data", marker = ".", markersize = 10)
plt.xlabel("Year")
plt.ylabel(r'Error of $CO_2$ Level in the periodic model (in unit of ppm)')
plt.xticks(np.arange(1959, 2023, step = 5))
plt.legend()
plt.title("Residuals from the periodic model")
plt.show()
































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

