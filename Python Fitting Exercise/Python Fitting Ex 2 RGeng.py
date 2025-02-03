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
original_year, original_month, original_dec_date, original_mean, original_deseason, original_ndays, original_std, original_unc = np.loadtxt("co2_mm_mlo.csv", delimiter = ',', skiprows=41, unpack=True)

year = np.array([])
month = np.array([])
dec_date = np.array([])
mean = np.array([])
deseason = np.array([])
ndays = np.array([])
std = np.array([])
unc = np.array([])


#Removes the data entries where the uncertainty is exactly 0 or negative.
for u in range(len(original_unc)):
    if original_unc[u]>0.0:
        year= np.append(year, original_year[u])
        month = np.append(month, original_month[u])
        dec_date = np.append(dec_date, original_dec_date[u])
        mean = np.append(mean, original_mean[u])
        deseason = np.append(deseason, original_deseason[u])
        ndays = np.append(ndays, original_ndays[u])
        std = np.append(std, original_std[u])
        unc = np.append(unc, original_unc[u])



####################################################################################
#Defining the periodic model to be used for our curve fitting exercise

#Quadratic from EX1, since it fitted pretty nicely
def quadratic_model(x_val, A, B, C):
    return A*(x_val-1960)**4+B*(x_val-1960)+C

#Same format as the lecture slide's but we are also implementing the quadratic model into it.
#Periodic component is of the format Ct*sin(2pi*D*t + E)
def periodic_model(x_val, A, B, C, D, E, phi):
    return quadratic_model(x_val, A, B, C) + D*np.sin(2*E*np.pi*(x_val-1960)-phi)


#Curve fitting for the models' parameters
# #quad_popt, quad_pcov = curve_fit(quadratic_model, dec_date, mean, p0=(0.01348, 0.77451), sigma = unc, absolute_sigma=True)
quad_popt, quad_pcov = curve_fit(quadratic_model, dec_date, mean, sigma = unc, absolute_sigma=True)

period_popt, period_pcov = curve_fit(periodic_model, dec_date, mean, p0=(quad_popt[0], quad_popt[1], quad_popt[2], 1, 1, 1), sigma = unc, absolute_sigma=True)
###############################################################################################
#Residual Calculation

periodic_model_data = dec_date*0
periodic_residual = dec_date*0

for p in range(len(dec_date)):
    periodic_model_data[p] = periodic_model(dec_date[p], period_popt[0], period_popt[1], period_popt[2], period_popt[3], period_popt[4], period_popt[5])
    periodic_residual[p] = mean[p] - periodic_model_data[p]

#################################################################################################
per_chi2=np.sum( (mean - periodic_model_data)**2 / unc**2 )
per_reduced_chi2 = per_chi2/(mean.size - len(period_popt))

print("Periodic Chi squared ", per_chi2)
print("Periodic Chi reduced squared ", per_reduced_chi2)        

#################################################################################################
#Calculate the uncertainty of the entire function via propagation.

#Diagonalizing the covariance variable
period_pcov = np.diag(period_pcov)
period_parameter_unc = np.sqrt(period_pcov)

def period_model_unc(year):
    x=year-1960
    return np.sqrt((period_parameter_unc[0]*(x**4))**2+(x*period_parameter_unc[1])**2 + period_parameter_unc[2]**2
                   + (period_parameter_unc[3]*np.sin(2*period_popt[4]*np.pi*x-period_popt[5]))**2
                   + (2*np.pi*x*period_popt[3]*period_parameter_unc[4]*np.cos(2*period_popt[4]*np.pi*x-period_popt[5]))**2
                   + (-1*period_popt[3]*period_parameter_unc[5]*np.cos(2*period_popt[4]*np.pi*x-period_popt[5]))**2)


#################################################################################################
#Plotting the dataset and the fitted model

plt.figure(figsize = (8, 16))

#First subplot corresponding to the original data set and the periodic model's fitting.
plt.subplot(2, 1, 1)
plt.errorbar(dec_date, mean, yerr=unc, fmt='o', capsize=0, ecolor = "red", label = "Data", marker = ".", color = "red", markersize = 2)
plt.plot(dec_date, periodic_model(dec_date, period_popt[0], period_popt[1], period_popt[2], period_popt[3], period_popt[4], period_popt[5]), label = "Periodic Model Curve Fit", color="blue", linewidth =0.75)
plt.xlabel("Year")
plt.ylabel(r'$CO_2$ Level (in unit of ppm)')
plt.xticks(np.arange(1973, 2024, step = 5))
plt.legend()
plt.title("Mean CO$_2$ level with periodic model curve fitting")

#Second subplot for the residuals, with a newly defined variable zero_residual_line as the line where the residual is 0.
zero_residual_line = np.zeros(len(dec_date))
plt.subplot(2, 1, 2)
plt.plot(dec_date, zero_residual_line, label="Zero residual line")
plt.errorbar(dec_date, periodic_residual, yerr=unc, fmt='o', capsize=0, ecolor = "red", label = "Residual of the periodic model versus actual data", marker = ".", markersize = 10)
plt.xlabel("Year")
plt.ylabel(r'Error of $CO_2$ Level in the periodic model (in unit of ppm)')
plt.xticks(np.arange(1973, 2024, step = 5))
plt.legend()
plt.title("Residuals from the periodic model")
plt.show()


#################################################################################################
#Report questions


























