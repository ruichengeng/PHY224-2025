# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:14:21 2025

Created by Rui (Richard) Chen Geng Li
Student number: 1003048454

Code created for PHY224 Fitting Exercise 1
Due Sunday January 20th, 2025
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Imports and reads the data from file named co2_annmean_mlo for the annual average of daily measurement of atmospheric co2 from 1960 to 2023 from the top of Mauna Loa in Hawaii.
#Below: We skips 44 rows so that the first data entry is from 1959 aka the first row containing data
#We also separated the data category based on separation character ',' 
#Then we unpacked these into different arrays of data according to the type (i.e. years separated from co2 level separated from the uncertainty)
#year is an array with each element as the year in unit of year
#mean is an array with each element as the mean of the co2 level in a given year in unit of ppm
#unc is an array with each element as the uncertainty of the mean co2 level in a given year, also in unit of ppm
year, mean, unc = np.loadtxt("co2_annmean_mlo.csv", delimiter = ',', skiprows=44, unpack=True)


#Here we define the different types of fitting models, the independent variable is x_val and coefficients A, B, and C where they are applicable
#We will use curve_fit function later to estimate/fit the values of the above-mentioned coefficients 

#Linear fitting model using the same style as the formula f(x)=ax+b
def linear_model(x_val, A, B):
    return A*x_val+B

#Power fitting models:

#Quadratic fitting model using the same style as f(x)=ax^2+bx+c
def quadratic_model(x_val, A, B, C):
    return A*x_val**2+B*x_val+C


#Note for both power_model and exponential_model we encountered the runtime error:
#RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800.
#To fix this, we introduced the term -1959 to x_val to fit the model aka x_val[0]=1959-1959 = 0 for the computer to be able to handle the simulation.

#Power fitting model using the same style as f(x)=ax^b+c
def power_model(x_val, A, B, C):
    return A*(x_val-1959)**B+C
    
#Exponential fitting model using the same style as f(x)=ae^(bx)+c
def exponential_model(x_val, A, B, C):
    return A*np.exp(B*(x_val-1959))+C


#Using the curve_fit function for each of the above models

#Define new variables for the linear fitting model so that we can only account for the last 20 years of data.
lin_year = year[-20:]
lin_mean = mean[-20:]
lin_unc = unc[-20:]
#Linear model curve fitting using the new variables
lin_popt, lin_pcov = curve_fit(linear_model, lin_year, lin_mean)

#Quadratic model curve fitting
quad_popt, quad_pcov = curve_fit(quadratic_model, year, mean)

#Power model curve fitting
pow_popt, pow_pcov = curve_fit(power_model, year, mean)

#Exponential model curve fitting
#Here we are providing the initial value estimating value of A = 1, B = almost 0, C = 200 (this is still below the smallest co2 level in the given data set)
exp_popt, exp_pcov = curve_fit(exponential_model, year, mean, p0=(1, 1e-6, 200))


#Model values and their residuals
#Data returned by the linear model
lin_model_data = lin_year*0
lin_residual = lin_year*0
for l in range(len(lin_year)):
    lin_model_data[l]=linear_model(lin_year[l], lin_popt[0], lin_popt[1])
    lin_residual[l]=lin_mean[l]-lin_model_data[l]

#Initializing the arrays of the expected values from the models after curve fit
quad_model_data = year*0
quad_residual = year*0

pow_model_data = year*0
pow_residual = year*0

exp_model_data = year*0
exp_residual = year*0

#Calculating the expected values and finding their respective residuals
for i in range (len(year)):
    #First the expected values
    quad_model_data[i]=quadratic_model(year[i], quad_popt[0], quad_popt[1], quad_popt[2])
    pow_model_data[i]=power_model(year[i], pow_popt[0], pow_popt[1], pow_popt[2])
    exp_model_data[i]=exponential_model(year[i], exp_popt[0], exp_popt[1], exp_popt[2])
    #Then calculate the residuals
    quad_residual[i] = mean[i]-quad_model_data[i]
    pow_residual[i] = mean[i]-pow_model_data[i]
    exp_residual[i] = mean[i]-exp_model_data[i]


#Plotting the datas and models and the residues
#Plotting the linear model, the respective original data set, and the residuals
plt.figure(figsize = (8, 16))

#First subplot corresponding to the original data set and the linear model's fitting.
plt.subplot(2, 1, 1)
#plt.plot(lin_year, lin_mean, label = "Data (last 20 years)")
plt.plot(lin_year, linear_model(lin_year, lin_popt[0], lin_popt[1]), label = "Linear Model Curve Fit", color="blue")
plt.errorbar(lin_year, lin_mean, yerr=lin_unc, fmt='o', capsize=0, ecolor = "black", label = "Data (last 20 years)", marker = ".", markersize = 10)
plt.xlabel("Year")
plt.ylabel(r'$CO_2\:Level\:(in\:unit\:of\:ppm)$')
plt.xticks(np.arange(2004, 2023, step = 2))
plt.legend()
plt.title("Mean CO$_2$ level with linear model curve fitting")

#Second subplot for the residuals, with a newly defined variable lin_zero_err as the line where the residual is 0.
lin_zero_err = np.zeros(len(lin_year))
plt.subplot(2, 1, 2)
plt.plot(lin_year, lin_zero_err)






#missing residual error bars





tempError = np.zeros(len(lin_year))
plt.errorbar(lin_year, lin_residual, yerr=tempError, fmt='o', capsize=0, color = "red", label = "Data (last 20 years)", marker = ".", markersize = 10)
#plt.plot(lin_year, lin_residual, label = "Residual of the linear model versus actual data", marker = ".", color = "red")

plt.xlabel("Year")
plt.ylabel(r'$Error\:of\:CO_2\:Level\:in\:the\:linear\:model\:(in\:unit\:of\:ppm)$')
plt.xticks(np.arange(2004, 2023, step = 2))
plt.legend()
plt.title("Residuals from the linear model")
plt.show()




# print ("A value:", popt2[0])
# print ("B value:", popt2[1])
# print ("C value:", popt2[2])

# print (power_model(2060, popt[0], popt[1], popt[2]))
# print (exponential_model(2060, popt2[0], popt2[1], popt2[2]))

# #perr = np.sqrt(np.diag(pcov))

# plt.plot(year, quadratic_model(year, quad_popt[0], quad_popt[1], quad_popt[2]), label = "Quadratic Curve Fit")
# plt.xlabel("x values")
# plt.ylabel("y values")
# plt.legend()
# plt.show()

# #print ("A value:", popt[0])
# #print ("B value:", popt[1])
# #print ("C value:", popt[2])

# #print (quad_quiz(2060, popt[0], popt[1], popt[2]))