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
    return A*(x_val-2010)+B

#Power fitting models:

#Quadratic fitting model using the same style as f(x)=ax^2+bx+c
# def quadratic_model(x_val, A, B, C):
#     return A*(x_val-2005)**2+B*(x_val-2005)+C
def quadratic_model(x_val, A, B):
    return A*(x_val-1959)**2+B*(x_val-1959)+mean[0]


#Note for both power_model and exponential_model we encountered the runtime error:
#RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800.
#To fix this, we introduced the term -1959 to x_val to fit the model aka x_val[0]=1959-1959 = 0 for the computer to be able to handle the simulation.

#Power fitting model using the same style as f(x)=ax^b+c
# def power_model(x_val, A, B, C):
#     return A*(x_val-1959)**B+C
def power_model(x_val, A, B):
    return A*(x_val-1959)**B+mean[0]
    
#Exponential fitting model using the same style as f(x)=ae^(bx)+c
# def exponential_model(x_val, A, B, C):
#     return A*np.exp(B*(x_val-2005))+C


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
# exp_popt, exp_pcov = curve_fit(exponential_model, year, mean, p0=(1, 1e-6, 200))


#Model values and their residuals
#Data returned by the linear model
#Initialized array also need to match the size of that of the year to avoid length mismatch
lin_model_data = lin_year*0
lin_residual = lin_year*0
for l in range(len(lin_year)):
    lin_model_data[l]=linear_model(lin_year[l], lin_popt[0], lin_popt[1])
    lin_residual[l]=lin_mean[l]-lin_model_data[l]

#Initializing the arrays of the expected values from the models after curve fit
#Initialized arrays also need to match the size of that of the year to avoid length mismatch
quad_model_data = year*0
quad_residual = year*0

pow_model_data = year*0
pow_residual = year*0

# exp_model_data = year*0
# exp_residual = year*0

#Calculating the expected values and finding their respective residuals
for i in range (len(year)):
    #First the expected values
    # quad_model_data[i]=quadratic_model(year[i], quad_popt[0], quad_popt[1], quad_popt[2])
    quad_model_data[i]=quadratic_model(year[i], quad_popt[0], quad_popt[1])
    pow_model_data[i]=power_model(year[i], pow_popt[0], pow_popt[1])
    # exp_model_data[i]=exponential_model(year[i], exp_popt[0], exp_popt[1], exp_popt[2])
    #Then calculate the residuals
    quad_residual[i] = mean[i]-quad_model_data[i]
    pow_residual[i] = mean[i]-pow_model_data[i]
    # exp_residual[i] = mean[i]-exp_model_data[i]


#Calculating the uncertainties in our models

#Linear Model
lin_pcov=np.diag(lin_pcov)
lin_unc_total=lin_model_data*0 #initializing empty total uncertainty array matching the size of model data
#Since f(x)=ax+b, where x is the year. At each iteration, we can treat the year with no uncertainty (because we are not saying i.e. 1960 +- 1 year for the corresponding c02 data)
for w in range(len(lin_unc_total)):
    #Apply point 2 in the uncertainty lecture slide to get the first part (ax)
    #Since unc(year) is 0, then u(f)=f*sqrt(((u(a)/a)**2)+0)=f*u(a)/a
    temp_unc_a=np.sqrt(lin_pcov[0])
    temp_unc_1=lin_year[w]*lin_popt[0]*temp_unc_a/lin_popt[0]
    #Apply point 1 in the lecture slide to calculate the model's uncertainty counting (+b)
    #Since uncertainty of b is sqrt(lin_pcov[1,1]),if we square it for the uncertainty, we simply get lin_pcov[1,1]
    
    # temp_unc_2=np.sqrt(temp_unc_1**2+lin_pcov[1]) #This is our model's uncertainty
    temp_unc_2=np.sqrt(temp_unc_1**2+np.sqrt(lin_pcov[1]**2)) #This is our model's uncertainty
    
    #Now calculate the unc of the residual by applying point 1 of the slide again
    temp_unc_total_per_year = np.sqrt(lin_unc[w]**2+temp_unc_2**2)
    
    #Chi squared values
    #temp_chi2 = np.sum( (lin_year[l] - linear_model(lin_year[l], lin_popt[0], lin_popt[1]))**2 / temp_unc_total_per_year**2 )
    temp_chi2 = (lin_year[l] - linear_model(lin_year[w], lin_popt[0], lin_popt[1]))**2 / temp_unc_total_per_year**2
    temp_red_chi2 = temp_chi2/(lin_year.size - 3)
    lin_unc_total[w]=temp_red_chi2


#Quadratic Model


#Power Model



#Plotting the datas and models and the residues
#Plotting the linear model, the respective original data set, and the residuals
plt.figure(figsize = (8, 16))

#First subplot corresponding to the original data set and the linear model's fitting.
plt.subplot(2, 1, 1)
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
plt.plot(lin_year, lin_residual,'o', label = "Residual of the linear model versus actual data", marker = ".", color = "red")
plt.xlabel("Year")
plt.ylabel(r'$Error\:of\:CO_2\:Level\:in\:the\:linear\:model\:(in\:unit\:of\:ppm)$')
plt.xticks(np.arange(2004, 2023, step = 2))
plt.legend()
plt.title("Residuals from the linear model")
plt.show()

linear_chi2=np.sum( (lin_mean - lin_model_data)**2 / lin_unc**2 )
linear_reduced_chi2 = linear_chi2/(lin_mean.size - 2)

print("Linear Chi squared ", linear_chi2)
print("Linear Chi reduced squared ", linear_reduced_chi2)

###################################################################################################################################################
#Plotting for the quadratic model
plt.figure(figsize = (8, 16))

#First subplot corresponding to the original data set and the linear model's fitting.
plt.subplot(2, 1, 1)
# plt.plot(year, quadratic_model(year, quad_popt[0], quad_popt[1], quad_popt[2]), label = "Quadratic Model Curve Fit", color="blue")
plt.plot(year, quadratic_model(year, quad_popt[0], quad_popt[1]), label = "Quadratic Model Curve Fit", color="blue")
plt.errorbar(year, mean, yerr=unc, fmt='o', capsize=0, ecolor = "black", label = "Data", marker = ".", markersize = 10)
plt.xlabel("Year")
plt.ylabel(r'$CO_2\:Level\:(in\:unit\:of\:ppm)$')
plt.xticks(np.arange(1959, 2023, step = 5))
plt.legend()
plt.title("Mean CO$_2$ level with Quadratic model curve fitting")

#Second subplot for the residuals, with a newly defined variable lin_zero_err as the line where the residual is 0.
zero_residual_line = np.zeros(len(year))
plt.subplot(2, 1, 2)
plt.plot(year, zero_residual_line)
plt.plot(year, quad_residual,'o', label = "Residual of the quadratic model versus actual data", marker = ".", color = "red")
plt.xlabel("Year")
plt.ylabel(r'$Error\:of\:CO_2\:Level\:in\:the\:quadratic\:model\:(in\:unit\:of\:ppm)$')
plt.xticks(np.arange(1959, 2023, step = 5))
plt.legend()
plt.title("Residuals from the quadratic model")
plt.show()

quad_chi2=np.sum( (mean - quad_model_data)**2 / unc**2 )
quad_reduced_chi2 = quad_chi2/(mean.size - 2)

print("Quadratic Chi squared ", quad_chi2)
print("Quadratic Chi reduced squared ", quad_reduced_chi2)


###################################################################################################################################################
#Plotting for the power model
plt.figure(figsize = (8, 16))

#First subplot corresponding to the original data set and the linear model's fitting.
plt.subplot(2, 1, 1)
plt.plot(year, power_model(year, pow_popt[0], pow_popt[1]), label = "Power Model Curve Fit", color="blue")
plt.errorbar(year, mean, yerr=unc, fmt='o', capsize=0, ecolor = "black", label = "Data", marker = ".", markersize = 10)
plt.xlabel("Year")
plt.ylabel(r'$CO_2\:Level\:(in\:unit\:of\:ppm)$')
plt.xticks(np.arange(1959, 2023, step = 5))
plt.legend()
plt.title("Mean CO$_2$ level with Power model curve fitting")

#Second subplot for the residuals, with a newly defined variable lin_zero_err as the line where the residual is 0.
zero_residual_line = np.zeros(len(year))
plt.subplot(2, 1, 2)
plt.plot(year, zero_residual_line)
plt.plot(year, pow_residual,'o', label = "Residual of the power model versus actual data", marker = ".", color = "red")
plt.xlabel("Year")
plt.ylabel(r'$Error\:of\:CO_2\:Level\:in\:the\:power\:model\:(in\:unit\:of\:ppm)$')
plt.xticks(np.arange(1959, 2023, step = 5))
plt.legend()
plt.title("Residuals from the power model")
plt.show()

power_chi2=np.sum( (mean - pow_model_data)**2 / unc**2 )
power_reduced_chi2 = power_chi2/(mean.size - 2)

print("Power Chi squared ", power_chi2)
print("Power Chi reduced squared ", power_reduced_chi2)



####################################################################################################################################################

##########################PLEASE IGNORE THE FOLLOWING###############################################################################################

#Scratches and things might be usedful for later project(s)

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