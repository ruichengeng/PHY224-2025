# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:14:21 2025

Created by Rui (Richard) Chen Geng Li
Student number: 1003048454

Code created for PHY224 Fitting Exercise 1
Prof. Sergio De La Barrera
Ta: Weigeng Peng
Due Sunday January 20th, 2025
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
year, mean, unc = np.loadtxt("co2_annmean_mlo.csv", delimiter = ',', skiprows=44, unpack=True)


######################################################################################################################################################################################################################################


#Here we define the different types of fitting models, the independent variable is x_val and coefficients A, and B where they are applicable
#We will use curve_fit function later to estimate/fit the values of the above-mentioned coefficients 

#Linear fitting model using the same style as the formula f(x)=ax+b
def linear_model(x_val, A, B):
    return A*(x_val-2010)+B #The offset of 2010 seemed to yield a relatively nice reduced chi-squared value after testing out a bunch of other ones.

#Power fitting models:

#Quadratic fitting model using the same style as f(x)=ax^2+bx+c
#Here we define C to be the initial value (mean of co2 level at 1959). Where at x=0. It is the year 1959, the very first data sample year we have available.
#The year 1959 is chosen because it was the first year the data is available to us, 
#but also most other ones would result in the runtime error that the curve_fit function was not able to find a nice fitting parameter for our model to come close to satisfying the given dataset.
def quadratic_model(x_val, A, B):
    return A*(x_val-1959)**2+B*(x_val-1959)+mean[0] 

#Similarly we have applied the offset of 1959 to the power model as well, to avoid the same runtime error.
#Power fitting model using the same style as f(x)=ax^b+c
#We have let the C component to be set as the initial co2 value in the year 1959.
def power_model(x_val, A, B):
    return A*(x_val-1959)**B+mean[0]
    

#############################################################################################################################################################################################################################################################


#Using the curve_fit function with parameters specified from the pdf for each of the above models

#Define new variables for the linear fitting model so that we can only account for the last 20 years of the given data.
lin_year = year[-20:]
lin_mean = mean[-20:]
lin_unc = unc[-20:]

#Linear model curve fitting using the new variables
lin_popt, lin_pcov = curve_fit(linear_model, lin_year, lin_mean, sigma = lin_unc, absolute_sigma=True)

#Quadratic model curve fitting
quad_popt, quad_pcov = curve_fit(quadratic_model, year, mean, sigma = unc, absolute_sigma=True)

#Power model curve fitting
#Initial guess is input by looking at the value of popt without specified initial value, and then putting a number approximately to that value, I just thought it would be fun.
pow_popt, pow_pcov = curve_fit(power_model, year, mean, p0=(0.3, 1.4), sigma = unc, absolute_sigma=True)


##############################################################################################################################################################################################################################################################


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

temp_pow_model_data = year*0
temp_pow_residual = year*0

#Calculating the expected values of both the quadratic and power models and finding their respective residuals
for i in range (len(year)):
    
    #First the expected values
    quad_model_data[i]=quadratic_model(year[i], quad_popt[0], quad_popt[1])
    pow_model_data[i]=power_model(year[i], pow_popt[0], pow_popt[1])
    
    #Then calculate the residuals
    quad_residual[i] = mean[i]-quad_model_data[i]
    pow_residual[i] = mean[i]-pow_model_data[i]


##############################################################################################################################################################################################################################################################


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




##############################################################################################################################################################################################################################################################


#Plotting the datas and models and the residues
#Plotting the linear model, the respective original data set, and the residuals
plt.figure(figsize = (8, 16))

#First subplot corresponding to the original data set and the linear model's fitting.
plt.subplot(2, 1, 1)
plt.plot(lin_year, linear_model(lin_year, lin_popt[0], lin_popt[1]), label = "Linear Model Curve Fit", color="blue")
plt.errorbar(lin_year, lin_mean, yerr=lin_unc, fmt='o', capsize=0, ecolor = "black", label = "Data (last 20 years)", marker = ".", markersize = 10)
plt.xlabel("Year")
plt.ylabel(r'$CO_2$ Level (in unit of ppm)')
plt.xticks(np.arange(2004, 2023, step = 2))
plt.legend()
plt.title("Mean CO$_2$ level with linear model curve fitting")

#Second subplot for the residuals, with a newly defined variable lin_zero_err as the line where the residual is 0.
lin_zero_err = np.zeros(len(lin_year))
plt.subplot(2, 1, 2)
plt.plot(lin_year, lin_zero_err)
plt.plot(lin_year, lin_residual,'o', label = "Residual of the linear model versus actual data", marker = ".", color = "red")
plt.xlabel("Year")
plt.ylabel(r'Error of $CO_2$ Level in the linear model (in unit of ppm)')
plt.xticks(np.arange(2004, 2023, step = 2))
plt.legend()
plt.title("Residuals from the linear model")
plt.show()


###################################################################################################################################################


#Plotting for the quadratic model
plt.figure(figsize = (8, 16))

#First subplot corresponding to the original data set and the linear model's fitting.
plt.subplot(2, 1, 1)
# plt.plot(year, quadratic_model(year, quad_popt[0], quad_popt[1], quad_popt[2]), label = "Quadratic Model Curve Fit", color="blue")
plt.plot(year, quadratic_model(year, quad_popt[0], quad_popt[1]), label = "Quadratic Model Curve Fit", color="blue")
plt.errorbar(year, mean, yerr=unc, fmt='o', capsize=0, ecolor = "black", label = "Data", marker = ".", markersize = 10)
plt.xlabel("Year")
plt.ylabel(r'$CO_2$ Level (in unit of ppm)')
plt.xticks(np.arange(1959, 2023, step = 5))
plt.legend()
plt.title("Mean CO$_2$ level with Quadratic model curve fitting")

#Second subplot for the residuals, with a newly defined variable lin_zero_err as the line where the residual is 0.
zero_residual_line = np.zeros(len(year))
plt.subplot(2, 1, 2)
plt.plot(year, zero_residual_line)
plt.plot(year, quad_residual,'o', label = "Residual of the quadratic model versus actual data", marker = ".", color = "red")
plt.xlabel("Year")
plt.ylabel(r'Error of $CO_2$ Level in the quadratic model (in unit of ppm)')
plt.xticks(np.arange(1959, 2023, step = 5))
plt.legend()
plt.title("Residuals from the quadratic model")
plt.show()


###################################################################################################################################################


#Plotting for the power model
plt.figure(figsize = (8, 16))

#First subplot corresponding to the original data set and the linear model's fitting.
plt.subplot(2, 1, 1)
plt.plot(year, power_model(year, pow_popt[0], pow_popt[1]), label = "Power Model Curve Fit", color="blue")
plt.errorbar(year, mean, yerr=unc, fmt='o', capsize=0, ecolor = "black", label = "Data", marker = ".", markersize = 10)
plt.xlabel("Year")
plt.ylabel(r'$CO_2$ Level (in unit of ppm)')
plt.xticks(np.arange(1959, 2023, step = 5))
plt.legend()
plt.title("Mean CO$_2$ level with Power model curve fitting")

#Second subplot for the residuals, with a newly defined variable lin_zero_err as the line where the residual is 0.
zero_residual_line = np.zeros(len(year))
plt.subplot(2, 1, 2)
plt.plot(year, zero_residual_line)
plt.plot(year, pow_residual,'o', label = "Residual of the power model versus actual data", marker = ".", color = "red")
plt.xlabel("Year")
plt.ylabel(r'Error of$ CO_2$ Level in the power model (in unit of ppm)')
plt.xticks(np.arange(1959, 2023, step = 5))
plt.legend()
plt.title("Residuals from the power model")
plt.show()


##############################################################################################################################################################################################################################################################


#Calculation of chi-squared values and reduced chi-squared values of the models, and printing them out.

linear_chi2=np.sum( (lin_mean - lin_model_data)**2 / lin_unc**2 )
linear_reduced_chi2 = linear_chi2/(lin_mean.size - 2)

print("Linear Chi squared ", linear_chi2)
print("Linear Chi reduced squared ", linear_reduced_chi2)

quad_chi2=np.sum( (mean - quad_model_data)**2 / unc**2 )
quad_reduced_chi2 = quad_chi2/(mean.size - 2)

print("Quadratic Chi squared ", quad_chi2)
print("Quadratic Chi reduced squared ", quad_reduced_chi2)

power_chi2=np.sum( (mean - pow_model_data)**2 / unc**2 )
power_reduced_chi2 = power_chi2/(mean.size - 2)

print("Power Chi squared ", power_chi2)
print("Power Chi reduced squared ", power_reduced_chi2)


############################ END OF THE LAB EXERCISE #####################################################################################################

####################################################################################################################################################

########################## PLEASE IGNORE THE FOLLOWING ###############################################################################################

#Scratches and things might be usedful for later project(s)

#Exponential fitting model using the same style as f(x)=ae^(bx)+c
def exponential_model(x_val, A, B, C):
    return A*np.exp(B*(x_val-1959))+C

#Exponential model curve fitting
#Here we are providing the initial value estimating value of A = 1, B = almost 0, C = 200 (this is still below the smallest co2 level in the given data set)
exp_popt, exp_pcov = curve_fit(exponential_model, year, mean,p0=(60, 1e-2, 250), sigma=unc, absolute_sigma=True)

exp_model_data = year*0
exp_residual = year*0

for i in range (len(year)):
    #First the expected values
    exp_model_data[i]=exponential_model(year[i], exp_popt[0], exp_popt[1], exp_popt[2])
    #Then calculate the residuals
    exp_residual[i] = mean[i]-exp_model_data[i]
    
exp_chi2=np.sum( (mean - exp_model_data)**2 / unc**2 )
power_reduced_chi2 = exp_chi2/(mean.size - 2)

print("Exp Chi squared ", power_chi2)
print("Exp Chi reduced squared ", power_reduced_chi2)

#Plotting for the exp model
plt.figure(figsize = (8, 16))

#First subplot corresponding to the original data set and the linear model's fitting.
plt.subplot(2, 1, 1)
plt.plot(year, exponential_model(year, exp_popt[0], exp_popt[1], exp_popt[2]), label = "Exp Model Curve Fit", color="blue")
plt.errorbar(year, mean, yerr=unc, fmt='o', capsize=0, ecolor = "black", label = "Data", marker = ".", markersize = 10)
plt.xlabel("Year")
plt.ylabel(r'$CO_2\:Level\:(in\:unit\:of\:ppm)$')
plt.xticks(np.arange(1959, 2023, step = 5))
plt.legend()
plt.title("Mean CO$_2$ level with exp model curve fitting")

#Second subplot for the residuals, with a newly defined variable lin_zero_err as the line where the residual is 0.
zero_residual_line = np.zeros(len(year))
plt.subplot(2, 1, 2)
plt.plot(year, zero_residual_line)
plt.plot(year, exp_residual,'o', label = "Residual of the exp model versus actual data", marker = ".", color = "red")
plt.xlabel("Year")
plt.ylabel(r'$Error\:of\:CO_2\:Level\:in\:the\:exp\:model\:(in\:unit\:of\:ppm)$')
plt.xticks(np.arange(1959, 2023, step = 5))
plt.legend()
plt.title("Residuals from the exp model")
plt.show()

#Temporary power model for the incorporation of both power, cubic and quadratic
# def temp_power_model(x_val, A, B, C, D, E):
#     return A*(x_val-1959)**B+C*(x_val-1959)**3+D*(x_val-1959)**2+E*(x_val-1959)+mean[0]

# temp_pow_popt, temp_pow_pcov=curve_fit(temp_power_model, year, mean, p0=(1e-3, 1e-3, 1e-3, 1e-3, 1e-3), sigma = unc, absolute_sigma=True)

# for i in range (len(year)):
#     #First the expected values
#     temp_pow_model_data[i]=temp_power_model(year[i], temp_pow_popt[0], temp_pow_popt[1], temp_pow_popt[2], temp_pow_popt[3], temp_pow_popt[4])
    
#     #Then calculate the residuals
#     temp_pow_residual[i] = mean[i]-temp_pow_model_data[i]

#Plotting for the temp_power model
# plt.figure(figsize = (8, 16))

#First subplot corresponding to the original data set and the linear model's fitting.
# plt.subplot(2, 1, 1)
# plt.plot(year, temp_power_model(year, temp_pow_popt[0], temp_pow_popt[1], temp_pow_popt[2], temp_pow_popt[3], temp_pow_popt[4]), label = "Temp Power Model Curve Fit", color="blue")
# plt.errorbar(year, mean, yerr=unc, fmt='o', capsize=0, ecolor = "black", label = "Data", marker = ".", markersize = 10)
# plt.xlabel("Year")
# plt.ylabel(r'$CO_2\:Level\:(in\:unit\:of\:ppm)$')
# plt.xticks(np.arange(1959, 2023, step = 5))
# plt.legend()
# plt.title("Mean CO$_2$ level with Temp Power model curve fitting")

# #Second subplot for the residuals, with a newly defined variable lin_zero_err as the line where the residual is 0.
# zero_residual_line = np.zeros(len(year))
# plt.subplot(2, 1, 2)
# plt.plot(year, zero_residual_line)
# plt.plot(year, temp_pow_residual,'o', label = "Residual of the temp power model versus actual data", marker = ".", color = "red")
# plt.xlabel("Year")
# plt.ylabel(r'$Error\:of\:CO_2\:Level\:in\:the\:temp_power\:model\:(in\:unit\:of\:ppm)$')
# plt.xticks(np.arange(1959, 2023, step = 5))
# plt.legend()
# plt.title("Residuals from the temp power model")
# plt.show()

# temp_power_chi2=np.sum( (mean - temp_pow_model_data)**2 / unc**2 )
# temp_power_reduced_chi2 = temp_power_chi2/(mean.size - 2)

# print("Temp Power Chi squared ", temp_power_chi2)
# print("Temp Power Chi reduced squared ", temp_power_reduced_chi2)

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