# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 07:10:49 2025

Created by Rui (Richard) Chen Geng Li
Student number: 1003048454

Code created for PHY224 Fitting Exercise 3
Prof. Sergio De La Barrera
Ta: Weigeng Peng
Due February 6th, 2025
"""

#Note:
#The structure of this code is ordered in a similar fassion as the bullet point listed in the IntroFitting.pdf file
#With rare exceptions such as that at the very end of this script, there is a distinguishable section of scratches for testing out ideas, unused models and possibly saving them for future uses. Please disregard that scratch section for this exercise.


#Necessary modules
import numpy as np #Useful helper functions that can convert our data into formats acceptable to the curve fitting components.
from scipy.optimize import curve_fit #The main component of this fitting exercise
import matplotlib.pyplot as plt #Handles the plotting aspect of this lab


#Loading the dataset, until row 996 as after that there is nothing available for the average temperature anymore.
year, temp_mean, temp_std, co2_mean, co2_std = np.loadtxt("co2_temp_1000.csv", delimiter = ',', skiprows=1, max_rows=996, unpack=True)


#Model that we will be using for the histogram and curve fitting
def log_model(co2, A, B):
    return A*np.log(co2)+B

#Separation of the dataset based on the industrial periods
industrial_year=1769

pre_industrial_year = np.array([])
pre_industrial_temp_mean = np.array([])
pre_industrial_temp_std = np.array([])
pre_industrial_co2_mean = np.array([])
pre_industrial_co2_std = np.array([])

post_industrial_year = np.array([])
post_industrial_temp_mean = np.array([])
post_industrial_temp_std = np.array([])
post_industrial_co2_mean = np.array([])
post_industrial_co2_std = np.array([])

for i in range(len(year)):
    if year[i]<industrial_year:
        pre_industrial_year = np.append(pre_industrial_year, year[i])
        pre_industrial_temp_mean = np.append(pre_industrial_temp_mean, temp_mean[i])
        pre_industrial_temp_std = np.append(pre_industrial_temp_std, temp_std[i])
        pre_industrial_co2_mean = np.append(pre_industrial_co2_mean, co2_mean[i])
        pre_industrial_co2_std = np.append(pre_industrial_co2_std, co2_std[i])
    elif year[i]>=industrial_year:
        post_industrial_year = np.append(post_industrial_year, year[i])
        post_industrial_temp_mean = np.append(post_industrial_temp_mean, temp_mean[i])
        post_industrial_temp_std = np.append(post_industrial_temp_std, temp_std[i])
        post_industrial_co2_mean = np.append(post_industrial_co2_mean, co2_mean[i])
        post_industrial_co2_std = np.append(post_industrial_co2_std, co2_std[i])



#Plotting the histogram
bins_count = 40 #Setting the number of bins for the histogram
plt.figure(figsize = (8, 16))

#First subplot corresponding to the temperature
plt.subplot(2, 1, 1)

plt.hist(pre_industrial_temp_mean, bins = bins_count, label = "Pre-Industrial (<" +str(industrial_year)+")", density = True, alpha = 0.5)
plt.hist(post_industrial_temp_mean, bins = bins_count, label = "Post-Industrial (>=" +str(industrial_year)+")", density = True, alpha = 0.5)
plt.xlabel("Temperature Changes (°C)")
plt.ylabel("Density")
plt.legend()
plt.title("Temperature Change Distribution Before and During the Industrial Revolution")

plt.subplot(2, 1, 2)

plt.hist(pre_industrial_co2_mean, bins = bins_count, label = "Pre-Industrial (<" +str(industrial_year)+")", density = True, alpha = 0.5)
plt.hist(post_industrial_co2_mean, bins = bins_count, label = "Post-Industrial (>=" +str(industrial_year)+")", density = True, alpha = 0.5)
plt.xlabel("CO2 levels (in unit of ppm)")
plt.ylabel("Density")
plt.legend()
plt.title("CO2 Level Before and During the Industrial Revolution")

plt.show()


#Calculating the means of the datapoints corresponding to the temperature and co2 levels of different time periods

mean_pre_ind_temp = np.mean(pre_industrial_temp_mean)
std_pre_ind_temp = np.std(pre_industrial_temp_mean)
mean_pre_ind_co2 = np.mean(pre_industrial_co2_mean)
std_pre_ind_co2 = np.std(pre_industrial_co2_mean)

mean_post_ind_temp = np.mean(post_industrial_temp_mean)
std_post_ind_temp = np.std(post_industrial_temp_mean)
mean_post_ind_co2 = np.mean(post_industrial_co2_mean)
std_post_ind_co2 = np.std(post_industrial_co2_mean)

print(f'Before Industrial Revolution: Temp Mean Diff = {mean_pre_ind_temp}°C, Temp Diff Std = {std_pre_ind_temp}°C, CO2 Mean = {mean_pre_ind_co2} ppm, CO2 Std = {std_pre_ind_co2} ppm')
print(f'During Industrial Revolution: Temp Mean Diff = {mean_post_ind_temp}°C, Temp Diff Std = {std_post_ind_temp}°C, CO2 Mean = {mean_post_ind_co2} ppm, CO2 Std = {std_post_ind_co2} ppm')

# Check for overlap: compare the distance between the mean values relative to the spread
overlap_temp_check = (mean_post_ind_temp - mean_pre_ind_temp) < (std_pre_ind_temp + std_post_ind_temp)
overlap_co2_check = (mean_post_ind_co2 - mean_pre_ind_co2) < (std_pre_ind_co2 + std_post_ind_co2)

print(f'Do the temperature periods overlap: {overlap_temp_check}')
print(f'Do the CO2 periods overlap: {overlap_co2_check}')


#Plotting the level of CO2 just to see where we get see "spiking up"
plt.errorbar(year, co2_mean, yerr=co2_std , fmt='o', capsize=0, ecolor = "red", label = "Measured level of CO2", marker = ".", markersize = 1)
plt.xlabel("Year")
plt.ylabel(r'$CO_2$ Level (in unit of ppm)')
plt.legend()
plt.title(r'$CO_2$ level measured over the years')
plt.xticks(np.arange(1000, 2025, step = 100))

####################################################################################################################################################################
#We could see from above that approx 1900 the CO2 level spiked upwards, we can see that this might be our "fully industrialized" period
#Now repeat everything we had above.

#Separation of the dataset based on the industrial periods
industrial_year=1930

pre_industrial_year = np.array([])
pre_industrial_temp_mean = np.array([])
pre_industrial_temp_std = np.array([])
pre_industrial_co2_mean = np.array([])
pre_industrial_co2_std = np.array([])

post_industrial_year = np.array([])
post_industrial_temp_mean = np.array([])
post_industrial_temp_std = np.array([])
post_industrial_co2_mean = np.array([])
post_industrial_co2_std = np.array([])

for i in range(len(year)):
    if year[i]<industrial_year:
        pre_industrial_year = np.append(pre_industrial_year, year[i])
        pre_industrial_temp_mean = np.append(pre_industrial_temp_mean, temp_mean[i])
        pre_industrial_temp_std = np.append(pre_industrial_temp_std, temp_std[i])
        pre_industrial_co2_mean = np.append(pre_industrial_co2_mean, co2_mean[i])
        pre_industrial_co2_std = np.append(pre_industrial_co2_std, co2_std[i])
    elif year[i]>=industrial_year:
        post_industrial_year = np.append(post_industrial_year, year[i])
        post_industrial_temp_mean = np.append(post_industrial_temp_mean, temp_mean[i])
        post_industrial_temp_std = np.append(post_industrial_temp_std, temp_std[i])
        post_industrial_co2_mean = np.append(post_industrial_co2_mean, co2_mean[i])
        post_industrial_co2_std = np.append(post_industrial_co2_std, co2_std[i])



#Plotting the histogram
bins_count = 40 #Setting the number of bins for the histogram
plt.figure(figsize = (8, 16))

#First subplot corresponding to the temperature
plt.subplot(2, 1, 1)

plt.hist(pre_industrial_temp_mean, bins = bins_count, label = "Pre-Industrial (<" +str(industrial_year)+")", density = True, alpha = 0.5)
plt.hist(post_industrial_temp_mean, bins = bins_count, label = "Post-Industrial (>=" +str(industrial_year)+")", density = True, alpha = 0.5)
plt.xlabel("Temperature Changes (°C)")
plt.ylabel("Density")
plt.legend()
plt.title("Temperature Change Distrubution Before and During the Industrial Revolution")

plt.subplot(2, 1, 2)

plt.hist(pre_industrial_co2_mean, bins = bins_count, label = "Pre-Industrial (<" +str(industrial_year)+")", density = True, alpha = 0.5)
plt.hist(post_industrial_co2_mean, bins = bins_count, label = "Post-Industrial (>=" +str(industrial_year)+")", density = True, alpha = 0.5)
plt.xlabel("CO2 levels (in unit of ppm)")
plt.ylabel("Density")
plt.legend()
plt.title("CO2 Level Before and During the Industrial Revolution")

plt.show()


#Calculating the means of the datapoints corresponding to the temperature and co2 levels of different time periods

mean_pre_ind_temp = np.mean(pre_industrial_temp_mean)
std_pre_ind_temp = np.std(pre_industrial_temp_mean)
mean_pre_ind_co2 = np.mean(pre_industrial_co2_mean)
std_pre_ind_co2 = np.std(pre_industrial_co2_mean)

mean_post_ind_temp = np.mean(post_industrial_temp_mean)
std_post_ind_temp = np.std(post_industrial_temp_mean)
mean_post_ind_co2 = np.mean(post_industrial_co2_mean)
std_post_ind_co2 = np.std(post_industrial_co2_mean)

print(f'Before Industrial Revolution: Temp Mean Diff = {mean_pre_ind_temp}°C, Temp Diff Std = {std_pre_ind_temp}°C, CO2 Mean = {mean_pre_ind_co2} ppm, CO2 Std = {std_pre_ind_co2} ppm')
print(f'During Industrial Revolution: Temp Mean Diff = {mean_post_ind_temp}°C, Temp Diff Std = {std_post_ind_temp}°C, CO2 Mean = {mean_post_ind_co2} ppm, CO2 Std = {std_post_ind_co2} ppm')

# Check for overlap: compare the distance between the mean values relative to the spread
overlap_temp_check = (mean_post_ind_temp - mean_pre_ind_temp) < (std_pre_ind_temp + std_post_ind_temp)
overlap_co2_check = (mean_post_ind_co2 - mean_pre_ind_co2) < (std_pre_ind_co2 + std_post_ind_co2)

print(f'Do the temperature periods overlap: {overlap_temp_check}')
print(f'Do the CO2 periods overlap: {overlap_co2_check}')


####################################################################################################################################################
#Plotting the prediction model
#Curve fitting with the new parameter
#From the PDF, we will discount the first 700 years of data, and use the latter ones for a more accurate prediction
industrial_year=1000

post_industrial_year = np.array([])
post_industrial_temp_mean = np.array([])
post_industrial_temp_std = np.array([])
post_industrial_co2_mean = np.array([])
post_industrial_co2_std = np.array([])

for i in range(len(year)):
    if year[i]>=industrial_year:
        post_industrial_year = np.append(post_industrial_year, year[i])
        post_industrial_temp_mean = np.append(post_industrial_temp_mean, temp_mean[i])
        post_industrial_temp_std = np.append(post_industrial_temp_std, temp_std[i])
        post_industrial_co2_mean = np.append(post_industrial_co2_mean, co2_mean[i])
        post_industrial_co2_std = np.append(post_industrial_co2_std, co2_std[i])

popt, pcov = curve_fit(log_model, post_industrial_co2_mean, post_industrial_temp_mean, sigma = post_industrial_temp_std, absolute_sigma = True)
print("Parameter A's Uncertainty: ", np.sqrt(pcov[0][0]))
print("Parameter B's Uncertainty: ", np.sqrt(pcov[1][1]))

#Residual Calculation

model_data = post_industrial_co2_mean*0
model_residual = post_industrial_co2_mean*0

for c in range(len(post_industrial_co2_mean)):
    model_data[c] = log_model(post_industrial_co2_mean[c], popt[0], popt[1])
    model_residual[c] = post_industrial_temp_mean[c] - model_data[c]

#################################################################################################
chi2=np.sum( (post_industrial_co2_mean - model_data)**2 / std_post_ind_temp**2 )
reduced_chi2 = chi2/(post_industrial_co2_mean.size - len(popt))

print("Periodic Chi squared ", chi2)
print("Periodic Chi reduced squared ", reduced_chi2)


plt.figure(figsize = (8, 16))

#First subplot corresponding to the original data set and the periodic model's fitting.
plt.subplot(2, 1, 1)
plt.errorbar(post_industrial_co2_mean, post_industrial_temp_mean, yerr=post_industrial_temp_std, fmt='o', capsize=0, ecolor = "red", label = "Data", marker = ".", color = "red", markersize = 2)
plt.plot(post_industrial_co2_mean, log_model(post_industrial_co2_mean, popt[0], popt[1]), label = "Log Model Curve Fit: " + f'ΔT={np.round(popt[0],decimals=1)}Log($CO_2$)+{np.round(popt[1], decimals=1)}', color="blue", linewidth =0.75)
plt.ylabel("Temperature Change (°C)")
plt.xlabel(r'$CO_2$ Level (in unit of ppm)')
plt.legend()
plt.title("Mean CO$_2$ level versus Temperature Change with log model curve fitting")

#Second subplot for the residuals, with a newly defined variable zero_residual_line as the line where the residual is 0.
zero_residual_line = np.zeros(len(post_industrial_co2_mean))
plt.subplot(2, 1, 2)
plt.plot(post_industrial_co2_mean, zero_residual_line, label="Zero residual line")
plt.errorbar(post_industrial_co2_mean, model_residual, yerr=post_industrial_temp_std, fmt='o', capsize=0, ecolor = "red", label = "Residual of the log model versus actual data", marker = ".", markersize = 10)
plt.ylabel("Temperature Change (°C)")
plt.xlabel(r'Error of $CO_2$ Level (in unit of ppm)')
plt.legend()
plt.title("Residuals from the log model")
plt.show()