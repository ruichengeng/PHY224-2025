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

popt, pcov = curve_fit(log_model, co2_mean, temp_mean, sigma = temp_std, absolute_sigma = True)

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
bins_count = 10 #Setting the number of bins for the histogram
plt.figure(figsize = (8, 16))

#First subplot corresponding to the temperature
plt.subplot(2, 1, 1)

plt.hist(pre_industrial_temp_mean, bins = bins_count, label = "Pre-Industrial (<" +str(industrial_year)+")", density = True, alpha = 0.5)
plt.hist(post_industrial_temp_mean, bins = bins_count, label = "Post-Industrial (>=" +str(industrial_year)+")", density = True, alpha = 0.5)
plt.xlabel("Temperature Differences (°C)")
plt.ylabel("Density")
plt.legend()
plt.title("Temperature Distribution")




plt.show()

# plt.errorbar(year, temp_mean, yerr=temp_std, fmt='o', capsize=0, ecolor = "red", label = "Data", marker = ".", color = "red", markersize = 2)