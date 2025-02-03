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
#Done by getting a new array that contains only those data entries with the positive uncertainties.
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

#Printing out the uncertainties
print("Uncertainty of the parameters are: u(A)=", period_parameter_unc[0], " u(B)=", period_parameter_unc[1], " u(C)=", period_parameter_unc[2],
      " u(D)=", period_parameter_unc[3], " u(E)=", period_parameter_unc[4], " u(phi)=", period_parameter_unc[5])

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
plt.xticks(np.arange(1974, 2025, step = 5))
plt.legend()
plt.title("Mean CO$_2$ level with periodic model curve fitting")

#Second subplot for the residuals, with a newly defined variable zero_residual_line as the line where the residual is 0.
zero_residual_line = np.zeros(len(dec_date))
plt.subplot(2, 1, 2)
plt.plot(dec_date, zero_residual_line, label="Zero residual line")
plt.errorbar(dec_date, periodic_residual, yerr=unc, fmt='o', capsize=0, ecolor = "red", label = "Residual of the periodic model versus actual data", marker = ".", markersize = 10)
plt.xlabel("Year")
plt.ylabel(r'Error of $CO_2$ Level in the periodic model (in unit of ppm)')
plt.xticks(np.arange(1974, 2025, step = 5))
plt.legend()
plt.title("Residuals from the periodic model")
plt.show()


#################################################################################################
#Prediction into 50 years in the future

prediction_year = np.arange(year[0], year[-1]+50.0, 0.2)

plt.errorbar(dec_date, mean, yerr=unc, fmt='o', capsize=0, ecolor = "red", label = "Data", marker = ".", color = "red", markersize = 2)
plt.plot(prediction_year, periodic_model(prediction_year, period_popt[0], period_popt[1], period_popt[2], period_popt[3], period_popt[4], period_popt[5]), label = "Periodic Model Prediction", color="blue", linewidth =0.75)
plt.xlabel("Year")
plt.ylabel(r'$CO_2$ Level (in unit of ppm)')
plt.xticks(np.arange(1974, 2075, step = 10))
plt.legend()
plt.title("Periodic Model's Prediction of Future Mean CO$_2$ level (50 years from 2024)")

#################################################################################################
#Report questions

########################Month with the higest CO2 value##########################################
        
#Method 2: counting how many months in each year is the max
month_max_count = np.zeros(12)
current_year = 1974.0
current_co2 = 0.0
current_month = 0

for d in range(len(month)):
    if dec_date[d]>=current_year and dec_date[d]<current_year+1.0: #Checks between initial of the surpassed year (inclusive) and the initial of the following year (exclusive)
        if mean[d]>current_co2:
            current_co2=mean[d]
            current_month = int(month[d])
    else: 
        month_max_count[current_month-1]+=1
        current_month=0
        current_co2=0.0
        current_year = np.floor(current_year)+1.0


print("Month with the highest CO2 level value is: ", np.where(month_max_count==np.max(month_max_count))[0]+1)


#################Time of CO2 passing twice 285ppm#############################################

co2_to_pass = 2*285
temp_year = 1960.

while (periodic_model(temp_year, period_popt[0], period_popt[1], period_popt[2], period_popt[3], period_popt[4], period_popt[5])<co2_to_pass):
    temp_year+=0.01

print("We predict that the CO2 level can reach and then pass 570ppm by: ", temp_year)


########How long until CO2 minimum of a year pass the CO2 maximum of 2000######################

co2_max_2000 = np.array([0.0,0.0]) #1st number correspond to the position of the time, and 2nd number correspond to the value of co2 level

for y in range(len(dec_date)):
    if dec_date[y]>=2000.0 and dec_date[y]<2001.0: #Checks between 2000.0 (inclusive) and 2001.0 (exclusive)
        if mean[y]>co2_max_2000[1]:
            co2_max_2000[1]=mean[y]
            co2_max_2000[0] = dec_date[y]


#To check when is the minimum of a year is going to surpass the maximum of 2000. We will see when it is the last time we don't surpass the maximum.

unsurpassed_year = 0.0

for i in range (len(dec_date)):
    if mean[i] < co2_max_2000[1]:
        unsurpassed_year = dec_date[i]


#Now that we have the last unsurpassed year, we will floor this value and add 1 to indicate the very first year where the minimum is less than the maximum of 2000.

surpassed_year=np.floor(unsurpassed_year)+1.0

#Finding the minimum of that year
co2_min_surpassed = np.array([0.0,1000.0])
for d in range(len(dec_date)):
    if dec_date[d]>=surpassed_year and dec_date[d]<surpassed_year+1.0: #Checks between initial of the surpassed year (inclusive) and the initial of the following year (exclusive)
        if mean[d]<co2_min_surpassed[1]:
            co2_min_surpassed[1]=mean[d]
            co2_min_surpassed[0] = dec_date[d]

print("Maximunm CO2 level (ppm) in 2000 is: ", co2_max_2000[1], ", occuring at the decimal date: ", co2_max_2000[0])
print("The year that the minimum will pass the maximum of year 2000 is: ", int(surpassed_year))
print("This year will have the minimum CO2 level (ppm) of: ", co2_min_surpassed[1], ", occuring at the decimal date: ", co2_min_surpassed[0])
print("The amount of time it took to surpass is: ", co2_min_surpassed[0]-co2_max_2000[0])










###################################################################################################################################################################

######################################### SCRATCHES PLEASE DO NOT GRADE#########################################################################




#Trying to sort using a dictionary, does not account for the lack of data points for incomplete years
#Key is the month, then of the 2 numbers in the value: 1 correspond to the number of data entry and the other correspond to the total value
# co2_by_month = {1:[0,0], 2:[0,0], 3:[0,0], 4:[0,0], 5:[0,0], 6:[0,0], 7:[0,0], 8:[0,0], 9:[0,0], 10:[0,0], 11:[0,0], 12:[0,0]} 

# #Separate the values of the dataset based on the month they correspond to
# for m in range(len(month)):
#     co2_by_month[month[m]][0]+=1
#     co2_by_month[month[m]][1]+=mean[m]
        
# #Using the data obtained above, we will find the mean of the co2 levels based on the month and the number of datasets
# co2_by_month_mean = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}

# #Getting the new mean value per month
# for a in co2_by_month:
#     co2_by_month_mean[a]=co2_by_month[a][1]/co2_by_month[a][0]


# #Simple forloop sorting
# highest_month = 0
# highest_month_mean = 0
# for h in co2_by_month_mean:
#     if co2_by_month_mean[h] >= highest_month_mean:
#         highest_month=h
#         highest_month_mean = co2_by_month_mean[h]