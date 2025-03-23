# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:12:00 2025

Turki Almansoori
Rui (Richard) Chen Geng Li

Code created for PHY224 Free Choice Lab: Interferometer
Prof. Sergio De La Barrera
Due March 23rd, 2025
"""

#Necessary modules
import numpy as np 
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 

#Theoretical Values of certain variables
wavelength_theo = 534e-9 #in unit of nano meters
thermal_coefficient_theo = 23e-6 #Coefficient of thermal expansion of aluminium per degrees celsius


#Data imports
w_reading, w_dN, w_reading_unc, w_dN_unc = np.loadtxt("Final_Wavelength_data.csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)

#Prediction Model
#Wavelength
def deltaN(x_val, a):
    return x_val*(2.0/a)

w_dN_Per_dx = np.zeros(w_dN.size)
w_dN_Per_dx[0]=w_reading[0]
for n in range(1, len(w_dN)):
    w_dN_Per_dx[n]=w_dN[n]+w_dN_Per_dx[n-1]

for w in range(1, len(w_dN_unc)):
    w_dN_unc[w]=np.sqrt((w_dN_unc[w-1]**2) + (w_dN_unc[w]**2))
    
#Curve_fit
w_popt, w_pcov = curve_fit(deltaN, w_reading, w_dN_Per_dx, p0=(0.57), sigma=w_dN_unc, absolute_sigma = True)

#Plotting
#Main Plot
plt.errorbar(w_reading, w_dN_Per_dx, xerr=w_reading_unc, yerr=w_dN_unc, fmt = "o", color = "red", label = "Measured Data")
plt.plot(w_reading, deltaN(w_reading, *w_popt), color = "blue", label="Prediction Data")
plt.plot(w_reading, w_reading*2.0/(wavelength_theo*1e6), color = "green", label="Theoretical Prediction")
plt.xlabel("Change in unit of micrometer (µm)")
plt.ylabel("Change in unit of fringe count")
plt.title("Wavelength Prediction (change in mirror distance in µm versus change in fringe count)")
plt.legend()
plt.show()

#Residuals
w_residual = w_dN_Per_dx - deltaN(w_reading, *w_popt)
w_zero_line = np.zeros(w_residual.size)
plt.plot(w_reading, w_zero_line, color = "blue", label = "Reference zero residual line")
plt.errorbar(w_reading, w_residual, xerr=w_reading_unc, yerr=w_dN_unc, color = "red", fmt = "o", label = "Residual between measurement and prediction")
plt.xlabel("Change in unit of micrometer (µm)")
plt.ylabel("Error in the fringe count")
plt.title("Wavelength Residual between the measurement data and the prediction data")
plt.legend()
plt.show()

#Printing the predicted wavelength
print("Predicted Wavelength = ", w_popt[0]*1000.0, "nm ± ", np.sqrt(w_pcov[0][0])*1000.0, "nm")

w_std = np.sqrt(np.sum((w_reading-np.mean(w_reading))**2)/(w_reading.size-1))
reduced_chi2 = np.sum((w_reading-deltaN(w_reading, *w_popt))**2/w_std**2) /(w_reading.size - w_popt.size)
print ("The Reduced Chi Square Value is: ", reduced_chi2)

reduced_chi2 = np.sum((w_reading-deltaN(w_reading, *w_popt))**2/w_dN_unc**2) /(w_reading.size - w_popt.size)
print ("The Reduced Chi Square Value is: ", reduced_chi2)

###################### Index of Refraction ################################

#Reading Data
ir_reading, ir_min, ir_dN, ir_min_unc, ir_dN_unc = np.loadtxt("Index_of_Refraction_2.csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)

# ir_dN=ir_dN[::-1]

ir_dN_Per_dx = np.zeros(ir_dN.size)
ir_dN_Per_dx[0]=ir_dN[0]
for n in range(1, len(ir_dN)):
    ir_dN_Per_dx[n]=ir_dN[n]+ir_dN_Per_dx[n-1]


ir_reading += ir_min/60.0
ir_reading_unc = ir_min_unc/60.0

#Variables
thickness = 7e-3 #mm converted to m

ir_reading*=-1.0
ir_reading-=ir_reading[0]
ir_reading = np.radians(ir_reading)

ir_reading_unc = np.radians(ir_reading_unc)

ir_dN_unc[0]=0.1
ir_dN_unc_old = np.array(ir_dN_unc)
#dN Uncertainty propagation
for u in range(1, len(ir_dN_unc)):
    ir_dN_unc[u]=np.sqrt((ir_dN_unc[u-1]**2) + (ir_dN_unc[u]**2))

#Prediction Model
#Index of Refraction
def index_refraction_2(x_val, a):
    return (thickness/(w_popt[0]*1e-6))*((x_val)**2)*(1.0-(1.0/a))
    #return (thickness/(0.534*1e-6))*((x_val)**2)*(1.0-(1.0/a))

ir_popt, ir_pcov = curve_fit(index_refraction_2, ir_reading, ir_dN_Per_dx, p0=(1.68), sigma = ir_dN_unc, absolute_sigma = True)
#Excluding the last 4 data points as angles are getting larger than what is appropriate for small angle approximation.
# ir_popt, ir_pcov = curve_fit(index_refraction_2, ir_reading[:-4], ir_dN_Per_dx[:-4], p0=(1.68), sigma = ir_dN_unc[:-4], absolute_sigma = True)

plt.errorbar(ir_reading, ir_dN_Per_dx, xerr=ir_reading_unc, yerr=ir_dN_unc, fmt = "o", color = "red", label = "Index of Refraction Measurement")
plt.plot(ir_reading, index_refraction_2(ir_reading, *ir_popt), color = "blue", label = "Prediction")
plt.fill_between(ir_reading, (thickness/(0.534*1e-6))*((ir_reading)**2)*(1.0-(1.0/1.4)), (thickness/(0.534*1e-6))*((ir_reading)**2)*(1.0-(1.0/1.76)), color='green', alpha=0.35, interpolate=True, label = "Theoretical Prediction Range")
plt.xlabel("Change in unit of radians (rad)")
plt.ylabel("Change in unit of fringe count")
plt.title("Index of Refraction Prediction (change in plastic square rotation in rad versus change in fringe count)")
plt.legend()
plt.show()

#Residuals
ir_residual = ir_dN_Per_dx - index_refraction_2(ir_reading, *ir_popt)
ir_zero_line = np.zeros(ir_residual.size)
plt.plot(ir_reading, ir_zero_line, color = "blue", label = "Reference zero residual line")
plt.errorbar(ir_reading, ir_residual, xerr=ir_reading_unc, yerr=ir_dN_unc, color = "red", fmt = "o", label = "Residual between measurement and prediction")
plt.xlabel("Change in unit of radians (rad)")
plt.ylabel("Error in the fringe count")
plt.title("Index of Refraction Residual between the measurement data and the prediction data")
plt.legend()
plt.show()

print("Predicted Index of Refraction: n = ", ir_popt[0], " ± ", np.sqrt(ir_pcov[0][0]))

#Reduced Chi Squared Value
ir_prediction = index_refraction_2(ir_reading, *ir_popt)
ir_reduced_chi2 = np.sum((ir_reading-ir_prediction)**2/(ir_dN_unc**2)) /(ir_reading.size - ir_popt.size)
print ("The Reduced Chi Square Value is: ", ir_reduced_chi2)


###################### Thermal Expansion of Aluminium ################################

#Reading Data
t_temp, t_dN, t_current = np.loadtxt("Thermal_Expansion_Data_Combined.csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)

#Variables
L0=0.09012 #Aluminium rod length at base temperature


#Prediction Model
#Thermal Expansion
# def thermal_expansion(x_val, a):
#     return L0*np.exp()**(a*x_val)

def thermal_expansion(x_val, a):
    # return (2.0*L0/(0.534e-6))*a*(x_val-t_temp[0])
    return (2.0*L0/(w_popt[0]*1e-6))*a*(x_val-t_temp[0])


t_dN_total = np.zeros(t_dN.size)
t_dN_total[0]=t_dN[0]
for n in range(1, len(t_dN)):
    t_dN_total[n]=t_dN[n]+t_dN_total[n-1]
    
    

t_popt, t_pcov = curve_fit(thermal_expansion, t_temp, t_dN_total, p0=(29.33790993115546746e-06))

plt.errorbar(t_temp, t_dN_total, fmt="o", color = "red", label = "Measurement Data")
plt.plot(t_temp, thermal_expansion(t_temp, *t_popt), color = "blue", label = "Prediction")
plt.plot(t_temp, (2.0*L0/(wavelength_theo))*thermal_coefficient_theo*(t_temp-t_temp[0]), color = "green", label = "Theoretical Prediction")
plt.xlabel("Temperature in degrees Celsius (C)")
plt.ylabel("Change in unit of fringe counts")
plt.title("Thermal Coefficient of Aluminium Prediction")
plt.legend()
plt.show()

#Residuals
t_residual = t_dN_total - thermal_expansion(t_temp, *t_popt)
t_zero_line = np.zeros(t_residual.size)
plt.plot(t_temp, t_zero_line, color = "blue", label = "Reference zero residual line")
plt.errorbar(t_temp, t_residual, color = "red", fmt = "o", label = "Residual between measurement and prediction")
plt.xlabel("Temperature in degrees Celsius (C)")
plt.ylabel("Error in the fringe count")
plt.title("Thermal Coefficient Residual between the measurement data and the prediction data")
plt.legend()
plt.show()

print("Predicted thermal expansion coefficient for aluminium: ", t_popt[0], "/C", " ± ", np.sqrt(t_pcov[0][0]), "/C")

# #Reduced Chi Squared Value
# t_prediction = thermal_expansion(t_temp, *t_popt)
# t_reduced_chi2 = np.sum((t_temp-t_prediction)**2/(t_dN_unc**2)) /(t_temp.size - t_popt.size)
# print ("The Reduced Chi Square Value is: ", t_reduced_chi2)



