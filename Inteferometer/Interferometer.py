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

#Data imports
reading, dN, reading_unc, dN_unc = np.loadtxt("Final_Wavelength_data - Copy.csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)

#Prediction Model
#Wavelength
def deltaN(x_val, a):
    return x_val*(2.0/a)

dN_Per_dx = np.zeros(dN.size)
dN_Per_dx[0]=reading[0]
for n in range(1, len(dN)):
    dN_Per_dx[n]=dN[n]+dN_Per_dx[n-1]
    
#Curve_fit
cf_popt, cf_pcov = curve_fit(deltaN, reading, dN_Per_dx, p0=(0.57), sigma=dN_unc, absolute_sigma = True)
# cf_popt, cf_pcov = curve_fit(deltaN, reading, dN, p0=(0.5), sigma=dN_unc, absolute_sigma = True)

#Plotting
# plt.errorbar(reading, dN, xerr=reading_unc, yerr=dN_unc, label = "Measured Data")
plt.errorbar(reading, dN_Per_dx, xerr=reading_unc, yerr=dN_unc, label = "Measured Data")
plt.plot(reading, deltaN(reading, *cf_popt), color = "red", label="Prediction Data")
plt.plot(reading, deltaN(reading, 0.57)-10.0, color = "black", label="temp")
plt.legend()
plt.show()

#Printing the predicted wavelength
print("Predicted Wavelength = ", cf_popt[0]*1000.0, "nm ± ", np.sqrt(cf_pcov[0][0])*1000.0, "nm")

position_std = np.sqrt(np.sum((reading-np.mean(reading))**2)/reading.size)
#position_std += reading_unc*2.0
#position_std = reading_unc
reduced_chi2 = np.sum((reading-deltaN(reading, *cf_popt))**2/position_std**2) /(reading.size - cf_popt.size)
print ("The Reduced Chi Square Value is: ", reduced_chi2)

#position_std = np.sqrt(np.sum((position-np.mean(position))**2)/position.size)
reduced_chi2 = np.sum((reading-deltaN(reading, 0.57)-10.0)**2/position_std**2) /(reading.size - cf_popt.size)
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

# ir_reading_init = ir_reading[0]
# ir_reading *= -1.0
# ir_reading += ir_reading_init
#Variables
gamma = 534e-9 #Temp for the wavelength
thickness = 7e-3 #mm converted to m


ir_reading*=-1.0

# ir_reading=ir_reading[::-1]
# ir_dN_Per_dx=ir_dN_Per_dx[::-1]
# ir_dN_unc = ir_dN_unc[::-1]
# ir_reading_unc = ir_reading_unc[::-1]

ir_dN_unc_old = np.array(ir_dN_unc)
#dN Uncertainty propagation
for u in range(1, len(ir_dN_unc)):
    ir_dN_unc[u]=np.sqrt((ir_dN_unc[u-1]**2) + (ir_dN_unc[u]**2))

#Prediction Model
#Index of Refraction
# def index_refraction(x_val, a, b):
#     #a is t for thickness
#     #b is theta
#     return (((x_val*gamma/(2.0*a))+np.cos(b)-1)**2 + np.sin(b)**2)/(2.0*(1.0-np.cos(b)-(x_val*gamma/(2.0*a))))


def index_refraction_2(x_val, a, b):
    #return (thickness/cf_popt[0])*((x_val-b)**2)*(1.0-(1.0/a))
    return (thickness/gamma)*((x_val-b)**2)*(1.0-(1.0/a))

# ir_popt, ir_pcov = curve_fit(index_refraction_2, ir_reading[1:], ir_dN_Per_dx[1:], sigma = np.ones(ir_reading[1:].size)*0.5, absolute_sigma = True)
#ir_popt, ir_pcov = curve_fit(index_refraction_2, ir_reading, ir_dN_Per_dx, p0=(1.5, -30.0),sigma = ir_dN_unc, absolute_sigma = True, maxfev = 100000)


plt.errorbar(ir_reading, ir_dN_Per_dx, xerr=ir_min_unc*0.01, color = "red", label = "Index of Refraction Measurement")
#plt.plot(ir_reading, index_refraction_2(ir_reading, *ir_popt), color = "blue", label = "Prediction")
# plt.plot(np.arange(-30.0, 0.0, 0.5), index_refraction_2(np.arange(-30.0, 0.0, 0.5), 1.015, -30.0), color = "blue")
plt.plot(ir_reading, index_refraction_2(ir_reading, 1+1.5e-4, -30.0), color = "blue")

plt.legend()



###################### Thermal Expansion of Aluminium ################################

#Reading Data


#Variables
L0=1.0 #Temp, length at base temperature

#Prediction Model
#Thermal Expansion
def thermal_expansion(x_val, a):
    return L0*np.exp()**(a*x_val)