# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 06:06:02 2025

Turki Almansoori
Rui (Richard) Chen Geng Li
Student number: 1003048454

Code created for PHY224 Fitting Exercise 3
Prof. Sergio De La Barrera
Ta: Weigeng Peng
Due February 23rd, 2025
"""

#Necessary modules
import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 


#Setting resolution of image and importing raw data
plt.figure(dpi=200)
source, voltage, current, v_unc, c_unc = np.loadtxt("circuit_diode_data.csv", delimiter = ',', skiprows=2, unpack=True)


def exponential_model(x_val, a, b, c, d):
    return a*(b**(c*x_val+d))

def shockley_model(x_val, a,b):
    return -1*current[0]*(np.exp((x_val-b)*a)-1.0)

#Defining functions input_power_model,... which calculates the parameters and 
#model fit, the uncertainties, the residuals and reduced chi-squared. Done for
#both negative and positive portions. This is done for code legibility and
#simplicity
def input_exponential_model(voltage=voltage,current=current,v_unc=v_unc,c_unc=c_unc, 
                      fit=False):
    #Initial guess based off desmos equation modeling
    popt, pcov = curve_fit(exponential_model, voltage, current,
                                   p0 = (0.6353310698, 1031.75089, 3.723623702, 
                                         -2.268601667), 
                                   sigma=c_unc, 
                                   absolute_sigma = True, 
                                   maxfev = 10000)
    a,b,c,d = popt[0], popt[1], popt[2], popt[3]
    
    ###Calculating the uncertainty of all variables
    #Note pvar is a diagonalised array of the variance of the parameters of the
    #model, to get the value of uncertainty of the parameters, we sqrt the variance.
    model_current = exponential_model(voltage, a, b, c, d)
    pstd = np.sqrt(np.diag(pcov))
    
    #Calculating reduced chi-squared
    residuals = current - model_current
    dof = len(current) - len(popt)  # Degrees of freedom (N - m)
    rchi_squared = np.sum((residuals/c_unc)**2)/dof
    
    if not fit:
        return [exponential_model(voltage,a,b,c,d),popt]
    elif fit:
        return [residuals, rchi_squared, pstd]

def input_shockley_model(voltage=voltage,current=current,v_unc=v_unc,c_unc=c_unc, 
                      fit=False):
        
    popt, pcov = curve_fit(shockley_model, voltage, current, 
                                   p0 = (20.473,0.1), 
                                   sigma=c_unc, 
                                   absolute_sigma = True,
                                   maxfev = 10000)
    a,b = popt[0],popt[1]
    
    ###Calculating the uncertainty of all variables
    #Note pvar is a diagonalised array of the variance of the parameters of the
    #model, to get the value of uncertainty of the parameters, we sqrt the variance.
    model_current = shockley_model(voltage, a,b)
    pstd = np.sqrt(np.diag(pcov))
    
    #Calculating reduced chi-squared
    residuals = current - model_current
    dof = len(current) - len(popt)  # Degrees of freedom (N - m)
    rchi_squared = np.sum((residuals/c_unc)**2)/dof
    
    if not fit:
        return [shockley_model(voltage,a,b),popt]
    elif fit:
        return [residuals, rchi_squared, pstd]

main_graph = plt.subplot(5, 1, (1,3))
plt.title("Voltage current plot")

plt.plot(voltage,input_exponential_model()[0], 
          linewidth=1, label="fitted: power")
plt.plot(voltage,input_shockley_model()[0], 
          linewidth=1, label="fitted: shockley")

plt.errorbar(voltage, current, xerr= v_unc, yerr=c_unc, fmt='o', markersize=1, 
              color='black', ecolor='black', linewidth=0.5, label= "observed")
plt.xlabel("Voltage(V)")
plt.xticks(np.arange(-1.4,1.1,0.2))
plt.ylabel("Current(mA)")
plt.legend()
plt.grid()


linear_residual = plt.subplot(5,1,5)
plt.title("Residuals")
plt.axhline(y = 0, linewidth=1, color = 'r', linestyle = '-')
plt.errorbar(voltage, input_exponential_model(fit=True)[0], 
                  xerr = v_unc, yerr=c_unc, label='observed',
                  fmt='o', markersize=1, color='black', 
                  ecolor='black', linewidth=0.5)

plt.xlabel("Voltage(V)")
plt.xticks(np.arange(-1.4,1.1,0.2))
plt.ylabel("Error (mA)")

print("EXPONENTIAL MODEL:")
print("The reduced chi squared in the exponential model is", 
      input_exponential_model(fit=True)[1])      
param = input_exponential_model()[1] #Getting parameter values
pstd = input_exponential_model(fit=True)[2] #Getting parameter uncertainties
print("a = ", param[0], "+-", pstd[0])
print("b = ", param[1], "+-", pstd[1])
print("c = ", param[2], "+-", pstd[2])
print("d = ", param[3], "+-", pstd[3])

print("SHOCKLEY MODEL:")
print("The reduced chi squared in the shockley model is", 
      input_shockley_model(fit=True)[1])      
param = input_shockley_model()[1] #Getting parameter values
pstd = input_shockley_model(fit=True)[2] #Getting parameter uncertainties
print("a = ", param[0], "+-", pstd[0])
print("b = ", param[1], "+-", pstd[1])

plt.legend()
plt.show()
