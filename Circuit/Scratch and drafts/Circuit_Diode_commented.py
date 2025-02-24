"""
Diode exercise code submission

Turki Almansoori
Rui (Richard) Chen Geng Li

Code created for PHY224 Fitting Exercise 3
Prof. Sergio De La Barrera
Due February 23rd, 2025
"""

#Necessary modules
import numpy as np 
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 


#Loading the dataset and setting figure size
source, volt, amp, volt_unc, amp_unc = np.loadtxt("circuit_diode_data.csv", 
                                                  delimiter = ',', 
                                                  skiprows=2, unpack=True)
plt.figure(figsize = (8, 12))


#Defining model functions
def exponential_model(x_val, a, b):
    return a*np.exp(b*x_val)

def shockley_model(x_val, a):
    return -1*amp[0]*(np.exp(x_val*a)-1.0)

#Curve fitting with initial guesses based off trial and error
exp_popt, exp_pcov = curve_fit(exponential_model, volt, amp, 
                               p0 = (1e-6,22), 
                               sigma=amp_unc, absolute_sigma = True,
                               maxfev = 10000)
shock_popt, shock_pcov = curve_fit(shockley_model, volt, amp, 
                                   p0=(20.5), 
                                   sigma=amp_unc, absolute_sigma = True, 
                                   maxfev = 100000)

#Parameters uncertainties
exp_pcov = np.diag(exp_pcov)
shock_pcov = np.diag(shock_pcov)

print("Exponential Parameter a = ", exp_popt[0], u"\u00B1", np.sqrt(exp_pcov[0]))
print("Exponential Parameter b = ", exp_popt[1], u"\u00B1", np.sqrt(exp_pcov[1]))

print("Shockley Parameter a = ", shock_popt[0], u"\u00B1", np.sqrt(shock_pcov[0]))

####First subplot corresponding to the main voltage versus current graph
plt.subplot(2, 1, 1)

plt.title("Voltage current plot")
plt.errorbar(volt, amp, xerr=volt_unc, yerr=amp_unc, 
             fmt='o', ecolor="red", label="Measured Data", 
             color = "red", markersize = 5)

#plotting the model predictions
smooth_volt = np.arange(-1.4, 0.86, 0.01) #This variable is for the model plot, 
                                          #so that the curve reaches every data 
                                          #point
                                          
plt.plot(smooth_volt, exponential_model(smooth_volt, exp_popt[0],  #Exp model
                                        exp_popt[1]), 
         label="Exponential Model Fitting", color="blue", linewidth = 1)
plt.plot(smooth_volt, shockley_model(smooth_volt, shock_popt[0]), #Shockley model
         label="Shockley Model Fitting", color="green", linewidth = 1)

plt.xlabel("Voltage(V)")
plt.xticks(np.arange(-1.4,0.9,0.2))
plt.ylabel("Current(mA)")
plt.legend()

###Second subplot for the residuals
#Fitting the model data to the observed voltages for residuals
exp_model_data = exponential_model(volt, exp_popt[0], exp_popt[1])
shockley_model_data = shockley_model(volt, shock_popt[0])

exp_residual = amp - exp_model_data
shockley_residual = amp - shockley_model_data

#Plotting residuals
plt.subplot(2, 1, 2)
plt.axhline(y = 0, linewidth=1, color = 'r', linestyle = '-', label="Zero Residual Line") #zero line
plt.errorbar(volt, exp_residual, xerr = volt_unc, yerr=amp_unc, 
             fmt='o', capsize=0, color = "blue", ecolor = "blue", 
             label = "Residual of the exponential model versus actual data", 
             markersize = 5)
plt.errorbar(volt, shockley_residual, xerr = volt_unc, yerr=amp_unc, 
             fmt='o', capsize=0, color = "green", ecolor = "green", 
             label = "Residual of the Shockley model versus actual data", 
             markersize = 5)
plt.xlabel("Voltage(V)")
plt.ylabel("Error of the Current(mA) between models and the actual measured data")
plt.xticks(np.arange(-1.4,0.9,0.2))
plt.legend()
plt.title("Residuals from both prediction models")
plt.tight_layout()
plt.show()

###Reduced chi squared calculations
chi2_exp=np.sum( (exp_residual)**2 / amp_unc**2 )
reduced_chi2_exp = chi2_exp/(volt.size - exp_popt.size)

chi2_shock=np.sum( (shockley_residual)**2 / amp_unc**2 )
reduced_chi2_shock = chi2_shock/(volt.size - shock_popt.size)

print("Reduced Chi-Squared Values:")
print("Exponential Model: ", reduced_chi2_exp)
print("Shockley Model: ", reduced_chi2_shock)


###Thermal Voltage Calculations
thermal_volt_lower = 1.0/(shock_popt[0]) #Lower bound (n=1)
thermal_volt_higher = 1.0/(2.0*shock_popt[0]) #Upper bound (n=2)

print("Given that the ideality factor η varies between 1 and 2, "
      "then our thermal voltage varies between: ", thermal_volt_lower*1000.0, 
      "mV and ", thermal_volt_higher*1000.0, "mV")
print("The leakage current is approximately: ", -1.0*amp[0], 
      "mA or approximately: ", -1000.0*amp[0], "μA")