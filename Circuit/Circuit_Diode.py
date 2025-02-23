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


#Loading the dataset
source, volt, amp, volt_unc, amp_unc = np.loadtxt("circuit_diode_data.csv", delimiter = ',', skiprows=2, unpack=True)

forward_volt = np.array([])
forward_amp = np.array([])
forward_volt_unc = np.array([])
forward_amp_unc = np.array([])

for f in range(len(volt)):
    if volt[f] >=0.1:
        forward_volt = np.append(forward_volt, volt[f])
        forward_amp = np.append(forward_amp, amp[f])
        forward_volt_unc = np.append(forward_volt_unc, volt_unc[f])
        forward_amp_unc = np.append(forward_amp_unc, amp_unc[f])


#Prediction Models to be fitted
def linear_log_model(x_val, a, b):
    return b*np.log((x_val+a))+np.log(-1*amp[0]) #Returns the log of the current
    # return b*np.log((x_val+a))+np.log(0.00001) #Returns the log of the current


def exponential_model(x_val, a, b, c, d):
    return a*(b**(c*x_val+d))

# def shockley_model(x_val, a, b, c):
#     return a*(np.exp(b*x_val)-c)

def shockley_model(x_val, a):
    return -1*amp[0]*(np.exp(x_val*a)-1.0)
    # return 0.00001*(np.exp(x_val*a)-1.0)




# exp_popt, exp_pcov = curve_fit(exponential_model, forward_volt, forward_amp, p0 = (0.6353310698, 1031.75089, 3.723623702, -2.268601667), sigma=forward_amp_unc, absolute_sigma = True, maxfev = 10000)
exp_popt, exp_pcov = curve_fit(exponential_model, volt, amp, p0 = (0.6353310698, 1031.75089, 3.723623702, -2.268601667), sigma=amp_unc, absolute_sigma = True, maxfev = 10000)

# shock_popt, shock_pcov = curve_fit(shockley_model, volt, amp, p0 = (0.6353310698, 25.0, 0.1), sigma=amp_unc, absolute_sigma = True, maxfev = 10000)
# shock_popt, shock_pcov = curve_fit(shockley_model, volt, amp, p0=(1.05017e-06, 1.0/22.823, 1.5), sigma=amp_unc, absolute_sigma = True, maxfev = 10000)
shock_popt, shock_pcov = curve_fit(shockley_model, volt, amp, p0=(20.473), sigma=amp_unc, absolute_sigma = True, maxfev = 100000)
# shock_popt, shock_pcov = curve_fit(shockley_model, volt, amp, p0=(50.473), sigma=amp_unc, absolute_sigma = True)



# pow_popt, pow_pcov = curve_fit(power_model, volt, amp, sigma = amp_unc, absolute_sigma = True)
log_amp_unc = forward_amp_unc /(forward_amp)#*np.log(10))
log_volt_unc = forward_volt_unc/(forward_volt*np.log(10))
# log_popt, log_pcov = curve_fit(linear_log_model, forward_volt, np.log(forward_amp), p0=(0.7, 40.0), sigma = log_amp_unc, absolute_sigma = True)
log_popt, log_pcov = curve_fit(linear_log_model, forward_volt, np.log(forward_amp), p0=(0.7, 40.0), sigma = log_amp_unc, absolute_sigma = True, maxfev=1000000)


# plt.errorbar(forward_volt, forward_amp, xerr=log_volt_unc, yerr=log_amp_unc, fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)

#plt.plot(forward_volt, exponential_model(forward_volt, log_popt[0], log_popt[1]), label="Positive Log Model Fitting", color="blue", linewidth = 1)
#plt.plot(forward_volt, np.exp(linear_log_model(forward_volt, log_popt[0], log_popt[1])), label="Positive Log Model Fitting", color="blue", linewidth = 1)

plt.errorbar(volt, amp, xerr=volt_unc, yerr=amp_unc, fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)
#exp_volt=np.arange(-1.5, 0.9, 0.05)
plt.plot(volt, exponential_model(volt, exp_popt[0], exp_popt[1], exp_popt[2], exp_popt[3]), label="Positive Exp Model Fitting", color="blue", linewidth = 1)
#plt.xticks(np.arange(-1.5,1.0, step=0.25))


plt.plot(forward_volt, np.exp(linear_log_model(forward_volt, log_popt[0], log_popt[1])), label="Positive Log Model Fitting", color="orange", linewidth = 1)



# plt.plot(exp_volt, shockley_model(exp_volt, shock_popt[0], shock_popt[1], shock_popt[2]), label="Shockley Model Fitting", color="green", linewidth = 1)
plt.plot(volt, shockley_model(volt, shock_popt[0]), label="Shockley Model Fitting", color="green", linewidth = 1)
plt.legend()
plt.show()


# plt.errorbar(forward_volt, forward_amp, xerr=log_volt_unc, yerr=log_amp_unc, fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)
# plt.errorbar(volt, amp, xerr=volt_unc, yerr=amp_unc, fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)
# plt.errorbar(volt, amp, xerr=volt_unc, yerr=amp_unc, fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)

# # plt.plot(forward_volt, np.exp(linear_log_model(forward_volt, log_popt[0], log_popt[1])), label="Positive Log Model Fitting", color="blue", linewidth = 1)

# plt.plot(np.log(volt), np.log(amp))

# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.show()

exp_model_data = exponential_model(volt, exp_popt[0], exp_popt[1], exp_popt[2], exp_popt[3])
shockley_model_data = shockley_model(volt, shock_popt[0])


log_model_data = linear_log_model(forward_volt, log_popt[0], log_popt[1])


plt.plot(volt, exp_model_data)


#a*(b**(c*x_val+d))
#

chi2_exp=np.sum( (amp - exp_model_data)**2 / amp_unc**2 )
reduced_chi2_exp = chi2_exp/(volt.size - exp_popt.size)


chi2_shock=np.sum( (amp - shockley_model_data)**2 / amp_unc**2 )
reduced_chi2_shock = chi2_shock/(volt.size - shock_popt.size)






# b*np.log((x_val+a))+np.log(0.00001)

log_unc_1 = np.sqrt(forward_volt_unc**2+log_pcov[0][0]) #x_val+a
log_unc_2 = np.abs(log_unc_1/(forward_volt+log_popt[0])) #log
log_unc_3 = log_popt[1]*np.log(forward_volt + log_popt[0])*np.sqrt((np.sqrt(log_pcov[1][1])/log_popt[1])**2+(log_unc_2/(forward_volt+log_popt[0]))**2)
log_unc_4 = np.sqrt(log_unc_3**2+amp_unc[0])

dlogI_dV = log_popt[1] / (forward_volt + log_popt[0])  # ∂(log I) / ∂V
dlogI_dI = 1 / -1*amp[0]  # ∂(log I) / ∂I

#log_model_unc = np.sqrt((dlogI_dV * forward_volt_unc) ** 2 + (dlogI_dI * forward_amp_unc) ** 2)
# chi2_log=np.sum( (np.log(forward_amp) - log_model_data)**2 / forward_volt_unc**2 )
chi2_log=np.sum( (np.log(forward_amp) - log_model_data)**2 / (log_unc_4)**2 )

reduced_chi2_log = chi2_log/(forward_volt.size - log_popt.size)

# residual_exp = (amp-exponential_model(volt, exp_popt[0], exp_popt[1], exp_popt[2], exp_popt[3]))
# residual_shock = (amp-shockley_model(volt, shock_popt[0]))
# residual_log = (forward_amp - np.exp(linear_log_model(forward_volt, log_popt[0], log_popt[1])))

# chi2_exp = (1.0/(len(amp)-len(exp_popt)))*np.sum((residual_exp/volt_unc)**2)
# chi2_shock = (1.0/(len(amp)-len(shock_popt)))*np.sum((residual_shock/volt_unc)**2)
# chi2_log = (1.0/(len(amp)-len(log_popt)))*np.sum((residual_log/forward_volt_unc)**2)

print("Reduced Chi-Squared Values:")
print("Log Model: ", reduced_chi2_log)
print("Exponential Model: ", reduced_chi2_exp)
print("Shockley Model: ", reduced_chi2_shock)



# ### LOG MODEL ERROR PROPAGATION ###
# dlogI_dV = log_popt[1] / (forward_volt + log_popt[0])  # ∂(log I) / ∂V
# dlogI_dI = 1 / -1*amp[0]  # ∂(log I) / ∂I

# log_model_unc = np.sqrt((dlogI_dV * forward_volt_unc) ** 2 + (dlogI_dI * forward_amp_unc) ** 2)

# ### EXPONENTIAL MODEL ERROR PROPAGATION ###
# c, b = exp_popt[2], exp_popt[1]  # Extract fitted parameters
# exp_model = exponential_model(volt, *exp_popt)  # Compute model values
# dexp_dV = exp_model * c * np.log(b)  # ∂I/∂V
# dexp_dI = np.ones_like(amp)  # ∂I/∂I

# exp_model_unc = np.sqrt((dexp_dV * volt_unc) ** 2 + (dexp_dI * amp_unc) ** 2)

# ### SHOCKLEY MODEL ERROR PROPAGATION ###
# a = shock_popt[0]  # Extract fitted parameter
# shock_model = shockley_model(volt, a)  # Compute model values
# dshock_dV = -amp[0] * a * np.exp(a * volt)  # ∂I/∂V
# dshock_dI = np.ones_like(amp)  # ∂I/∂I

# shock_model_unc = np.sqrt((dshock_dV * volt_unc) ** 2 + (dshock_dI * amp_unc) ** 2)

# ### PRINT RESULTS ###
# print("Log Model Uncertainty:", log_model_unc)
# print("Exponential Model Uncertainty:", exp_model_unc)
# print("Shockley Model Uncertainty:", shock_model_unc)


# # Number of data points
# N = len(amp)
# N_log = len(forward_amp)
# # Number of parameters for each model
# p_log = len(log_popt)
# p_exp = len(exp_popt)
# p_shock = len(shock_popt)

# # Degrees of freedom
# dof_log = N_log - p_log
# dof_exp = N - p_exp
# dof_shock = N - p_shock

# # Compute residuals (difference between observed and model values)
# log_residuals = (forward_amp - np.exp(linear_log_model(forward_volt, *log_popt))) / log_model_unc
# exp_residuals = (amp - exponential_model(volt, *exp_popt)) / exp_model_unc
# shock_residuals = (amp - shockley_model(volt, *shock_popt)) / shock_model_unc

# # Compute chi-squared values
# chi2_log = np.sum(log_residuals**2) / dof_log
# chi2_exp = np.sum(exp_residuals**2) / dof_exp
# chi2_shock = np.sum(shock_residuals**2) / dof_shock

# # Print results
# print("Reduced Chi-Squared Values:")
# print("Log Model: ", chi2_log)
# print("Exponential Model: ", chi2_exp)
# print("Shockley Model: ", chi2_shock)

