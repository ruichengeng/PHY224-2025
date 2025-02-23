# -*- coding: utf-8 -*-
"""

"""
#Necessary modules
import numpy as np #Useful helper functions that can convert our data into formats acceptable to the curve fitting components.
from scipy.optimize import curve_fit #The main component of this fitting exercise
import matplotlib.pyplot as plt #Handles the plotting aspect of this lab

def show_plot(plot='usb'):
    if plot == 'usb':
        #We split the data into 3 arrays: the time, the total voltage across 
        #the circuit, the, the voltage across the resister, voltage across the 
        #capacitor
        time, total_volt, res_volt, cap_volt, res_unc = np.loadtxt(
            "scope_4.csv", delimiter = ',', skiprows=2, unpack=True)
        time = time*1000000

        zero_pos = len(time)//2
        charge_time = time[:zero_pos]
        discharge_time = time[zero_pos:]

        charge_res_volt = res_volt[:zero_pos]
        discharge_res_volt = res_volt[zero_pos:]
        def charge_volt_model(x_val, R, C):
            return charge_res_volt[8]*np.exp(1*(x_val-charge_time[8])/(R*C))

        def discharge_volt_model(x_val, R, C):
            return discharge_res_volt[8]*np.exp((-1*x_val)/(R*C))

        charge_popt, charge_pcov = curve_fit(charge_volt_model, charge_time, 
                                             charge_res_volt, 
                                             absolute_sigma = True)
        discharge_popt, discharge_pcov = curve_fit(discharge_volt_model, 
                                                   discharge_time, 
                                                   discharge_res_volt, 
                                                   absolute_sigma = True)

        c_res,c_cap = charge_popt[0], charge_popt[1]
        d_res,d_cap = discharge_popt[0], discharge_popt[1]
        c_pvar = np.diag(charge_pcov)
        
        ###Plotting
        main_graph = plt.subplot(5, 1, (1,3))
        plt.errorbar(charge_time, charge_res_volt, yerr=res_unc[:zero_pos], 
                     fmt="o", color = "black", markersize=0.1, label='')
        plt.errorbar(discharge_time, discharge_res_volt, yerr=res_unc[zero_pos:], 
                     fmt="o", color = "black", markersize=0.1,)
        plt.plot(charge_time, 
                 charge_volt_model(charge_time, c_res, c_cap), color = "red")
        plt.plot(discharge_time, discharge_volt_model(discharge_time, d_res, 
                                                      d_cap), color = "blue")
        plt.xlabel(r'Time ($\mu$s)')
        plt.ylabel("Voltage across resistor (V)")
        plt.legend()
        plt.grid()
        
        ###Array of the difference between mean and data for residual
        residuals = []
        model_avg = []
        for i in range(len(charge_time - 1)):
            model_avg.append(charge_volt_model(charge_time[i], c_res, c_cap))
            difference = charge_res_volt[i] - charge_volt_model(charge_time[i], 
                                                                c_res, c_cap)
            residuals.append(difference)
            
        for i in range(len(discharge_time - 1)):
            model_avg.append(discharge_volt_model(discharge_time[i], 
                                                  d_res, d_cap))
            difference = (discharge_res_volt[i] - 
                          discharge_volt_model(discharge_time[i], d_res, d_cap))
            residuals.append(difference)
        
        ###Plot of the residuals
        linear_residual = plt.subplot(5,1,5)
        plt.title("Residuals")
        plt.axhline(y = 0, linewidth=1, color = 'r', linestyle = '-')
        plt.errorbar(time, residuals, yerr=0.001, xerr= 0.000000001, fmt='o', 
                     markersize=0.5, color='black', ecolor='black', 
                     linewidth=0.5)
        plt.xlabel(r'Time ($\mu$s)')
        plt.ylabel("Error (V)")
        # plt.yticks(np.arange(-0.5,0.51,step=0.5))
        plt.show()
        
    elif plot =='manual':
        #We manually measured data points for the voltage across the resister 
        #in a seperate csv file, which we will curve fit for each charging and 
        #discharging portion of the curve
        charge_time, charge_volt, discharge_time, discharge_volt = np.loadtxt(
            "circuit_pt2_data.csv", delimiter = ',', skiprows=2, unpack=True)
        charge_volt = charge_volt/1000
        discharge_volt = discharge_volt/1000

        
        def charge_volt_model(x_val, R, C):
            return charge_volt[0]*np.exp((x_val-charge_time[0])/(R*C))

        def discharge_volt_model(x_val, R, C):
            return discharge_volt[0]*np.exp((-1*x_val)/(R*C))

        charge_popt, charge_pcov = curve_fit(charge_volt_model, charge_time, 
                                             charge_volt, absolute_sigma = True)
        discharge_popt, discharge_pcov = curve_fit(
            discharge_volt_model, discharge_time, discharge_volt, 
            absolute_sigma = True)

        c_res,c_cap = charge_popt[0], charge_popt[1]
        d_res,d_cap = discharge_popt[0], discharge_popt[1]
        c_pvar = np.diag(charge_pcov)
        
        main_graph = plt.subplot(5, 1, (1,3))
        plt.plot(charge_time, charge_volt_model(charge_time, c_res, c_cap), 
                 color = "red")
        plt.plot(discharge_time, discharge_volt_model(discharge_time, d_res, 
                                                      d_cap), color = "blue")
        plt.errorbar(charge_time, charge_volt, yerr=0.1, color = "black",fmt='o')
        plt.errorbar(discharge_time, discharge_volt, yerr=0.1, 
                     color = "black",fmt='o')
        plt.xlabel(r'Time ($\mu$s)')
        plt.ylabel("Voltage across resistor (V)")
        plt.legend()
        plt.grid()
        
        ###Array of the difference between mean and data for residual
        residuals = []
        model_avg = []
        time = []
        for i in range(len(charge_time - 1)):
            model_avg.append(charge_volt_model(charge_time[i], c_res, c_cap))
            difference = charge_volt[i] - charge_volt_model(charge_time[i], 
                                                            c_res, c_cap)
            residuals.append(difference)
            time.append(charge_time[i])
            
        for i in range(len(discharge_time - 1)):
            model_avg.append(discharge_volt_model(discharge_time[i], 
                                                  d_res, d_cap))
            difference = discharge_volt[i] - discharge_volt_model(
                discharge_time[i], d_res, d_cap)
            residuals.append(difference)
            time.append(discharge_time[i])
        
        ###Plot of the residuals
        linear_residual = plt.subplot(5,1,5)
        plt.title("Residuals")
        plt.axhline(y = 0, linewidth=1, color = 'r', linestyle = '-')
        plt.errorbar(time, residuals, yerr=0.001, xerr= 0.000000001, fmt='o', 
                     markersize=1, color='black', 
                      ecolor='black', linewidth=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Error (V)")
        plt.yticks(np.arange(-0.025,0.026,step=0.025))
        plt.show()
    
    else:
        print('invalid plot')