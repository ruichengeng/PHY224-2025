# -*- coding: utf-8 -*-
"""
IMPORTANT NOTE: To see the plots, use the show_plot() function. See text below
for more details
"""
#Necessary modules
import numpy as np #Useful helper functions that can convert our data into formats acceptable to the curve fitting components.
from scipy.optimize import curve_fit #The main component of this fitting exercise
import matplotlib.pyplot as plt #Handles the plotting aspect of this lab

print('To view the plots of the raw data from the oscilloscope, the oscilloscope'
      ' data with analysis, and the manual data and analysis, please use the'
      ' show_plot() function with an input of "raw", "usb", or "manual" '
      'respectively')
#We define a function show_plot which takes as input whichever dataset to display 
#the curve fitting for
def show_plot(plot='usb'):
    if plot == 'raw':
        #This plot will display only the raw data from the oscilloscope. 
        #Analysis will be done with the plot=='usb' input
        data = np.loadtxt(
            "scope_4 w unc.csv", delimiter = ',', skiprows=2, unpack=True)
        time, total_volt, res_volt, cap_volt = [data[i] for i in (0,1,2,3)]
        time = time*1000000 #adjusting unit magnitude
        
        #Due to the raw data having a y offset, we will find the minimum point
        #of the capacitance voltage is the y offset from v=0, which will align
        #with the resistance curve
        min_pos = np.where(cap_volt == np.min(cap_volt))[0][0]
        min_val = abs(cap_volt[min_pos])
        
        #Plotting the curve
        #Note we have calculated the y error by referring to oscilloscope 
        #specification and found due to similar range of values that it is a 
        #constant value of 0.13
        plt.title('Voltage against time for all three components(data from oscilloscope')
        plt.errorbar(time, total_volt + min_val, yerr= 0.13, fmt="o", color = "red", 
                 linestyle="None", 
                 markersize=0.3, 
                 label='voltage across circuit')
        plt.errorbar(time, cap_volt + min_val, yerr= 0.13,fmt="o", color = "blue", 
                 linestyle="None", 
                 markersize=0.3, 
                 label='voltage across capacitor')
        plt.errorbar(time, res_volt, yerr=0.13, fmt="o", color = "orange", 
                 linestyle="None", 
                 markersize=0.3, 
                 label='voltage across resistor')
        plt.xlabel(r'Time($\mu$s)')
        plt.xticks(np.arange(-50,51,step=10))
        plt.ylabel("Voltage(V)")
        plt.yticks(np.arange(-1.5,1.51,step=0.5))
        plt.legend()
        plt.grid()
        plt.show()
        
    elif plot == 'usb':
        #We import the data from the oscilloscope and split it into 4 arrays: 
        #the time, the total voltage across the circuit, the voltage across the 
        #resistor, the voltage across the capacitor. Also one additional array
        #for the uncertainty of the voltage we're analysing(resistor)
        #Note that we have removed outliers from the dataset
        time, total_volt, res_volt, cap_volt, res_unc = np.loadtxt(
            "scope_4 w unc clean.csv", delimiter = ',', skiprows=2, unpack=True)
        time = time*1000000 #adjusting unit magnitude
        
        #Splitting the data into two sections: one for the voltage across the 
        #resistor as the capacitor charges, and one for the voltage as the
        #capacitor discharges
        zero_pos = len(time)//2 #Position of when the curve shape changes
        
        charge_time = time[:zero_pos] 
        discharge_time = time[zero_pos:]
        
        charge_res_volt = res_volt[:zero_pos]
        discharge_res_volt = res_volt[zero_pos:]
        
        
        #Functions to fit. We need seperate curve fitting functions due to
        #different curve shapes of the charging and discharging portion. Note
        #the index for the array is from trial and error: the best fit is found
        #on setting the index as the 8th element in the voltage data
        def charge_volt_model(x_val, R, C):
            return charge_res_volt[8]*np.exp(-1*(x_val-charge_time[8])/(R*C))

        def discharge_volt_model(x_val, R, C):
            return discharge_res_volt[8]*np.exp((-1*x_val)/(R*C))

        charge_popt, charge_pcov = curve_fit(charge_volt_model, charge_time, 
                                             charge_res_volt, 
                                              sigma= res_unc[:zero_pos],
                                             absolute_sigma = True)
        discharge_popt, discharge_pcov = curve_fit(discharge_volt_model, 
                                                   discharge_time, 
                                                   discharge_res_volt,
                                                    sigma=res_unc[zero_pos:],
                                                   absolute_sigma = True)
        
        #Assigning variable names to charging and discharging models' popt for 
        #code legibility
        c_res,c_cap = charge_popt[0], charge_popt[1]
        c_pvar = np.diag(charge_pcov)
        d_res,d_cap = discharge_popt[0], discharge_popt[1]
        d_pvar = np.diag(discharge_pcov)
        
        ###Plotting
        main_graph = plt.subplot(5, 1, (1,3))
        plt.title('Voltage across resistor against time(data from oscilloscope)')
        plt.errorbar(charge_time, charge_res_volt, yerr=res_unc[:zero_pos], 
                     fmt="o", color = "black", markersize=0.1, label='data point')
        plt.errorbar(discharge_time, discharge_res_volt, yerr=res_unc[zero_pos:], 
                     fmt="o", color = "black", markersize=0.1)
        plt.plot(charge_time, 
                 charge_volt_model(charge_time, c_res, c_cap), color = "red", 
                 label='fitted: charge')
        plt.plot(discharge_time, discharge_volt_model(discharge_time, d_res, 
                                                      d_cap), color = "blue", 
                 label='fitted: discharge')
        plt.xlabel(r'Time($\mu$s)')
        plt.xticks(np.arange(-50,51,step=10))
        plt.ylabel("Voltage(V)")
        plt.yticks(np.arange(-1.5,1.51,step=0.5))
        plt.legend()
        plt.grid()
        
        ###Array of the difference between model voltage and data for residual
        residuals = []
        for i in range(len(charge_time)):
            difference = charge_res_volt[i] - charge_volt_model(charge_time[i], 
                                                                c_res, c_cap)
            residuals.append(difference)
            
        for i in range(len(discharge_time)):
            difference = (discharge_res_volt[i] - 
                          discharge_volt_model(discharge_time[i], d_res, d_cap))
            residuals.append(difference)
        
        ###Plot of the residuals
        residual = plt.subplot(5,1,5)
        plt.title("Residuals")
        plt.axhline(y = 0, linewidth=1, color = 'r', linestyle = '-')
        plt.errorbar(time, residuals, yerr=res_unc, fmt='o', 
                     markersize=0.5, color='black', ecolor='black', 
                     linewidth=0.5)
        plt.xlabel(r'Time ($\mu$s)')
        plt.xticks(np.arange(-50,51,step=10))
        plt.ylabel("Error (V)")
        plt.show()
        
        print("Charge R: ",c_res, "+-", np.sqrt(c_pvar[0]))
        print("Charge C: ",c_cap, "+-", np.sqrt(c_pvar[1]))
        print("Discharge R: ",d_res, "+-", np.sqrt(d_pvar[0]))
        print("Discharge C: ",d_cap, "+-", np.sqrt(d_pvar[1]))
        
        ###Calculating the reduced chi squared
        #Degrees of freedom: N - m = len(volt) - 5 = , where m is the # of parameters
        rchi_squared = (1/(len(time) - 2))*np.sum(((residuals)/res_unc)**2)
        print("The reduced chi squared is", rchi_squared)
        
    elif plot =='manual':
        #We manually measured data points for the voltage across the resister 
        #in a seperate csv file, which we will curve fit for each charging and 
        #discharging portion of the curve
        time,volt,time_unc,volt_unc = np.loadtxt("circuit_pt2_data w unc.csv", delimiter = ',', skiprows=2, unpack=True)
        
        #As explained above, we split the data into two seperate portions
        charge_time = time[0:6]
        discharge_time = time[6:12]
        charge_time_unc = time_unc[0:6]
        discharge_time_unc = time_unc[6:12]
        
        charge_volt = volt[0:6]
        discharge_volt = volt[6:12]
        charge_volt_unc = volt_unc[0:6]
        discharge_volt_unc = volt_unc[6:12]
        
        #adjusting unit magnitudes
        charge_volt = charge_volt/1000
        discharge_volt = discharge_volt/1000

        #Same as above, we define 2 model functions to fit the data to due to
        #the different curve shapes
        def charge_volt_model(x_val, R, C):
            return charge_volt[0]*np.exp((x_val-charge_time[0])/(R*C))

        def discharge_volt_model(x_val, R, C):
            return discharge_volt[0]*np.exp((-1*x_val)/(R*C))

        charge_popt, charge_pcov = curve_fit(charge_volt_model, charge_time, 
                                             charge_volt, 
                                             sigma=charge_volt_unc, 
                                             absolute_sigma = True)
        discharge_popt, discharge_pcov = curve_fit(discharge_volt_model, 
                                                   discharge_time, 
                                                   discharge_volt, 
                                                   sigma=discharge_volt_unc,
                                                   absolute_sigma = True)
        
        #Assigning variable names to charging and discharging models' popt for 
        #code legibility
        c_res,c_cap = charge_popt[0], charge_popt[1]
        c_pvar = np.diag(charge_pcov)
        d_res,d_cap = discharge_popt[0], discharge_popt[1]
        d_pvar = np.diag(discharge_pcov)
        
        main_graph = plt.subplot(5, 1, (1,3))
        plt.title('Voltage across resistor against time(data from manual '
                  'measurement)')
        plt.errorbar(charge_time, charge_volt, 
                     yerr=charge_volt_unc, 
                     xerr=charge_time_unc, 
                     color = "black",fmt='o',label='observed')
        plt.errorbar(discharge_time, discharge_volt, 
                     yerr=discharge_volt_unc,
                     xerr=discharge_time_unc,
                     color = "black", fmt='o')
        plt.plot(charge_time, charge_volt_model(charge_time, c_res, c_cap), 
                 color = "red",label='fitted: charge')
        plt.plot(discharge_time, 
                 discharge_volt_model(discharge_time, d_res, d_cap), 
                 color = "blue", label='fitted: discharge')
        plt.xlabel(r'Time($\mu$s)')
        plt.ylabel("Voltage(V)")
        plt.legend()
        plt.grid()
        
        ###Creating arrays for residual plotting and chi squared calculations
        residuals = []
        for i in range(len(charge_time)):
            difference = charge_volt[i] - charge_volt_model(charge_time[i], 
                                                            c_res, c_cap)
            residuals.append(difference)
            
        for i in range(len(discharge_time)):
            difference = discharge_volt[i] - discharge_volt_model(
                discharge_time[i], d_res, d_cap)
            residuals.append(difference)
        
        ###Plot of the residuals
        residual = plt.subplot(5,1,5)
        plt.title("Residuals")
        plt.axhline(y = 0, linewidth=1, color = 'r', linestyle = '-')
        plt.errorbar(time, residuals, 
                      yerr=volt_unc, 
                      xerr=time_unc, 
                      fmt='o', markersize=1, color='black', 
                      ecolor='black', linewidth=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Error (V)")
        plt.show()
        
        print("Charge R: ",c_res, "+-", np.sqrt(c_pvar[0]))
        print("Charge C: ",c_cap, "+-", np.sqrt(c_pvar[1]))
        print("Discharge R: ",d_res, "+-", np.sqrt(d_pvar[0]))
        print("Discharge C: ",d_cap, "+-", np.sqrt(d_pvar[1]))
        
        ###Calculating the reduced chi squared
        #Degrees of freedom: N - m = len(volt) - 2 = , where m is the # of 
        #parameters
        rchi_squared = (1/(len(volt) - 2))*np.sum(((residuals)/volt_unc)**2)
        print("The reduced chi squared is", rchi_squared)
        
    else:
        print('invalid plot')