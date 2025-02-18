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
        
        #Splitting the data into two sections: one for the negative voltage(down)
        #and one for the positive voltage(up).
        zero_pos = len(time)//2 #Position of when the curve shape changes
        
        down_time = time[:zero_pos] 
        up_time = time[zero_pos:]
        
        down_res_volt = res_volt[:zero_pos]
        up_res_volt = res_volt[zero_pos:]
        
        
        #Functions to fit. We need seperate curve fitting functions due to
        #different curve shapes of the charging and discharging portion. Note
        #the index for the array is from trial and error: the best fit is found
        #on setting the index as the 8th element in the voltage data
        def down_volt_model(x_val, R, C):
            return down_res_volt[8]*np.exp(-1*(x_val-down_time[8])/(R*C))

        def up_volt_model(x_val, R, C):
            return up_res_volt[8]*np.exp((-1*x_val)/(R*C))

        down_popt, down_pcov = curve_fit(down_volt_model, down_time, 
                                             down_res_volt, 
                                              sigma= res_unc[:zero_pos],
                                             absolute_sigma = True)
        up_popt, up_pcov = curve_fit(up_volt_model, 
                                                   up_time, 
                                                   up_res_volt,
                                                    sigma=res_unc[zero_pos:],
                                                   absolute_sigma = True)
        
        #Assigning variable names to charging and discharging models' popt for 
        #code legibility
        d_res,d_cap = down_popt[0], down_popt[1]
        d_pvar = np.diag(down_pcov)
        u_res,u_cap = up_popt[0], up_popt[1]
        u_pvar = np.diag(up_pcov)
        
        ###Plotting
        main_graph = plt.subplot(5, 1, (1,3))
        plt.title('Voltage across resistor against time(data from oscilloscope)')
        plt.errorbar(down_time, down_res_volt, yerr=res_unc[:zero_pos], 
                     fmt="o", color = "black", markersize=0.1, label='data point')
        plt.errorbar(up_time, up_res_volt, yerr=res_unc[zero_pos:], 
                     fmt="o", color = "black", markersize=0.1)
        plt.plot(down_time, 
                 down_volt_model(down_time, d_res, d_cap), color = "red", 
                 label='fitted: discharge')
        plt.plot(up_time, up_volt_model(up_time, u_res, 
                                                      u_cap), color = "blue", 
                 label='fitted: charge')
        plt.xlabel(r'Time($\mu$s)')
        plt.xticks(np.arange(-50,51,step=10))
        plt.ylabel("Voltage(V)")
        plt.yticks(np.arange(-1.5,1.51,step=0.5))
        plt.legend()
        plt.grid()
        
        ###Array of the difference between model voltage and data for residual
        residuals = []
        for i in range(len(down_time)):
            difference = down_res_volt[i] - down_volt_model(down_time[i], 
                                                                d_res, d_cap)
            residuals.append(difference)
            
        for i in range(len(up_time)):
            difference = (up_res_volt[i] - 
                          up_volt_model(up_time[i], u_res, u_cap))
            residuals.append(difference)
        
        ###Plot of the residuals
        residual = plt.subplot(5,1,5)
        plt.title("Residuals")
        plt.axhline(y = 0, linewidth=1, color = 'r', linestyle = '-')
        plt.errorbar(time, residuals, yerr=res_unc, fmt='o', 
                     markersize=0.5, color='black', ecolor='black', 
                     linewidth=0.5)
        plt.xlabel(r'Time ($\mu$s)')
        plt.xticks(np.arange(-50,50,step=10))
        plt.ylabel("Error (V)")
        plt.show()
        
        print("down R: ",d_res, "+-", np.sqrt(d_pvar[0]))
        print("down C: ",d_cap, "+-", np.sqrt(d_pvar[1]))
        print("up R: ",u_res, "+-", np.sqrt(u_pvar[0]))
        print("up C: ",u_cap, "+-", np.sqrt(u_pvar[1]))
        
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
        down_time = time[0:6]
        up_time = time[6:12]
        down_time_unc = time_unc[0:6]
        up_time_unc = time_unc[6:12]
        
        down_volt = volt[0:6]
        up_volt = volt[6:12]
        down_volt_unc = volt_unc[0:6]
        up_volt_unc = volt_unc[6:12]
        
        #adjusting unit magnitudes
        down_volt = down_volt/1000
        up_volt = up_volt/1000

        #Same as above, we define 2 model functions to fit the data to due to
        #the different curve shapes
        def down_volt_model(x_val, R, C):
            return down_volt[0]*np.exp((x_val-down_time[0])/(R*C))

        def up_volt_model(x_val, R, C):
            return up_volt[0]*np.exp((-1*x_val)/(R*C))

        down_popt, down_pcov = curve_fit(down_volt_model, down_time, 
                                             down_volt, 
                                             sigma=down_volt_unc, 
                                             absolute_sigma = True)
        up_popt, up_pcov = curve_fit(up_volt_model, 
                                                   up_time, 
                                                   up_volt, 
                                                   sigma=up_volt_unc,
                                                   absolute_sigma = True)
        
        #Assigning variable names to charging and discharging models' popt for 
        #code legibility
        d_res,d_cap = down_popt[0], down_popt[1]
        d_pvar = np.diag(down_pcov)
        u_res,u_cap = up_popt[0], up_popt[1]
        u_pvar = np.diag(up_pcov)
        
        main_graph = plt.subplot(5, 1, (1,3))
        plt.title('Voltage across resistor against time(data from manual '
                  'measurement)')
        plt.errorbar(down_time, down_volt, 
                     yerr=down_volt_unc, 
                     xerr=down_time_unc, 
                     color = "black",fmt='o',label='observed')
        plt.errorbar(up_time, up_volt, 
                     yerr=up_volt_unc,
                     xerr=up_time_unc,
                     color = "black", fmt='o')
        plt.plot(down_time, down_volt_model(down_time, d_res, d_cap), 
                 color = "red",label='fitted: discharge')
        plt.plot(up_time, 
                 up_volt_model(up_time, u_res, u_cap), 
                 color = "blue", label='fitted: charge')
        plt.xlabel(r'Time($\mu$s)')
        plt.xticks(np.arange(-50,51,step=10))
        plt.ylabel("Voltage(V)")
        plt.legend()
        plt.grid()
        
        ###Creating arrays for residual plotting and chi squared calculations
        residuals = []
        for i in range(len(down_time)):
            difference = down_volt[i] - down_volt_model(down_time[i], 
                                                            d_res, d_cap)
            residuals.append(difference)
            
        for i in range(len(up_time)):
            difference = up_volt[i] - up_volt_model(
                up_time[i], u_res, u_cap)
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
        plt.xticks(np.arange(-50,50,step=10))
        plt.ylabel("Error (V)")
        plt.show()
        
        print("down R: ",d_res, "+-", np.sqrt(d_pvar[0]))
        print("down C: ",d_cap, "+-", np.sqrt(d_pvar[1]))
        print("up R: ",u_res, "+-", np.sqrt(u_pvar[0]))
        print("up C: ",u_cap, "+-", np.sqrt(u_pvar[1]))
        
        ###Calculating the reduced chi squared
        #Degrees of freedom: N - m = len(volt) - 2 = , where m is the # of 
        #parameters
        rchi_squared = (1/(len(volt) - 2))*np.sum(((residuals)/volt_unc)**2)
        print("The reduced chi squared is", rchi_squared)
        
    else:
        print('invalid plot')