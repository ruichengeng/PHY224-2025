"""
Lightbulb Exercise 2
NOTE: To see the plot of the raw data, please use the show_raw() function
"""
#Importing critical models
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Setting resolution of image and importing raw data
plt.figure(dpi=200)
# Define figure layout: 3 rows, 1 columns
fig, axs = plt.subplots(3, 2, figsize=(15, 17), gridspec_kw={'height_ratios': [2.5, 2.5, 2]})
data = np.loadtxt("Lightbulb_data_clean.csv", delimiter = ",", 
                             skiprows=1, unpack= True)

###NOTE: Since the data takes 2 different shapes depending on the positive and
###negative voltage, causing the curve_fit to fail every time, we split the 
###data into two portions and curve fit them individually
voltage,current,v_unc, c_unc = (data[1][11:],
                                data[2][11:],
                                data[3][11:],
                                data[4][11:]) #Positive portion of data
negvoltage,negcurrent,negv_unc, negc_unc = (abs(data[1][:11]),
                                            abs(data[2][:11]),
                                            abs(data[3][:11]),
                                            abs(data[4][:11])) #negative portion of data
#Defining the models for the curve fit functions
def power_model(V, a, b):
    return a*(V)**b
def log_model(V, a, b):
    return b*np.log(V) + np.log(a)
def ideal_model(V, a):
    return a*((V)**(3/5))

#Defining functions input_power_model,... which calculates the parameters and 
#model fit, the uncertainties, the residuals and reduced chi-squared. Done for
#both negative and positive portions. This is done for code legibility and
#simplicity
def input_power_model(voltage=voltage,current=current,v_unc=v_unc,c_unc=c_unc, 
                      fit=False, negative=False):
    ###Curve fitting
    if negative:
        voltage,current,v_unc, c_unc = negvoltage,negcurrent,negv_unc,negc_unc
        
    popt, pcov = curve_fit(power_model, voltage, current, absolute_sigma= True, 
                           p0=(8.5,3/5))
    a,b = popt[0], popt[1]
    
    ###Calculating the uncertainty of all variables
    #Note pvar is a diagonalised array of the variance of the parameters of the
    #model, to get the value of uncertainty of the parameters, we sqrt the variance.
    model_current = power_model(voltage, a, b)
    pstd = np.sqrt(np.diag(pcov))
    a_unc, b_unc = pstd
    
    #Uncertainty in the dataset
    data_unc = np.sqrt(v_unc**2 + c_unc**2)

    #Model uncertainty propagation
    model_unc = np.sqrt(((voltage**b)*a_unc)**2 + 
                        ((a*voltage**b)*np.log(voltage)*b_unc)**2 + 
                        (a*b*voltage**(b-1)*v_unc)**2)
    
    #Total uncertainty per data point
    total_unc = np.sqrt(data_unc**2 + model_unc**2)
    
    #Calculating reduced chi-squared
    residuals = current - model_current
    dof = len(current) - len(popt)  # Degrees of freedom (N - m)
    rchi_squared = np.sum((residuals/total_unc)** 2)/dof
    
    #Finding the uncertainty in the y val for the residual plot
    residuals_yerr = np.sqrt(model_unc**2 + c_unc**2)
    
    if not fit:
        return [power_model(voltage,a,b),a,b,pstd]
    elif fit:
        return [residuals,rchi_squared,residuals_yerr]
    
def input_log_model(voltage=voltage,current=current,v_unc=v_unc,c_unc=c_unc, 
                    fit=False, negative=False):
    if negative:
        voltage,current,v_unc, c_unc = negvoltage,negcurrent,negv_unc,negc_unc
    
    ###Curve fitting
    popt, pcov = curve_fit(log_model, voltage, np.log(current), sigma=np.log(c_unc), absolute_sigma= True,p0=(8,0.6))
    a,b = popt[0], popt[1]
    
    #Converting data uncertainties to log
    log_v_unc = v_unc /(voltage*np.log(10))
    log_c_unc = c_unc /(current*np.log(10))
    
    ###Calculating the uncertainty of all variables
    model_current = log_model(voltage, a, b)
    pstd = np.sqrt(np.diag(pcov))
    a_unc, b_unc = pstd
    
    #Uncertainty in the dataset
    data_unc = np.sqrt(v_unc**2 + a_unc**2)

    #Model uncertainty propagation
    model_unc = np.sqrt((np.log(voltage) * a_unc)**2 + b_unc**2)
    
    #Total uncertainty per data point
    total_unc = np.sqrt(data_unc**2 + model_unc**2)
    
    #Calculating reduced chi-squared
    residuals = np.log(current) - model_current
    dof = len(current) - len(popt)  #degrees of freedom (N - m)
    rchi_squared = np.sum((residuals/total_unc)** 2)/dof
    
    #Finding the uncertainty in the y val for the residual plot
    residuals_yerr = np.sqrt(model_unc**2 + a_unc**2)
    
    if not fit:
        return [log_model(voltage,a,b),a,b,log_v_unc,log_c_unc]
    elif fit:
        return [residuals,rchi_squared,residuals_yerr]

def input_ideal_model(voltage=voltage,current=current,v_unc=v_unc,c_unc=c_unc, 
                      fit=False, negative=False):
    ###Curve fitting
    if negative:
        voltage,current,v_unc, c_unc = negvoltage,negcurrent,negv_unc,negc_unc
        
    popt, pcov = curve_fit(ideal_model, voltage, current, absolute_sigma= True, p0=(8.5))
    a = popt[0]
    
    ###Calculating the uncertainty of all variables
    #Note pvar is a diagonalised array of the variance of the parameters of the
    #model, to get the value of uncertainty of the parameters, we sqrt the variance.
    model_current = ideal_model(voltage, a)
    pstd = np.sqrt(np.diag(pcov))
    a_unc= pstd
    
    #Uncertainty in the dataset
    data_unc = np.sqrt(v_unc**2 + c_unc**2)

    #Model uncertainty propagation
    model_unc = voltage**(3/5)*a_unc
    
    #Total uncertainty per data point
    total_unc = np.sqrt(data_unc**2 + model_unc**2)
    
    #Calculating reduced chi-squared
    residuals = current - model_current
    dof = len(current) - len(popt)  # Degrees of freedom (N - m)
    rchi_squared = np.sum((residuals/total_unc)** 2)/dof
    
    #Finding the uncertainty in the y val for the residual plot
    residuals_yerr = np.sqrt(model_unc**2 + c_unc**2)
    
    if not fit:
        return [ideal_model(voltage,a),a,pstd]
    elif fit:
        return [residuals,rchi_squared,residuals_yerr]

################## POSITIVE SECTION OF GRAPH #################################
##Power graph
axs[0,0].set_title("Voltage current plot (positive section)")
axs[0,0].plot(voltage, input_power_model()[0], 
          linewidth=1, label="fitted: power") #Power model plot

axs[0,0].plot(voltage, input_ideal_model()[0], 
          linewidth=1, label="fitted: ideal") #Ideal model plot

axs[0,0].errorbar(voltage, current, xerr=v_unc, yerr=c_unc, 
                  fmt='o', markersize=5, 
                  color='black', ecolor='black', 
                  linewidth=0.5, label= "observed") #Data points
axs[0,0].set_xlabel("Voltage(V)")
axs[0,0].set_xticks(np.arange(0,13.5, step=1))
axs[0,0].set_ylabel("Current(mA)")
axs[0,0].legend()
axs[0,0].grid()

#Log graph
log_v_unc = input_log_model()[3] #Getting the converted uncertainties from 
                                 #the log function
log_c_unc = input_log_model()[4]
axs[1,0].set_title("Voltage current plot(log)")

#Need y value to be raised by e for a linear fit
axs[1,0].plot(voltage, np.exp(input_log_model()[0]), 
          linewidth=1, label="fitted:log",color='green') #Log model plot

axs[1,0].errorbar(voltage, current, xerr=log_v_unc, yerr=log_c_unc, 
                  fmt='o', markersize=5, 
                  color='black', ecolor='black', 
                  linewidth=0.5, label= "observed") #Data points
axs[1,0].set_xlabel("Voltage(V)")
axs[1,0].set_xscale('log')
axs[1,0].set_ylabel("Current(mA)")
axs[1,0].set_yscale('log')
axs[1,0].legend()
axs[1,0].grid()

###Plot of the residuals
#Retrieving y value uncertainty from respective model functions
perror_y = input_power_model(fit=True)[2]
ierror_y = input_ideal_model(fit=True)[2]
lerror_y = input_log_model(fit=True)[2]

axs[2,0].set_title("Residuals")
axs[2,0].axhline(y = 0, linewidth=1, color = 'r', linestyle = '-')

axs[2,0].errorbar(voltage, input_power_model(fit=True)[0], 
                  xerr = v_unc, yerr=perror_y, label='power',
                  fmt='o', markersize=3, color='blue', 
                  ecolor='blue', linewidth=0.5) #Power residuals

axs[2,0].errorbar(voltage, input_ideal_model(fit=True)[0], 
                  xerr = v_unc, yerr=ierror_y, label='ideal',
                  fmt='o', markersize=3, color='orange', 
                  ecolor='orange', linewidth=0.5) #Ideal residuals

axs[2,0].errorbar(voltage, input_log_model(fit=True)[0], 
                  xerr = log_v_unc, yerr=log_c_unc, label='log',
                  fmt='o', markersize=3, color='green', 
                  ecolor='green', linewidth=0.5) #Log residuals
axs[2,0].legend(loc='lower left')
axs[2,0].set_xlabel("Voltage")
axs[2,0].set_ylabel("Error (mA)")

print("POWER MODEL(positive):")
print("The reduced chi squared in the power model is", input_power_model(fit=True)[1])
power_std = input_power_model()[3] #Getting parameter uncertainties
print("a = ", input_power_model()[1], "+-", power_std[0])
print("b = ", input_power_model()[2], "+-", power_std[1])

print("IDEAL MODEL(positive):")
print("The reduced chi squared in the ideal model is", input_ideal_model(fit=True)[1])
ideal_std = input_ideal_model()[2]
print("a = ", input_power_model()[1], "+-", ideal_std[0])

print("LOG MODEL(positive):")
print("The reduced chi squared in the log model is", input_log_model(fit=True)[1])
log_std = input_log_model()[3]
print("a = ", input_log_model()[1], "+-", log_std[0])
print("b = ", input_log_model()[2], "+-", log_std[1])


################## NEGATIVE SECTION OF GRAPH #################################

##Power graph
axs[0,1].set_title("Voltage current plot (negative section)")
axs[0,1].plot(-1*negvoltage, -1*input_power_model(negative=True)[0], 
          linewidth=1, label="fitted: power")

axs[0,1].plot(-1*negvoltage, -1*input_ideal_model(voltage, negative=True)[0], 
          linewidth=1, label="fitted: ideal")

axs[0,1].errorbar(-1*negvoltage, -1*negcurrent, xerr= negv_unc, yerr=negc_unc, fmt='o', markersize=5, 
              color='black', ecolor='black', linewidth=0.5, label= "observed")
axs[0,1].set_xlabel("Voltage(V)")
axs[0,1].set_ylabel("Current(mA)")
axs[0,1].legend()
axs[0,1].grid()

###Log graph
log_v_unc = input_log_model(negative=True)[3]
log_c_unc = input_log_model(negative=True)[4]
axs[1,1].set_title("Voltage current plot(log)")

axs[1,1].plot(negvoltage, np.exp(input_log_model(negative=True)[0]), 
          linewidth=1, label="fitted:log",color='green')

axs[1,1].errorbar(negvoltage, negcurrent, xerr=log_v_unc, yerr=log_c_unc, fmt='o', markersize=5, 
              color='black', ecolor='black', linewidth=0.5, label= "observed")
axs[1,1].set_xlabel("Voltage(-V)")
axs[1,1].set_xscale('log')
axs[1,1].set_ylabel("Current(-mA)")
axs[1,1].set_yscale('log')
axs[1,1].legend()
axs[1,1].grid()


###Plot of the residuals
#Retrieving y value uncertainties from the model functions
perror_y = input_power_model(fit=True,negative=True)[2]
ierror_y = input_ideal_model(fit=True,negative=True)[2]
lerror_y = input_log_model(fit=True,negative=True)[2]

axs[2,1].set_title("Residuals")
axs[2,1].axhline(y = 0, linewidth=1, color = 'r', linestyle = '-')
axs[2,1].errorbar(-1*negvoltage, input_power_model(fit=True,negative=True)[0], 
                  xerr = negv_unc, yerr=perror_y, label='power',
                  fmt='o', markersize=3, color='blue', 
                  ecolor='blue', linewidth=0.5)
axs[2,1].errorbar(-1*negvoltage, input_ideal_model(fit=True,negative=True)[0], 
                  xerr = negv_unc, yerr=ierror_y, label='ideal',
                  fmt='o', markersize=3, color='orange', 
                  ecolor='orange', linewidth=0.5)
axs[2,1].errorbar(-1*negvoltage, input_log_model(fit=True,negative=True)[0], 
                  xerr = log_v_unc, yerr=log_c_unc, label='log',
                  fmt='o', markersize=3, color='green', 
                  ecolor='green', linewidth=0.5)
axs[2,1].set_xlabel("Voltage(V)")
axs[2,1].set_ylabel("Error (mA)")
axs[2,1].legend(loc='lower right')

print("POWER MODEL(negative):")
print("The reduced chi squared in the power model is", input_power_model(fit=True,negative=True)[1])        
pstd = input_power_model(negative=True)[3] #Getting parameter uncertainties
print("a = ", input_power_model(negative=True)[1], "+-", pstd[0])
print("b = ", input_power_model(negative=True)[2], "+-", pstd[1])

print("IDEAL MODEL(negative):")
print("The reduced chi squared in the power model is", input_ideal_model(fit=True,negative=True)[1])        
istd = input_ideal_model(negative=True)[2]
print("a = ", input_ideal_model(negative=True)[1], "+-", istd[0])

print("LOG MODEL(negative):")
print("The reduced chi squared in the log model is", input_log_model(fit=True,negative=True)[1])
log_std = input_log_model(negative=True)[3]
print("a = ", input_log_model(negative=True)[1], "+-", log_std[0])
print("b = ", input_log_model(negative=True)[2], "+-", log_std[1])
plt.show()

###Plot of the raw data
def show_raw():
    plt.title("Voltage current plot(raw)")
    plt.errorbar(data[1], data[2], 
                 xerr= data[3], yerr=data[4], label= "observed",
                 fmt='o', markersize=1, 
                 color='black', ecolor='black', linewidth=0.5)
    plt.xlabel("Voltage(V)")
    plt.xticks(np.arange(-5,13.1, step=1))
    plt.ylabel("Current(mA)")
    plt.legend()
    plt.grid()
    plt.show()