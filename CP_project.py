# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:58:20 2022

@author: pokey
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import ticker, cm

#setting plotting defaults

default = {'font.size': 15,
          'figure.figsize': (10, 6)}
plt.rcParams.update(default) 

#%% First plotting the histogram of the datra

#data_oe stores the experimental data with oscillation
#data_oe = np.loadtxt('data1.txt', skiprows = 2, max_rows = 200) # 200 data points
data_oe = np.loadtxt('data.txt', skiprows = 2, max_rows = 200) # 200 data points


#data_us stores the unoscillated simulated data
#data_us = np.loadtxt('data1.txt', skiprows = 205, max_rows = 200)
data_us = np.loadtxt('data.txt', skiprows = 205, max_rows = 200)

#e_list is the coreesponding energy level of the data
e_list = np.linspace(0.025,9.975,200)

#%% plotting
#plotting
plt.bar(e_list,data_oe, width = .06)
plt.xlabel('energy (GeV)')
plt.ylabel('number of incidence')
plt.title('histogram of muons at different energy levels')


#%% Coding the prob density function P

# Coding the prob density function P

#m_23_2 denotes the difference of square mass difference
def pdf(E, L, theta_23, m_23_2):
    #val = 1 - (np.sin(2 * theta_23))**2 * (np.sin(1.267 * m_23_2 * L / E))**2
    x1 = np.sin(2 * theta_23)
    x2 = np.sin(1.267 * m_23_2 * L / E)
    #print('x2 is', np.shape(x2))
    val = 1 - x1*x1* x2*x2
    #print(np.shape(x1))
    return val

#%% Testing the funtion with some energy E

# Testing the funtion with some energy E

#using given values of parameters
theta_23 = np.pi / 4
m_23_2 = 2.4e-3
L = 295
eV = 1.6e-19

e_list = np.linspace(00.025,9.975,200)
prob_e = pdf(e_list, L = L, theta_23 = theta_23, m_23_2 = m_23_2)
plt.plot(e_list, prob_e,'.')
plt.xlabel('energy (GeV)')
plt.ylabel('PDF')
plt.title('the PDF p(E), equation 1')
plt.show()


#%%

# plot the unoscillatory simulated data
plt.bar(e_list,data_us, width = .06)
plt.xlabel('energy (GeV)')
plt.ylabel('# of muons')
plt.title('unosci_simu data vs. energy')
plt.show()


#%%
#combine the PDF with the unosci_simudata
data_os = prob_e * data_us
plt.plot(e_list,data_os, 'r-', label = 'osci_simu data')
plt.xlabel('energy (GeV)')
plt.ylabel('# of muons')
plt.title('the osci_simu data vs. energy')

#adding the real data for comparison
plt.bar(e_list,data_oe, width = .06, label = 'exp data')

plt.legend()
plt.show()

#%% Writing NLL
m = data_oe

theta_23 = np.pi / 4
m_23_2 = 2.4e-3
L = 295


e_list = np.linspace(0.025,9.975,200)

# The lambda func
# calculating expected osci_simu data at each energy interval
# u:the parameters to fit
# var_fit : tell the function to intake which variable 
def lamb(u, var_fit, theta_23 = theta_23, E = e_list, m_23_2 = m_23_2):
    '''
    This function takes in parameters and output the corresponding ocsi_simu data(expected values)
    '''
    if var_fit == 'theta_23':
        PDF = pdf(theta_23 = u, E = e_list, L = L,  m_23_2 = m_23_2 )

    elif var_fit == 'th&m': # fitting both theta 23 and m_23_2
        PDF = pdf(theta_23 = u[0], E = e_list, L = L,  m_23_2 = u[1] )
    
    
    else: 
        print('please input correct variable name')
    osci_simu = PDF * data_us
    #PDF = lambda theta_23: pdf(theta_23, E = e_list, L = L,  m_23_2 = m_23_2) * data_us
    # osci_simu = PDF * data_us
    return osci_simu




# the NLL function
# did not include the factorial term
def NLL(u, m, var_fit):   # u is the parameters for minimisation, m is the experimental data
    '''
    the negative log likihood function
    takes in vector of parameters, the experimental values and the names of variables to fit
    returns the negative log likelihood
    '''
    val = sum( lamb(u, var_fit) - m * np.log(lamb(u, var_fit)) )
    return val


#%%
NLL(theta_23, data_oe, 'theta_23')


#%% plotting NLL to spot the approximate minimum
t_list = np.linspace(0,np.pi/2,10000) # the oscillation is periodic for pi/2
NLL_list = []
for i in range(10000):
    NLL_list.append(NLL(t_list[i], m, 'theta_23'))
plt.plot(t_list, NLL_list)
plt.title(r"NLL vs. $\theta_{23}$")
plt.ylabel('NLL')
plt.xlabel(r' $\theta_{23}$ ')
plt.show()


#%%

def new_point_para(init_points, func):
    '''
    The function for finding the minimum point of x3 with given points x0, x1, x2, using parabolic method.
    Input func as the name of the function to find the minimum point.
    Note that the func can only have one parameter for this 1D fitting update function, so the NLL need to be preprocessed to intake only 1 param.

    '''
    x0, x1, x2 = init_points


    # if func is NLL: # Because the NLL function needs extra input of experiment data and the name of variable to fit
    #     numer = (x2*x2 - x1*x1) * func(x0, m, 'theta_23') + (x0*x0 - x2*x2) * func(x1, m, 'theta_23') + (x1*x1 - x0*x0) * func(x2, m, 'theta_23')
    #     denum = (x2 - x1) * func(x0, m, 'theta_23') + (x0 - x2) * func(x1, m, 'theta_23') + (x1 - x0) * func(x2, m, 'theta_23')
    #     x3 = .5 * numer / denum
    #     return x3

     # other general function
    numer = (x2*x2 - x1*x1) * func(x0) + (x0*x0 - x2*x2) * func(x1) + (x1*x1 - x0*x0) * func(x2)
    denum = (x2 - x1) * func(x0) + (x0 - x2) * func(x1) + (x1 - x0) * func(x2)
    x3 = .5 * numer / denum
    return x3
        



def minimise_para(func, init_guess):
    '''
    This is the 1D parabolic minimization function
    It takes in the name of the function(of only one parameter) and a set of initial guess as a list.
    Outputs the value of the parameter tjhat minimizes the function
    '''
    
    x0, x1, x2 = init_guess
    val_list = [func(x0), func(x1), func(x2)]
    #print(111111,max(val_list) - min(val_list))
    while max(val_list) - min(val_list) > 1e-4: # convergin criterion is 1e-4
        x3 = new_point_para(init_guess, func)
        init_guess.append(x3)
        val_list.append(func(x3))
        posi_max = val_list.index(max(val_list))
        init_guess.pop(posi_max)
        val_list.pop(posi_max)
    return init_guess[2]

NLL_1D_theta_23 = lambda theta_23: NLL(theta_23, data_oe, 'theta_23') # only consider NLL of 1D minimization, parameter to fit is theta_23, and the experimental data is the given one.

#%% The first minimum
min1 = minimise_para(NLL_1D_theta_23, [0.5, .6, .78])

# The second minimum
# min2 = minimise_para(NLL_1D_theta_23, [0.8, .85, .9])

# print('the first minimum is', min1,'\n'
#       ,'the second minimum is',  min2)
print('the minimum is ', min1 )

plt.plot(t_list, NLL_list, label = 'NLL curve')
plt.title(r"NLL vs. $\theta_{23}$")
plt.ylabel('NLL')
plt.xlabel(r' $\theta_{23}$ ')

f_min1 = NLL_1D_theta_23(min1)
# f_min2 = NLL_1D_theta_23(min2)

plt.plot(min1, f_min1,'rx', label = 'min')
# plt.plot(min2, f_min2, 'gx', label = 'second min')

plt.legend()
plt.show()


#%% 3.5 Finding the accuracy of the test

f_min1_p = f_min1 + 0.5
# f_min1_n = f_min1 - 0.5 

# we know the y values, now we need a function to trace back the value of x(theta_23)

def trace_x(y_list, x_list, y_val):
    '''
    trace the value of x given a function value y
    Takes in the list of function values, list of x values and the given value y to be traced.
    Returns the x value corresponding to the given y
    '''
    
    diff_list = abs(y_list - np.ones(len(y_list)) * y_val)
    
    # print(22222222, min(diff_list))
    
    ind = np.where(diff_list == min(diff_list))
    #print(11111111, ind)
    return x_list[ind]


# Test if the tracing function is working corrctly

x = trace_x(NLL_list, t_list, f_min1_p)

plt.plot(t_list, NLL_list, label = 'NLL curve')
plt.plot(x[0], f_min1_p, 'ro')
plt.plot(x[0], NLL_1D_theta_23(x[1]), 'ro', label = 'smaller edge for SD')

plt.plot(x[1], f_min1_p, 'go')
plt.plot(x[1], NLL_1D_theta_23(x[1]), 'go', label = 'larger edge for SD')

plt.legend()
plt.show()
#plt.plot(x,  f_min1_p)


#%%




def find_sd(func, x_list, x_val):
    y_list = []
    
    for i in x_list:
        y_list.append(func(i))
    # print(y_list)
    # y_list = func(x_list)
    
    
    y_val = func(x_val)
    
    y_val_p = y_val + 0.5
    #y_val_n = y_val - 0.5
    
    x_p = trace_x(y_list, x_list, y_val_p)
    # x_p = [i for i in x_p_t if i > x_val]
    
    # x_n_t = trace_x(y_list, x_list, y_val_n)
    # x_n = [i for i in x_n_t if i < x_val]
    
    # print(11111, x_p)
    # print(22222, x_n)
    # print(33333, x_val)
    return (x_p[1] - x_p[0]) / 2, x_p

sd_test,  x_p = find_sd(NLL_1D_theta_23, t_list, min1)

plt.plot(t_list, NLL_list, label = 'NLL curve')
plt.plot(x_p[0], NLL_1D_theta_23(x_p[0]), 'rx', label = 'left edge for SD')
plt.plot(x_p[1], NLL_1D_theta_23(x_p[1]), 'gx', label = 'right edge for SD')

plt.plot(min1,  NLL_1D_theta_23(min1), 'bx', label = 'minimum')

plt.legend()

plt.xlim([0.74, 0.83])
plt.ylim([63.5, 65.5])
plt.title(' The zoomed in plot at the minimum of NLL')

plt.ylabel('magnitude of NLL')
plt.xlabel(r'magnitude of $\theta_{23}$')

plt.show()


print('the standard deviation is ', sd_test)
print(r'the minimum is at theta_23 = ', min1)


#%% Finding the curvature around the minimum ???

def minimise_para_cur(func, init_guess):
    '''
    This is the revised version of 1D parabolic minimization function
    it returns the last three points that converges
    '''
    
    x0, x1, x2 = init_guess
    val_list = [func(x0), func(x1), func(x2)]
    #print(111111,max(val_list) - min(val_list))
    while max(val_list) - min(val_list) > 1e-4: # convergin criterion is 1e-4
        x3 = new_point_para(init_guess, func)
        init_guess.append(x3)
        val_list.append(func(x3))
        posi_max = val_list.index(max(val_list))
        init_guess.pop(posi_max)
        val_list.pop(posi_max)
    return init_guess



points = minimise_para_cur(NLL_1D_theta_23, [0.5, .6, .78])
points.sort()

p0, p1, p2 = points

f0 = NLL_1D_theta_23(p0)
f1 = NLL_1D_theta_23(p1)
f2 = NLL_1D_theta_23(p2)

d0 = (f1 - f0) / (p1 - p0)
d1 = (f2 - f1) / (p2 - p1)

cur = 2 * (d1 - d0) / (p2-p0)

err = np.sqrt(1 / cur)
print(err)


#%% Finding the curvature again using numerical calculus
f0 = NLL_1D_theta_23(min1 - 1e-4)
f1 = NLL_1D_theta_23(min1)
f2 = NLL_1D_theta_23(min1 + 1e-4)

cur = (f2- 2*f1 + f0) / (2 * 1e-4)**2 

err = np.sqrt(1 / cur)
print('the error is estimated by curvature to be:', err)


 
#%% 4. Two-dimensional minimisation

# 4.1 The univariate method



#%%

# First trying to plot the 2D NLL of m_23_2 and theta_23

# thm_list = np.linspace((0.025, 0.001), (9.975, 0.029), 1000)
# thm_list = np.zeros((10000,2))
# th_list = np.linspace(.025, 9.975, 100)
# m_list = np.linspace(.001, .029, 100)

# for i in range(len(th_list)):
#     for j in range(len( m_list)):
#         thm_list[100 * i + j] = np.array([th_list[i],m_list[j]])


# #%%

# NLL_2D_list = []

# for i in thm_list:
#     val = NLL(i, data_oe, 'th&m')
#     NLL_2D_list.append(val)
    

#%%

# getting the data for the contour map
N = 100

m_list = np.linspace(.001, .005, N)           
th_list = np.linspace(.55, 1, N)       

       
X, Y = np.meshgrid(m_list, th_list)
Z_t = np.ones((N,N))

Z = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        #print([th_list[i], m_list[j]])
        Z[j,i] = NLL([th_list[j], m_list[i]], data_oe, 'th&m')
        # Note that the index and coordinate are inver in order
        
# plotting the contour map      
        
       
cs = plt.contourf(X, Y, Z, 15 , 
                  #hatches =['-', '/','\\', '//'],
                  cmap ='Greens')
plt.locator_params(axis='both', nbins=6)
cbar = plt.colorbar(cs, label = 'Magnitude of NLL')
plt.xlabel(r'$\Delta m_{23}^2$')
plt.ylabel(r'$\theta_{23}$')
plt.title(r'NLL vs. $\Delta m_{23}^2$ and $\theta_{23}$ ')





















#%%
# plt.scatter(X , Y, Z, cmap = 'bwr_r')





#%%

# plt.contour([thm_list[:,0],thm_list[: , 1 ],] NLL_2D_list, cmap = 'bwr_r')
















# #%%
# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import ma
# from matplotlib import ticker, cm
# N = 1000
# x = np.linspace(-6.0, 6.0, N)
# y = np.linspace(-7.0, 7.0, N)
# X, Y = np.meshgrid(x, y)
   
# Z1 = np.exp(X * Y)
# z = 50 * Z1
# z[:5, :5] = -1
# z = ma.masked_where(z <= 0, z)
   
# cs = plt.contourf(X, Y, z,
#                   locator = ticker.LogLocator(),
#                   cmap ="bone")
  
# cbar = plt.colorbar(cs)
  
# plt.title('matplotlib.pyplot.contourf() Example')
# plt.show()



















