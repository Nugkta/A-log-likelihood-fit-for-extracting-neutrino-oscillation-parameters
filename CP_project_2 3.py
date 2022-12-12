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
          'figure.figsize': (8,6)}
plt.rcParams.update(default) 

#%% First plotting the histogram of the datra

#data_oe stores the experimental data with oscillation

data_oe = np.loadtxt('data.txt', skiprows = 2, max_rows = 200) # 200 data points


#data_us stores the unoscillated simulated data

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

    elif var_fit == 'th&m': # changiong both theta 23 and m_23_2
        PDF = pdf(theta_23 = u[0], E = e_list, L = L,  m_23_2 = u[1] )
    elif var_fit == 'm_23_2':
        #print(u, type(u))
        PDF = pdf( m_23_2 = u, theta_23 = theta_23, E = e_list, L = L)
    
    else: 
        print('please input correct variable name')
    osci_simu = PDF * data_us
    #PDF = lambda theta_23: pdf(theta_23, E = e_list, L = L,  m_23_2 = m_23_2) * data_us
    # osci_simu = PDF * data_us
    return osci_simu




# the NLL function
# did not include the factorial term
def NLL(u, m, var_fit, theta_23 = theta_23, E = e_list, m_23_2 = m_23_2):   # u is the parameters for minimisation, m is the experimental data
    '''
    the negative log likihood function
    takes in vector of parameters, the experimental values and the names of variables to fit
    returns the negative log likelihood
    '''
    val = sum( lamb(u, var_fit, theta_23 = theta_23, E = e_list, m_23_2 = m_23_2) - m * np.log(lamb(u, var_fit, theta_23 = theta_23, E = e_list, m_23_2 = m_23_2)) )
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
        



def minimise_para(func, init_guess_single):
    '''
    This is the 1D parabolic minimization function
    It takes in the name of the function(of only one parameter) and a set of initial guess as a list.
    Outputs the value of the parameter tjhat minimizes the function
    '''
    
    x0, x1, x2 = init_guess_single, init_guess_single + .05, init_guess_single -.05
    init_guess = [x0, x1, x2]
    
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

NLL_1D_theta_23 = lambda theta_23: NLL(theta_23, data_oe, 'theta_23', m_23_2 = m_23_2) # only consider NLL of 1D minimization, parameter to fit is theta_23, and the experimental data is the given one.

#%% The first minimum
#%%time
min1 = minimise_para(NLL_1D_theta_23, .6)


#%%
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
#%%
print('the NLL_min = ',NLL_1D_theta_23(min1))

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
    '''
    Function for finding the standard deviation of the function around the minimum point
    
    '''
    
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

def minimise_para_cur(func, init_guess_single):
    '''
    This is the revised version of 1D parabolic minimization function
    it returns the last three points that converges
    '''
    
    x0, x1, x2 = init_guess_single, init_guess_single+ 1e-5, init_guess_single - 1e-5
    init_guess = [x0, x1, x2]
    val_list = [func(x0), func(x1), func(x2)]
    #print(111111,max(val_list) - min(val_list))
    while max(val_list) - min(val_list) > 1e-4: # convergin criterion is 1e-4
        x3 = new_point_para(init_guess, func)
        # print(1111111111111, x3)
        init_guess.append(x3)
        val_list.append(func(x3))
        # print(3333333333333, val_list)
        posi_max = val_list.index(max(val_list))
        init_guess.pop(posi_max)
        val_list.pop(posi_max)
        # print(222222222222, max(val_list) - min(val_list))
    return init_guess[2], init_guess



val, points = minimise_para_cur(NLL_1D_theta_23, .6)
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
# Before choosing which variable to minimize, first plotting the 2D contour map
# to observe the shape

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
plt.show()

#%% 4.1 The univariate method
'''
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------




4.1 THEUNIVARIATE METHOD -------------------------------------------------------------

---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
'''
#%%
#%%time 


d = 5
# ig = [.65, .003076]     #[theta_23, m_23]
# ig = [.0, .002]

ig = [.9, .002] 

path_x = [ig[0]]
path_y = [ig[1]]
while d > 1e-5:
    NLL_1D_m = lambda m_23_2: NLL(m_23_2, var_fit = 'm_23_2' ,m = data_oe ,theta_23 = ig[0])
    m_e, m_e_list = minimise_para_cur(NLL_1D_m, ig[1])
    # print(1111111111)
    NLL_m_temp = NLL_1D_m(m_e)
    ig[1] = m_e
    path_x.append(ig[0])
    path_y.append(ig[1])
    
    NLL_1D_t = lambda theta_23: NLL(theta_23, m = data_oe, var_fit = 'theta_23', m_23_2 = ig[1])
    t_e, t_e_list = minimise_para_cur(NLL_1D_t, ig[0])
    # print(222222222 )
    ig = [t_e, m_e]
    d =  NLL_m_temp - NLL_1D_t(t_e)
    path_x.append(ig[0])
    path_y.append(ig[1])
    #print(d)


#%% Finding the accuracy using the curvature
def find_err_cur(points, func):
    p0, p1, p2 = points
    
    f0 = func(p0)
    f1 = func(p1)
    f2 = func(p2)
    
    d0 = (f1 - f0) / (p1 - p0)
    d1 = (f2 - f1) / (p2 - p1)
    
    cur = 2 * (d1 - d0) / (p2-p0)
    
    err = np.sqrt(1 / cur)
    return err

m_err = find_err_cur(m_e_list, NLL_1D_m)
t_err = find_err_cur(t_e_list, NLL_1D_t)

print('Univarite Method: The value of m_23_2 is %.5f +- %.5f'%(m_e, m_err))

print('Univarite Method: The value of theta_23 is %.2f +- %.2f'%(t_e, t_err))

print('Univarite Method: The value of NLL is: ', NLL_1D_t(t_e))


#%% Plotting the path of univariate method

cs = plt.contourf(X, Y, Z, 15 , 
                  #hatches =['-', '/','\\', '//'],
                  cmap ='Greens')
plt.locator_params(axis='both', nbins=6)
cbar = plt.colorbar(cs, label = 'Magnitude of NLL')
plt.xlabel(r'$\Delta m_{23}^2$')
plt.ylabel(r'$\theta_{23}$')
plt.title(r'NLL vs. $\Delta m_{23}^2$ and $\theta_{23}$ (Univariate)')

plt.plot(path_y,path_x,  'r-', label = 'the convergent path')
plt.legend()

plt.locator_params(axis='both', nbins=6)
plt.show()

#%%
path_y_u,path_x_u = path_y,path_x




#%% Plot a Zoomed in plot of path of univariate

cs = plt.contourf(X, Y, Z, 15 , 
                  #hatches =['-', '/','\\', '//'],
                  cmap ='Greens')
plt.locator_params(axis='both', nbins=6)
cbar = plt.colorbar(cs, label = 'Magnitude of NLL')
plt.xlabel(r'$\Delta m_{23}^2$')

plt.ylabel(r'$\theta_{23}$')
plt.title(r'NLL vs. $\Delta m_{23}^2$ and $\theta_{23}$ Zoomed (Univariate)')


plt.plot(path_y,path_x,  'r-', label = 'Univariate path')

plt.xlim([0.0022, 0.00235])
plt.ylim([0.79, 0.81])
plt.legend()

plt.locator_params(axis='both', nbins=4)

plt.show()



#%% 4.2 Simultaneous minimisation

'''
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------

4.2  NEWTON'S METHOD---------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------




'''



# Finding the gradient of a function numerically 
def gradient(f, x, h0, h1):
    '''
    Function for finding the gradient of a function numerically
    func: the function to find gradient for
    x: the vector input of the function
    h0: the step of the first variable
    h1: the step of the second variable
    
    '''
    var0 = x[0]
    var1= x[1]
    g1 = (f([var0 + h0, var1]) - f([var0, var1])) / h0
    g2 = (f([var0, var1 + h1]) - f([var0, var1])) / h1
    grad = np.array([g1,g2])
    return grad
# Findig the Hessian matrix of a function
def hessian(f, x, h0, h1):
    var0 = x[0]
    var1= x[1]
    f_xx = ( f([var0 + h0, var1]) - 2 * f([var0, var1]) + f([var0 - h0, var1]) ) / h0**2
    f_yy = ( f([var0, var1 + h1]) - 2 * f([var0, var1]) + f([var0, var1 - h1]) ) / h1**2
    f_xy = (f([x[0] + h0, x[1] + h1]) -f([x[0] + h0, x[1] - h1])- f([x[0] - h0, x[1] + h1]) + f([x[0] - h0, x[1] - h1])) / (4 * h0 * h1)
    H = np.array([[f_xx, f_xy],
                    [f_xy, f_yy]])
    return H

#%% the Newton iteration
def inverse(M):
    det = M[0][0]*M[1][1] - M[1][0]*M[0][1]
    M_t = np.ones([2,2])
    M_t[0][0] = M[1][1]
    M_t[0][1] = -M[0][1]
    M_t[1][0] = -M[1][0]
    M_t[1][1] = M[0][0]
    M_inv = M_t / det
    return M_inv

#%%

def Newton(f, ig, h0, h1):
    x = ig
    diff = 10
    path_th = [ig[0]]
    path_m = [ig[1]]
    while diff > 1e-5:
    #for i in range(it):

        H = hessian(f, x, h0, h1)
        grad =  gradient(f, x, h0, h1)
        x_new = x - inverse(H) @ grad 
        diff = np.linalg.norm(x - x_new)
        path_th.append(x_new[0])
        path_m.append(x_new[1])
        x = x_new
        #print(x)
    return x, path_th, path_m


NLL_2D = lambda u: NLL(u, data_oe, 'th&m')

ig = [0.9, 0.002]
h0 = 1e-5
h1 = 1e-5
#%%
#%%time

u, path_x, path_y = Newton(NLL_2D, ig, h0, h1)

#%%
path_y_n,path_x_n = path_y,path_x

#%% Finding the accuracy of the result using the curvature method

def cur(f, x, var):
    var0 = x[0]
    var1= x[1]
    f_xx = ( f([var0 + h0, var1]) - 2 * f([var0, var1]) + f([var0 - h0, var1]) ) / h0**2
    f_yy = ( f([var0, var1 + h1]) - 2 * f([var0, var1]) + f([var0, var1 - h1]) ) / h1**2
    if var == 'x':
        return f_xx
    elif var == 'y':
        return f_yy
    else:
        print('input correct variable for curvature')

def find_err_cur2(f, x, h0 = 1e-4, h1 = 1e-4, var = None):
    curv = cur(f, x, var)
    err = np.sqrt(1/curv)
    return err

m_err_n = find_err_cur2(NLL_2D, u, h0 = 1e-4, h1 = 1e-4, var = 'y')
th_err_n = find_err_cur2(NLL_2D, u, h0 = 1e-4, h1 = 1e-4, var = 'x')



print('Newton Method: The value of m_23_2 is %.5f +- %.5f'%(u[1], m_err))

print('Newton Method: The value of theta_23 is %.2f +- %.2f'%(u[0], t_err))

print('Newton Method: The value of NLL is: ', NLL_2D(u))

#%% Plotting the path of Newton method
cs = plt.contourf(X, Y, Z, 15 , 
                  #hatches =['-', '/','\\', '//'],
                  cmap ='Greens')
plt.locator_params(axis='both', nbins=6)
cbar = plt.colorbar(cs, label = 'Magnitude of NLL')
plt.xlabel(r'$\Delta m_{23}^2$')
plt.ylabel(r'$\theta_{23}$')
plt.title(r'NLL vs. $\Delta m_{23}^2$ and $\theta_{23}$ (Newton) ')

plt.plot(path_y,path_x,  'r-', label = 'the convergent path')
plt.legend()

plt.locator_params(axis='both', nbins=6)
plt.show()



#%% Plot a Zoomed in plot of path of newton

cs = plt.contourf(X, Y, Z, 15 , 
                  #hatches =['-', '/','\\', '//'],
                  cmap ='Greens')
plt.locator_params(axis='both', nbins=6)
cbar = plt.colorbar(cs, label = 'Magnitude of NLL')
plt.xlabel(r'$\Delta m_{23}^2$')

plt.ylabel(r'$\theta_{23}$')
plt.title(r'NLL vs. $\Delta m_{23}^2$ and $\theta_{23}$ Zoomed (Newton)')


plt.plot(path_y,path_x,  'b-', label = 'Newton path')

plt.xlim([0.0022, 0.00235])
plt.ylim([0.79, 0.81])
plt.legend()

plt.locator_params(axis='both', nbins=4)

plt.show()


#%%
prob_e = pdf(e_list, L = L, theta_23 = u[0], m_23_2 = u[1])

data_os_2 = prob_e * data_us
plt.plot(e_list,data_os, 'r-', label = 'osci_simu data') # the previous osillating simulated pdf
plt.plot(e_list,data_os_2, 'y-', label = 'osci_simu data(2D Newton fitted)')
plt.xlabel('energy (GeV)')
plt.ylabel('# of muons')
plt.title('the osci_simu data vs. energy (Using 2D fit (Newton))')

#adding the real data for comparison
plt.bar(e_list,data_oe, width = .06, label = 'exp data')

plt.legend()
plt.show()






#%% 4.3 MONTE-CARLO MINIMISATION
'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------


4.3 The Monte-Carlo minimisation--------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------


'''

#%% The MOnte minimisation function

def mon_car(f, ig, k, T_max, h):
    path_th = [ig[0]]
    path_m = [ig[1]]
    
    for T in range(T_max, 0, -1):
        val = f(ig)
        

        
        d0 = np.random.uniform(-h, h) * ig[0] + ig[0] # the normalised small step h, then times ig to maek sure the scale is consistent
        d1 = np.random.uniform(-h, h) * ig[1] + ig[1]
        val_new = f([d0, d1])
        diff = val_new - val
        
        if diff <= 0:
            ig[0], ig[1] = d0, d1
            path_th.append(ig[0])
            path_m.append(ig[1])
        else:
            # print(diff, k, T)
            p_acc = np.exp(-diff / (k * T))
            rand = np.random.uniform(0, 1)
            # print(p_acc)
            if rand < p_acc:
                ig[0], ig[1] = d0, d1
                path_th.append(ig[0])
                path_m.append(ig[1])
    return ig, path_th, path_m

#%% Running the monte minimisation

ig = [0.9, 0.002]
k = 1e-5
h = .1
T_max = 1000
#%%
# %%time

u_monte, path_x_monte, path_y_monte = mon_car(NLL_2D, ig, k, T_max, h)


#%%
path_y_m,path_x_m =  path_y_monte,path_x_monte

#%% Plotting the path of monte method
def plt_norm(): 
    '''
    function for plotting the background 2D NLL contour
    '''
    
    cs = plt.contourf(X, Y, Z, 15 , 
                      #hatches =['-', '/','\\', '//'],
                      cmap ='Greens')
    plt.locator_params(axis='both', nbins=6)
    cbar = plt.colorbar(cs, label = 'Magnitude of NLL')
    plt.xlabel(r'$\Delta m_{23}^2$')
    plt.ylabel(r'$\theta_{23}$')




plt_norm()
plt.title(r'NLL vs. $\Delta m_{23}^2$ and $\theta_{23}$ (Monte Carlo)')
plt.plot(path_y_monte,path_x_monte,  'r-', label = 'the convergent path')
plt.legend()
plt.locator_params(axis='both', nbins=6)
plt.show()


#%% Plot a Zoomed in plot of path of monte
def plt_zoom():
    '''
    function for plotting the zoomed version of background 2D NLL contour
    '''
    
    cs = plt.contourf(X, Y, Z, 15 , 
                      #hatches =['-', '/','\\', '//'],
                      cmap ='Greens')
    plt.locator_params(axis='both', nbins=6)
    cbar = plt.colorbar(cs, label = 'Magnitude of NLL')
    plt.xlabel(r'$\Delta m_{23}^2$')
    
    plt.ylabel(r'$\theta_{23}$')


plt_zoom()
plt.title(r'NLL vs. $\Delta m_{23}^2$ and $\theta_{23}$ Zoomed (Monte)')

plt.plot(path_y_monte,path_x_monte,  'y-', label = 'Monte path')

plt.xlim([0.0022, 0.00235])
plt.ylim([0.76, 0.86])
plt.legend()

plt.locator_params(axis='both', nbins=4)

plt.show()




#%%


m_err_n = find_err_cur2(NLL_2D, u_monte, h0 = 1e-4, h1 = 1e-4, var = 'y')
th_err_n = find_err_cur2(NLL_2D, u_monte, h0 = 1e-4, h1 = 1e-4, var = 'x')



print('Monte Method: The value of m_23_2 is %.5f +- %.5f'%(u_monte[1], m_err_n))

print('Monte Method: The value of theta_23 is %.2f +- %.2f'%(u_monte[0], th_err_n))

print('Monte Method: The value of NLL is: ', NLL_2D(u_monte))






#%% 3D minimisation

'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------


5. 3D Minimisation--------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------

'''



#%% 3D minimisation

'''

---------------------------------------------------------------------------------------------------------


5.1 Newton 3D Minimisation--------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------


'''


#%% rewriting the lambda function and NLL function

a = 10 # random number for default value

def lamb_3D(u, var_fit, theta_23 = theta_23, E = e_list, m_23_2 = m_23_2):
    '''
    This function takes in parameters and output the corresponding ocsi_simu data(expected values)
    '''
    if var_fit == 'theta_23':
        PDF = pdf(theta_23 = u, E = e_list, L = L,  m_23_2 = m_23_2 )

    elif var_fit == 'th&m' or 'th&m&a': # changiong both theta 23 and m_23_2
        # print(u)
        PDF = pdf(theta_23 = u[0], E = e_list, L = L,  m_23_2 = u[1] )
    elif var_fit == 'm_23_2':
        #print(u, type(u))
        PDF = pdf( m_23_2 = u, theta_23 = theta_23, E = e_list, L = L)
    
    else: 
        print('please input correct variable name')
    a = u[2]
    # print(a, 11111111111111)
    osci_simu = PDF * data_us
    #PDF = lambda theta_23: pdf(theta_23, E = e_list, L = L,  m_23_2 = m_23_2) * data_us
    # osci_simu = PDF * data_us
    # plt.plot(e_list, osci_simu * a * E)
    # plt.show()
    return osci_simu * a * E
    # return osci_simu 


def NLL_new(u, m, var_fit, theta_23 = theta_23, E = e_list, m_23_2 = m_23_2):   # u is the parameters for minimisation, m is the experimental data
    '''
    the negative log likihood function
    takes in vector of parameters, the experimental values and the names of variables to fit
    returns the negative log likelihood
    '''
    # print(u, 2222222222222)
    # print(lamb_3D(u, var_fit)[1:3], 1111111111111)
    val = sum( lamb_3D(u, var_fit, theta_23 = theta_23, E = e_list, m_23_2 = m_23_2) 
              - m * np.log(lamb_3D(u, var_fit, theta_23 = theta_23, E = e_list, m_23_2 = m_23_2)))
    return val






#%%

NLL_3D = lambda u: NLL_new(u, m = data_oe, var_fit = 'th&m&a')
print(NLL_3D([0.9, 0.002, 1]))

alist = np.linspace(1, 2, 100)
NLL_3D_list = []
for i in alist:
    vec = [0.77, 0.0023, i]
    NLL_3D_list.append(NLL_3D(vec))
    
plt.plot(alist, NLL_3D_list)
plt.ylabel('NLL_3D')
plt.xlabel(r'$\alpha$')
plt.title(r'NLL_3D vs. $\alpha$ with fixed $\Delta m _{23}^2$ and $\theta_{23}$')
plt.show()





#%% Newton Minimisation 3D

def hessian_3D(f, x, h):
    
    f_xx = (f([x[0] + h,x[1],x[2]])- 2*f([x[0], x[1],x[2]]) + f([x[0] - h,x[1],x[2]]))/(h*h)
    f_xy =  (f([x[0]+ h, x[1] + h,x[2]]) -f([x[0]+ h, x[1]-h,x[2]])- f([x[0]- h, x[1] + h,x[2]])+f([x[0]- h, x[1]- h,x[2]]))/(4*h*h)
    f_xz =  (f([x[0]+ h, x[1] ,x[2]+h]) -f([x[0]+ h, x[1],x[2]-h])- f([x[0]- h, x[1],x[2]+h])+f([x[0]- h, x[1],x[2]-h]))/(4*h*h)
    
    f_yy = (f([x[0], x[1] + h,x[2]]) -2*f([x[0], x[1],x[2]])+ f([x[0], x[1] - h,x[2]]))/(h*h)   
    f_yz =  (f([x[0], x[1]+ h ,x[2]+h]) -f([x[0], x[1]+h,x[2]-h])- f([x[0], x[1]-h,x[2]+h])+f([x[0], x[1]-h,x[2]-h]))/(4*h*h)
    
    f_zz =(f([x[0], x[1] ,x[2]+h]) -2*f([x[0], x[1],x[2]])+ f([x[0], x[1] ,x[2]-h]))/(h*h)
    
    H = np.array([[f_xx,f_xy,f_xz],
                  [f_xy,f_yy,f_yz],
                  [f_xz,f_yz,f_zz]]) 

    
    return H
    
    
    
def gradient_3D(f, x, h):
    
    '''
    Function for finding the gradient of a function numerically
    func: the function to find gradient for
    x: the vector input of the function
    h0: the step of the first variable
    h1: the step of the second variable
    
    '''
    var0 = x[0]
    var1 = x[1]
    var2 = x[2]
    g0 = (f([var0 + h, var1, var2]) - f([var0, var1, var2])) / h
    g1 = (f([var0, var1 + h, var2]) - f([var0, var1, var2])) / h
    g2 = (f([var0 , var1, var2 + h]) - f([var0, var1, var2])) / h
    
    grad = np.array([g0, g1, g2])
    return grad    

    



def inversion_3D(m):    
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = m.flatten()
    determinant = m1*m5*m9 + m4*m8*m3 + m7*m2*m6 - m1*m6*m8 - m3*m5*m7 - m2*m4*m9  
    return np.array([[m5*m9-m6*m8, m3*m8-m2*m9, m2*m6-m3*m5],
                     [m6*m7-m4*m9, m1*m9-m3*m7, m3*m4-m1*m6],
                     [m4*m8-m5*m7, m2*m7-m1*m8, m1*m5-m2*m4]])/determinant




def Newton_3D(f, ig, h):
    x = ig
    diff = 10
    path_th = [ig[0]]
    path_m = [ig[1]]
    path_a = [ig[2]]
    while diff > 1e-5:
    #for i in range(it):
        # print(x)
        # time.sleep(.1)
        H = hessian_3D(f, x, h)
        # print(H,'HHHHHHHHHHHHHHHHH')
        grad =  gradient_3D(f, x, h)
        x_new = x - inversion_3D(H) @ grad 
        diff = np.linalg.norm(x - x_new)
        path_th.append(x_new[0])
        path_m.append(x_new[1])
        path_a.append(x_new[2])
        x = x_new
        #print(x)
    return x, path_th, path_m, path_a


ig_3D = [0.77, 0.002, 4] # initial guess of a set to be 1
h = 1e-4



#%%
# %%time
u_3D_newton, path_x, path_y, path_z = Newton_3D(NLL_3D, ig_3D, h)

print('the minimum parameters are [a, theta_23, m_23_2] =', u_3D_newton)




#%% FINDING THE ACCURACY
def cur_3D(f, x, var, h):
    # var0 = x[0]
    # var1= x[1]
    f_xx = (f([x[0] + h,x[1],x[2]])- 2*f([x[0], x[1],x[2]]) + f([x[0] - h,x[1],x[2]]))/(h*h)
    f_yy = (f([x[0], x[1] + h,x[2]]) -2*f([x[0], x[1],x[2]])+ f([x[0], x[1] - h,x[2]]))/(h*h)   
    f_zz =(f([x[0], x[1] ,x[2]+h]) -2*f([x[0], x[1],x[2]])+ f([x[0], x[1] ,x[2]-h]))/(h*h)
    if var == 'x':
        return f_xx
    elif var == 'y':
        return f_yy
    elif var == 'z':
        return f_zz
    else:
        print('input correct variable for curvature')

def find_err_cur3(f, x, h = 1e-4, var = None):
    curv = cur_3D(f, x, var, h)
    err = np.sqrt(1/curv)
    return err

th_err_n = find_err_cur3(NLL_3D, u_3D_newton, h = 1e-4, var = 'x')
m_err_n = find_err_cur3(NLL_3D, u_3D_newton, h = 1e-4, var = 'y')
a_err_n = find_err_cur3(NLL_3D, u_3D_newton, h = 1e-4, var = 'z')


print('Newton Method 3D: The value of m_23_2 is %.5f +- %.5f'%(u_3D_newton[1], m_err_n))

print('Newton Method 3D: The value of theta_23 is %.2f +- %.2f'%(u_3D_newton[0], th_err_n))

print('Newton Method 3D: The value of a is %.2f +- %.2f'%(u_3D_newton[2], a_err_n))

print('Newton Method 3D: The value of NLL is: ', NLL_3D(u_3D_newton))



#%% Visualising the 4D data minimisation path
import matplotlib
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), s = 1)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    ax.set_xlabel(r'$\theta_{23}$', fontsize=15, rotation=60)
    ax.set_ylabel(r'$\Delta m_{23}^2$', fontsize=15, rotation=60)
    ax.set_zlabel(r'$\alpha$', fontsize=15, rotation=60)
    # ax.set_title(r'Visualisation of NLL($\theta_{23}$, $\Delta m_{23}^2$, $\alpha$)')
    plt.locator_params(axis='both', nbins=4)

    

density = 25

x = np.linspace(0.55,1,density)     #corresponds to the range of theta
y= np.linspace(0.001,0.005,density)   #corresponds to the range of m
z= np.linspace(1,4,density)

X3,Y3,Z3 = np.meshgrid(x,y,z)
f_list = np.zeros((density,density,density)) # list of values of function
for i in range(density):
    for j in range(density):
        for k in range(density):
            f_list[i][j][k] = NLL_3D([X3[i][j][k], Y3[i][j][k], Z3[i][j][k]])

scatter3d(np.ndarray.flatten(X3),np.ndarray.flatten(Y3),np.ndarray.flatten(Z3), np.ndarray.flatten(f_list), colorsMap='jet')
plt.plot(path_x, path_y, path_z, 'r-',linewidth=3)
plt.plot(path_x[-1], path_y[-1], path_z[-1], 'ro', markersize = 10)
plt.show()

#%%
path_x_n3, path_y_n3, path_z_n3 = path_x, path_y, path_z

#%%
# path_l = len(path_x)
# path_f = []
# for i in range(path_l):
#     print(i)
#     path_f.append(NLL_3D([path_x[i], path_y[i], path_z[i]]))
# scatter3d(path_x, path_y, path_z, path_f, colorsMap='jet')

#%%



#%% Plot the minimized comparison of predicted PDF
font = {
        'size'   : 25}

matplotlib.rc('font', **font)

# 2D parameters fitting
prob_e = pdf(e_list, L = L, theta_23 = u[0], m_23_2 = u[1])
data_os_2 = prob_e * data_us

# 3D parameters fitting
data_os_3 = lamb_3D(u_3D_newton, var_fit = 'th&m&a')
plt.figure(figsize = (18,6))

plt.plot(e_list,data_os, 'r-', label = 'osci_simu data', linewidth = 2.5) # the previous osillating simulated pdf
plt.plot(e_list,data_os_2, 'y-', label = 'osci_simu data(2D Newton fitted)', linewidth = 2.5)
plt.plot(e_list,data_os_3, 'k-', label = 'osci_simu data(3D Newton fitted)', linewidth = 2.5)


plt.xlabel('energy (GeV)')
plt.ylabel('# of muons')
plt.title('the osci_simu data vs. energy (Using 2D fit (Newton))')

#adding the real data for comparison
plt.bar(e_list,data_oe, width = .06, label = 'exp data')

plt.legend()
plt.show()








#%% 

'''

---------------------------------------------------------------------------------------------------------


5.2 Monte-carlo 3D Minimisation--------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------


'''
def mon_car_3D(f, ig, k, T_max, h):
    path_th = [ig[0]]
    path_m = [ig[1]]
    path_a = [ig[2]]
    
    for T in range(T_max, 0, -1):
        val = f(ig)
        

        
        d0 = np.random.uniform(-h, h) * ig[0] + ig[0] # the normalised small step h, then times ig to maek sure the scale is consistent
        d1 = np.random.uniform(-h, h) * ig[1] + ig[1]
        d2 = np.random.uniform(-h, h) * ig[2] + ig[2]
        val_new = f([d0, d1, d2])
        diff = val_new - val
        
        if diff <= 0:
            ig[0], ig[1], ig[2] = d0, d1, d2
            path_th.append(ig[0])
            path_m.append(ig[1])
            path_a.append(ig[2])
        else:
            # print(diff, k, T)
            p_acc = np.exp(-diff / (k * T))
            rand = np.random.uniform(0, 1)
            # print(p_acc)
            if rand < p_acc:
                ig[0], ig[1], ig[2] = d0, d1, d2
                path_th.append(ig[0])
                path_m.append(ig[1])
                path_a.append(ig[2])
    return ig, path_th, path_m, path_a


#%%
# %%time
ig_3D = [0.9, 0.002, 4]
k = 1e-5
h = .1
T_max = 100

u_monte_3D, path_x_monte_3D, path_y_monte_3D, path_z_monte_3D = mon_car_3D(NLL_3D, ig_3D, k, T_max, h)

#%% plotting the function
def path_acc_3D(path_x_monte_3D, path_y_monte_3D, path_z_monte_3D, method_name = ''):
    scatter3d(np.ndarray.flatten(X3),np.ndarray.flatten(Y3),np.ndarray.flatten(Z3), np.ndarray.flatten(f_list), colorsMap='jet') # The background scattering points with
                                                                                                                              # color need not to be changed
    plt.plot(path_x_monte_3D, path_y_monte_3D, path_z_monte_3D, 'r-',linewidth=3, label = 'Monte path')
    plt.plot(path_x_monte_3D[-1], path_y_monte_3D[-1], path_z_monte_3D[-1], 'ro', markersize = 10)
    
    
    
    th_err_n = find_err_cur3(NLL_3D, u_monte_3D, h = 1e-4, var = 'x')
    m_err_n = find_err_cur3(NLL_3D, u_monte_3D, h = 1e-4, var = 'y')
    a_err_n = find_err_cur3(NLL_3D, u_monte_3D, h = 1e-4, var = 'z')
    
    
    print(method_name, ': The value of m_23_2 is %.5f +- %.5f'%(u_monte_3D[1], m_err_n))
    
    print(method_name, ': The value of theta_23 is %.2f +- %.2f'%(u_monte_3D[0], th_err_n))
    
    print(method_name, ': The value of a is %.2f +- %.2f'%(u_monte_3D[2], a_err_n))
    
    print(method_name, ': The value of NLL is: ', NLL_3D(u_monte_3D))

#%%
path_acc_3D(path_x_monte_3D, path_y_monte_3D, path_z_monte_3D, 'Monte Method 3D')


plt.show()



#%%

'''
---------------------------------------------------------------------------------------------------------


Plotting 2D paths together--------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------


'''









#%%

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
font = {
        'size'   : 15}

matplotlib.rc('font', **font)
plt.figure(figsize = (18,6))    
cs = plt.contourf(X, Y, Z, 15 , 
                  #hatches =['-', '/','\\', '//'],
                  cmap ='Greens')
plt.locator_params(axis='both', nbins=6)
cbar = plt.colorbar(cs, label = 'Magnitude of NLL')
plt.xlabel(r'$\Delta m_{23}^2$')
plt.ylabel(r'$\theta_{23}$')
plt.title(r'NLL vs. $\Delta m_{23}^2$ and $\theta_{23}$ ')
plt.show()


plt.plot(path_y_u,path_x_u,  'r-', label = 'Univariate path')
plt.plot(path_y_n,path_x_n,  'b-', label = 'Newton path')
plt.plot(path_y_m,path_x_m,  'y-', label = 'Monte path')
plt.legend()

#%%

cs = plt.contourf(X, Y, Z, 15 , 
                  #hatches =['-', '/','\\', '//'],
                  cmap ='Greens')
plt.locator_params(axis='both', nbins=6)
cbar = plt.colorbar(cs, label = 'Magnitude of NLL')
plt.xlabel(r'$\Delta m_{23}^2$')

plt.ylabel(r'$\theta_{23}$')
plt.title(r'NLL vs. $\Delta m_{23}^2$ and $\theta_{23}$ Zoomed (Univariate)')



plt.xlim([0.0022, 0.00235])
plt.ylim([0.79, 0.81])
plt.legend()

plt.locator_params(axis='both', nbins=4)
plt.plot(path_y_u,path_x_u,  'r-', label = 'univariate path')
plt.plot(path_y_n,path_x_n,  'b-', label = 'newton path')
plt.plot(path_y_m,path_x_m,  'y-', label = 'monte path')
plt.show()



#%%

'''
---------------------------------------------------------------------------------------------------------


Plotting 3D paths together--------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------


'''

#%%
font = {
        'size'   : 14}



matplotlib.rc('font', **font)
path_acc_3D(path_x_monte_3D, path_y_monte_3D, path_z_monte_3D, 'Monte Method 3D')
plt.plot(path_x, path_y, path_z, 'b-',linewidth=3, label = 'Newton path')
plt.plot(path_x[-1], path_y[-1], path_z[-1], 'bo', markersize = 10)
plt.legend()
