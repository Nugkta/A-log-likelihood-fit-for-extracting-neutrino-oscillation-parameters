# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:58:20 2022

@author: pokey
"""
import matplotlib.pyplot as plt
import numpy as np
import math


#setting plotting defaults

default = {'font.size': 15,
          'figure.figsize': (10, 6)}
plt.rcParams.update(default) 

#%% First plotting the histogram of the datra

#data_oe stores the experimental data with oscillation
data_oe = np.loadtxt('data1.txt', skiprows = 2, max_rows = 200) # 200 data points


#data_us stores the unoscillated simulated data
data_us = np.loadtxt('data1.txt', skiprows = 205, max_rows = 200)


#e_list is the coreesponding energy level of the data
e_list = np.linspace(0.025,9.975,200)



#%% plotting
plt.bar(e_list,data_oe, width = .06)
plt.xlabel('energy (GeV)')
plt.ylabel('number of incidence')
plt.title('histogram of muons at different energy levels')



#%% Coding the prob density function P

#m_23_2 denotes the difference of square mass difference
def pdf(E, L, theta_23, m_23_2):
    #val = 1 - (np.sin(2 * theta_23))**2 * (np.sin(1.267 * m_23_2 * L / E))**2
    x1 = np.sin(2 * theta_23)
    x2 = np.sin(1.267 * m_23_2 * L / E)
    val = 1 - x1*x1 * x2*x2
    return val


#%% Testing the funtion with some energy E

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



#%%




















