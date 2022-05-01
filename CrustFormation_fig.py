#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 11:52:40 2022

@author: linecolin
"""
import numpy as np
import matplotlib.pyplot as plt
import CrustFormation as cf

cf.default_settings()

n = 50

Y = np.linspace(2, 1, n)

param_test1 = {'k':2, 'rho':3E3, 'Cp':1000, 'Lat': 4E5, 'C_0': 0.1, 'C_E':0.7, 
          'L0':1000, 'dR_bot0':-1E-8, 'h0':25E-12, 'lambda':0, 'D_AN':1E-2}
param_test2 = {'k':2, 'rho':3E3, 'Cp':1000, 'Lat': 4E5, 'C_0': 0.1, 'C_E':0.7, 
          'L0':1000, 'dR_bot0':-1E-8, 'h0':25E-12, 'lambda':np.abs(np.log(21/25)/300E6), 'D_AN':1}
param_test3 = {'k':2, 'rho':3E3, 'Cp':1000, 'Lat': 4E5, 'C_0': 0.1, 'C_E':0.7, 
          'L0':1000, 'dR_bot0':-1E-8, 'h0':25E-12, 'lambda':np.abs(np.log(21/25)/300E6), 'D_AN':1E-2}
param_test4 = {'k':2, 'rho':3E3, 'Cp':1000, 'Lat': 4E5, 'C_0': 0.1, 'C_E':0.7, 
          'L0':1000, 'dR_bot0':-1E-8, 'h0':25E-12, 'lambda':np.abs(np.log(21/25)/300E6), 'D_AN':1E-3}
param_test5 = {'k':2, 'rho':3E3, 'Cp':1000, 'Lat': 4E5, 'C_0': 0.1, 'C_E':0.7, 
          'L0':1000, 'dR_bot0':-1E-8, 'h0':0, 'lambda':0, 'D_AN':0}

data_test1 = cf.Evolution(n, param_test1)
data_test2 = cf.Evolution(n, param_test2)
data_test3 = cf.Evolution(n, param_test3)
data_test4 = cf.Evolution(n, param_test4)
data_test5 = cf.Evolution(n, param_test5)

plt.figure()
plt.plot(data_test1['time'], data_test1['r CRUST'], 'blue')
plt.plot(data_test1['time'], data_test1['r CUMULATS'], 'slateblue')

plt.plot(data_test2['time'], data_test2['r CRUST'], 'red')
plt.plot(data_test2['time'], data_test2['r CUMULATS'], 'brown')

plt.plot(data_test3['time'], data_test3['r CRUST'], 'orange')
plt.plot(data_test3['time'], data_test3['r CUMULATS'], 'goldenrod')

plt.plot(data_test4['time'], data_test4['r CRUST'], 'darkorchid')
plt.plot(data_test4['time'], data_test4['r CUMULATS'], 'violet')

plt.plot(data_test5['time'], data_test5['r CRUST'], 'green')
plt.plot(data_test5['time'], data_test5['r CUMULATS'], 'darkgreen')

plt.ylabel('radius [km]')
plt.xlabel('time [My]')
plt.grid(color='gray', linestyle='--', linewidth=0.75)
plt.savefig('radius.png')
plt.show()

plt.figure()
plt.plot(data_test1['time'], data_test1['H TOT'], 'blue')
plt.plot(data_test2['time'], data_test2['H TOT'], 'red')
plt.plot(data_test3['time'], data_test3['H TOT'], 'orange')
plt.plot(data_test4['time'], data_test4['H TOT'], 'darkorchid')
plt.plot(data_test5['time'], data_test5['H TOT'], 'green')
plt.ylabel('H TOT [W]')
plt.xlabel('time [My]')
plt.grid(color='gray', linestyle='--', linewidth=0.75)
plt.savefig('H_TOT.png')
plt.show()

plt.figure()
plt.plot(data_test1['time'], data_test1['h LMO'], 'blue')
plt.plot(data_test2['time'], data_test2['h LMO'], 'red')
plt.plot(data_test3['time'], data_test3['h LMO'], 'orange')
plt.plot(data_test4['time'], data_test4['h LMO'], 'darkorchid')
plt.plot(data_test5['time'], data_test5['h LMO'], 'green')
plt.ylabel('h LMO $[W m^{-3}]$')
plt.xlabel('time [My]')
plt.grid(color='gray', linestyle='--', linewidth=0.75)
plt.savefig('h_LMO.png')
plt.show()

plt.figure()
plt.plot(data_test1['time'], data_test1['VOLUME TOT'], 'blue')
plt.plot(data_test2['time'], data_test2['VOLUME TOT'], 'red')
plt.plot(data_test3['time'], data_test3['VOLUME TOT'], 'orange')
plt.plot(data_test4['time'], data_test4['VOLUME TOT'], 'darkorchid')
plt.plot(data_test5['time'], data_test5['VOLUME TOT'], 'green')
plt.ylabel('VOLUME TOT $[m^{-3}]$')
plt.xlabel('time [My]')
plt.grid(color='gray', linestyle='--', linewidth=0.75)
plt.savefig('V_TOT.png')
plt.show()


plt.figure(figsize = (14, 7))
plt.plot(data_test1['Temperature'], Y, 'blue')
plt.plot(data_test2['Temperature'], Y, 'red')
plt.plot(data_test3['Temperature'], Y, 'orange')
plt.plot(data_test4['Temperature'], Y, 'darkorchid')
plt.plot(data_test5['Temperature'], Y, 'green')
plt.ylabel('radius')
plt.xlabel('Temperature, $t=t_{I}$')
plt.grid(color='gray', linestyle='--', linewidth=0.75)
plt.savefig('T.png')
plt.show()

plt.figure()
plt.plot(data_test1['h(r, t=tf)'], Y, 'blue')
#plt.plot(data_test2['h(r, t=tf)'], Y, 'red')
plt.plot(data_test3['h(r, t=tf)'], Y, 'orange')
plt.plot(data_test4['h(r, t=tf)'], Y, 'darkorchid')
plt.ylabel('radius')
plt.xlabel('h(r) CRUST, $t=t_{I}$')
plt.grid(color='gray', linestyle='--', linewidth=0.75)
plt.savefig('h_r.png')
plt.show()